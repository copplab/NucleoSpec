from __future__ import annotations  # Allows using class name in its own type hints
import sys
import os
import logging
from typing import Optional, Union, Any
import numpy.typing as npt  # For numpy array type hints

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'lib'))

from flask import Flask, render_template, request, jsonify, session
import numpy as np
import json
import io
from pythoms.molecule import IPMolecule, composition_from_formula, bar_isotope_pattern, gaussian_isotope_pattern
from pythoms.tome import plot_mass_spectrum, resolution, autoresolution, localmax
from pythoms.spectrum import weighted_average
from pythoms.senko_charge_assignment import detect_all_peaks_with_charge
from scipy import signal

try:
    import IsoSpecPy as isospec
    ISOSPEC_AVAILABLE = True
except ImportError:
    ISOSPEC_AVAILABLE = False

ISOTOPE_LIBRARY = 'isospec' if ISOSPEC_AVAILABLE else 'pythoms'

_isotope_pattern_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
_peak_analysis_cache: dict[tuple[float, str, int], dict[str, Any]] = {}
_ISOTOPE_CACHE_MAX_SIZE = 1000

MAX_SILVER = 30
MAX_STRANDS = 3
MAX_COMPLEXES = 3


def to_subscript(n: int | str) -> str:
    subscript_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    return str(n).translate(subscript_map)


def to_superscript(n: int | str) -> str:
    superscript_map = str.maketrans('0123456789+-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻')
    return str(n).translate(superscript_map)


def format_adduct_name(adduct_name: str) -> str:
    if not adduct_name:
        return ''
    import re
    match = re.match(r'^(\d+)(.+)$', adduct_name)
    if match:
        return f"{match.group(2)}{to_subscript(match.group(1))}"
    return adduct_name


class DNASilverAnalyzer:
    """ESI-MS compositional analysis engine for nucleic acid–silver complexes."""

    def __init__(self):
        # Monoisotopic masses (NIST standard)
        self.m_p = 1.007825
        self.mC = 12.000000
        self.mN = 14.003074
        self.mO = 15.994915
        self.mP = 30.973763
        self.mAg = 106.905097

        self.MONOISOTOPIC_MASSES = {
            'H': 1.007825, 'C': 12.000000, 'N': 14.003074,
            'O': 15.994915, 'P': 30.973763, 'S': 31.972071,
            'F': 18.998403, 'Cl': 34.969402, 'Br': 78.918338,
            'I': 126.904473, 'Si': 27.976927, 'Se': 79.916522,
        }

        base_adducts = {
            'H': (1.007825, +1),
            'NH4': (18.033823, +1),
            'Na': (22.989769, +1),
            'Cl': (34.969402, -1),
            'Ag': (106.905097, +1),
        }

        # Generate 1× and 2× variants for each adduct
        self.adducts = {}
        for name, (mass, charge) in base_adducts.items():
            for n in range(1, 3):
                adduct_name = name if n == 1 else f"{n}{name}"
                self.adducts[adduct_name] = (mass * n, charge * n)

        self.custom_adducts: list[dict] = []

        logger.info(f"Adduct library initialized with {len(self.adducts)} base adducts")
        adduct_list = [name for name in self.adducts.keys() if not name.replace('2', '') in ['H', 'Ag']]
        logger.debug(f"Available adducts: {', '.join(sorted(adduct_list))}")

    @staticmethod
    def is_complex_strand_label(strand_type: Optional[str]) -> bool:
        """Check if strand_type indicates complex mode."""
        if strand_type is None:
            return False
        return strand_type in ['strand1', 'strand2', 'complex'] or strand_type.startswith('nd=')

    def determine_composition_type(self, num_ag: int, n0: int,
                                    strand_label: Optional[str] = None,
                                    is_complex: bool = False,
                                    custom_xna: Optional[dict] = None,
                                    conjugate_name: Optional[str] = None,
                                    conjugate_count: int = 0) -> str:
        """Determine composition type consistently across all code paths."""
        if num_ag == 0:
            if conjugate_name and conjugate_count > 0:
                return 'DNA/XNA+Conjugate'
            if custom_xna:
                return 'XNA Only'
            return 'DNA Only'
        if n0 == 0 and (is_complex or self.is_complex_strand_label(strand_label)):
            return 'XNA+Ag ion' if custom_xna else 'DNA+Ag ion'
        return 'nanocluster'

    def adduct_name_to_formula(self, adduct_name: str) -> str:
        """
        Convert adduct name (e.g., '2Cl', '2Na', '2NH4') to chemical formula format (e.g., 'Cl2', 'Na2', 'N2H8').
        This is needed for isotope pattern generation where the formula must be in standard notation.
        """
        import re
        match = re.match(r'^(\d+)(.+)$', adduct_name)
        if match:
            count = int(match.group(1))
            base_name = match.group(2)
            if base_name in ['Cl', 'Na', 'Ag']:
                return f"{base_name}{count}"
            if base_name == 'NH4':
                return f"N{count}H{4*count}"
            return f"{base_name}{count}"
        else:
            return adduct_name

    def load_custom_adducts(self) -> list[dict]:
        """Load custom adducts from JSON file"""
        try:
            with open('custom_adducts.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.warning(f"Could not load custom adducts: {e}")
            return []

    def save_custom_adducts(self) -> bool:
        """Save custom adducts to JSON file"""
        try:
            with open('custom_adducts.json', 'w') as f:
                json.dump(self.custom_adducts, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving custom adducts: {e}")
            return False

    def calculate_mass_from_formula_with_validation(self, formula: str) -> tuple[bool, Optional[float], Optional[str]]:
        """
        Calculate monoisotopic mass from chemical formula (with validation wrapper)
        Returns: (success, mass, error_message)
        """
        try:
            # Use PythoMS to calculate mass from formula
            mol = IPMolecule(formula)
            mass = mol.monoisotopic_mass  # Correct attribute name
            return True, mass, None
        except Exception as e:
            return False, None, f"Invalid formula: {str(e)}"

    def add_custom_adduct(self, name: str, mass_or_formula: Union[float, str], charge: int,
                          prioritized: bool = False) -> tuple[bool, str]:
        """
        Add a custom adduct with automatic multiples generation
        Accepts either mass (number) or chemical formula (string)

        Args:
            name: Adduct name (e.g., "Acetate", "Phosphate")
            mass_or_formula: Either mass in Da (float/int) or chemical formula (string like "C2H3O2")
            charge: Charge state (-5 to +5, including 0)
            prioritized: If True, this is a conjugate that attaches to DNA before silver binding

        Returns: (success, message)
        """
        # Validation
        if not name or not isinstance(name, str):
            return False, "Invalid adduct name"

        # Check if already exists
        if any(a['name'] == name for a in self.custom_adducts):
            return False, f"Adduct '{name}' already exists"

        # Parse charge
        try:
            charge = int(charge)
        except ValueError:
            return False, "Invalid charge value"

        # Validate charge range (NOW ALLOWS 0!)
        if charge < -5 or charge > 5:
            return False, "Charge must be between -5 and +5 (including 0)"

        # Determine if input is formula or mass
        formula = None
        mass = None

        if isinstance(mass_or_formula, str) and mass_or_formula.strip():
            # Try to parse as formula first
            formula = mass_or_formula.strip()
            success, calculated_mass, error = self.calculate_mass_from_formula_with_validation(formula)

            if success:
                mass = calculated_mass
                logger.debug(f"Calculated mass from formula '{formula}': {mass:.6f} Da")
            else:
                # Maybe it's a number string?
                try:
                    mass = float(formula)
                    formula = None  # Was actually a mass
                except ValueError:
                    return False, error or "Invalid formula"

        else:
            # Direct mass input
            try:
                mass = float(mass_or_formula)
            except ValueError:
                return False, "Invalid mass or formula"

        # Validate mass range
        if mass is None or mass < 0.1 or mass > 10000:
            return False, "Mass must be between 0.1 and 10000 Da"

        # Add to custom list
        adduct_data = {
            'name': name,
            'mass': mass,
            'charge': charge,
            'prioritized': prioritized
        }
        if formula:
            adduct_data['formula'] = formula

        self.custom_adducts.append(adduct_data)

        # For conjugates (prioritized + charge 0), don't generate multiples
        # Conjugate count is handled separately in the analysis
        is_conjugate = prioritized and charge == 0

        if is_conjugate:
            # Only add single entry for conjugates (no multiples)
            self.adducts[name] = (mass, charge)
        else:
            # Add multiples to adduct library (1x, 2x) for regular adducts
            for n in range(1, 3):
                if n == 1:
                    adduct_name = name
                else:
                    adduct_name = f"{n}{name}"
                self.adducts[adduct_name] = (mass * n, charge * n)

        # Save to file
        if self.save_custom_adducts():
            # Clear peak analysis cache since adducts changed
            global _peak_analysis_cache
            if '_peak_analysis_cache' in globals():
                _peak_analysis_cache.clear()
                logger.debug("Cleared peak analysis cache (adducts changed)")

            if is_conjugate:
                if formula:
                    logger.info(f"Added conjugate: {name} (formula={formula}, mass={mass:.4f} Da) [CONJUGATE - no multiples]")
                else:
                    logger.info(f"Added conjugate: {name} (mass={mass:.4f} Da) [CONJUGATE - no multiples]")
                return True, f"Successfully added conjugate {name} (mass: {mass:.4f} Da)"
            else:
                priority_str = " [PRIORITIZED]" if prioritized else ""
                if formula:
                    logger.info(f"Added custom adduct: {name} (formula={formula}, mass={mass:.4f}, charge={charge:+d}){priority_str}")
                else:
                    logger.info(f"Added custom adduct: {name} (mass={mass:.4f}, charge={charge:+d}){priority_str}")
                logger.debug(f"Generated: {name}, 2{name}")
                return True, f"Successfully added {name} with multiples (mass: {mass:.4f} Da)"
        else:
            return False, "Failed to save custom adducts"

    def remove_custom_adduct(self, name: str) -> tuple[bool, str]:
        """
        Remove a custom adduct and its multiples
        Returns: (success, message)
        """
        # Find and remove from custom list
        original_len = len(self.custom_adducts)
        self.custom_adducts = [a for a in self.custom_adducts if a['name'] != name]

        if len(self.custom_adducts) == original_len:
            return False, f"Adduct '{name}' not found"

        # Remove multiples from adduct library (1x, 2x)
        for n in range(1, 3):
            if n == 1:
                adduct_name = name
            else:
                adduct_name = f"{n}{name}"
            if adduct_name in self.adducts:
                del self.adducts[adduct_name]

        # Save to file
        if self.save_custom_adducts():
            # Clear peak analysis cache since adducts changed
            global _peak_analysis_cache
            if '_peak_analysis_cache' in globals():
                _peak_analysis_cache.clear()
                logger.debug("Cleared peak analysis cache (adducts changed)")

            logger.info(f"Removed custom adduct: {name} and its multiples")
            return True, f"Successfully removed {name}"
        else:
            return False, "Failed to save custom adducts"

    def get_custom_adduct_names(self) -> list[str]:
        """Get list of custom adduct names with their multiples for analysis
        Note: Conjugates (prioritized + charge 0) are excluded - they're handled separately
        """
        names = []
        for custom in self.custom_adducts:
            # Skip conjugates - they're not searched as regular adducts
            is_conjugate = custom.get('prioritized', False) and custom.get('charge', 0) == 0
            if is_conjugate:
                continue
            name = custom['name']
            # Add 1x, 2x for regular adducts
            names.extend([name, f"2{name}"])
        return names

    def clear_all_custom_adducts(self) -> tuple[bool, str]:
        """
        Clear all custom adducts and reset to default built-in adducts
        Returns: (success, message)
        """
        # Remove all custom adduct multiples from adduct library (1x, 2x)
        for custom in self.custom_adducts:
            name = custom['name']
            for n in range(1, 3):
                if n == 1:
                    adduct_name = name
                else:
                    adduct_name = f"{n}{name}"
                if adduct_name in self.adducts:
                    del self.adducts[adduct_name]

        # Clear custom list
        self.custom_adducts = []

        # Save empty list to file
        if self.save_custom_adducts():
            logger.info("Cleared all custom adducts - reset to default built-in adducts")
            return True, "All custom adducts cleared"
        else:
            return False, "Failed to save cleared adducts"

    def toggle_adduct_priority(self, name: str) -> tuple[bool, str]:
        """
        Toggle the prioritized status of a custom adduct
        Returns: (success, message)
        """
        for adduct in self.custom_adducts:
            if adduct['name'] == name:
                old_status = adduct.get('prioritized', False)
                adduct['prioritized'] = not old_status
                new_status = adduct['prioritized']

                if self.save_custom_adducts():
                    status_str = "prioritized (conjugate)" if new_status else "normal adduct"
                    logger.info(f"Toggled {name} to {status_str}")
                    return True, f"{name} is now {status_str}"
                else:
                    return False, "Failed to save adduct changes"

        return False, f"Adduct '{name}' not found"

    def get_prioritized_conjugate(self) -> Optional[dict]:
        """
        Get the first prioritized adduct with charge 0 (treated as a conjugate)
        Returns: dict with name, mass, formula (if available), or None
        """
        for adduct in self.custom_adducts:
            if adduct.get('prioritized', False) and adduct.get('charge', 0) == 0:
                return {
                    'name': adduct['name'],
                    'mass': adduct['mass'],
                    'formula': adduct.get('formula'),
                    'atoms': self.parse_formula_to_atoms(adduct.get('formula')) if adduct.get('formula') else None
                }
        return None

    def parse_formula_to_atoms(self, formula: str) -> Optional[dict]:
        """
        Parse a chemical formula to atom counts
        E.g., 'C17H26NO7P' -> {'C': 17, 'H': 26, 'N': 1, 'O': 7, 'P': 1}
        """
        if not formula:
            return None
        import re
        atoms = {}
        # Match element symbols (1-2 letters, first uppercase) followed by optional count
        pattern = r'([A-Z][a-z]?)(\d*)'
        for match in re.finditer(pattern, formula):
            element = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1
            atoms[element] = atoms.get(element, 0) + count
        return atoms if atoms else None

    def _try_dimer_fallback(self, peak_mz, charge, dna_sequence, exp_x0,
                            resolution, mz_values, intensity_values, custom_xna,
                            conjugate_name, conjugate_count, kwargs):
        if charge < 2 or kwargs.get('_dimer_fallback'):
            return None

        monomer_neutral_mass = exp_x0 * charge / 2
        z_half = charge / 2
        z_candidates = sorted({max(1, int(z_half)), max(1, int(z_half) + (1 if z_half != int(z_half) else 0))})

        best_candidates = None
        best_x0 = 999.0

        for z_mono in z_candidates:
            monomer_mz = monomer_neutral_mass / z_mono
            logger.info(f"Trying dimer fallback: z={charge} -> monomer z={z_mono}, "
                        f"monomer_neutral_mass={monomer_neutral_mass:.2f}, monomer_mz={monomer_mz:.4f}")
            mono_result = self.analyze_peak_with_smart_adduct_search(
                monomer_mz, z_mono, dna_sequence, monomer_mz,
                resolution=resolution, mz_values=mz_values, intensity_values=intensity_values,
                custom_xna=custom_xna, conjugate_name=conjugate_name,
                conjugate_count=conjugate_count, _dimer_fallback=True
            )
            if mono_result:
                mono_best = min(c.get('abs_x0_error', 999.0) for c in mono_result)
                if mono_best < best_x0:
                    best_x0 = mono_best
                    best_candidates = mono_result

        if not best_candidates:
            logger.info("Dimer fallback found no candidates")
            return None
        for comp in best_candidates:
            comp['is_multimer'] = True
            comp['multimer_label'] = 'Dimer (×2)'
            comp['z'] = charge
            if comp.get('ion_formula'):
                comp['ion_formula'] = self._multiply_formula(comp['ion_formula'], 2)
            if comp.get('formula'):
                comp['formula'] = self._multiply_formula(comp['formula'], 2)
            if comp.get('neutral_formula'):
                comp['neutral_formula'] = self._multiply_formula(comp['neutral_formula'], 2)
        logger.info(f"Dimer fallback found {len(best_candidates)} candidates")
        return best_candidates

    def _multiply_formula(self, formula: str, multiplier: int) -> str:
        import re
        atoms = {}
        for match in re.finditer(r'([A-Z][a-z]?)(\d*)', formula):
            element = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1
            atoms[element] = atoms.get(element, 0) + count
        order = ['C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Na', 'K', 'Ag']
        parts = []
        used = set()
        for el in order:
            if el in atoms:
                n = atoms[el] * multiplier
                parts.append(f"{el}{n}" if n > 1 else el)
                used.add(el)
        for el in sorted(atoms.keys()):
            if el not in used:
                n = atoms[el] * multiplier
                parts.append(f"{el}{n}" if n > 1 else el)
        return ''.join(parts)

    def _get_extra_conjugate_contribution(self, conj_atoms: dict, total_conjugates: int) -> tuple[float, str]:
        """
        Calculate mass and formula contribution from non-HCNOP elements in a conjugate.

        Args:
            conj_atoms: dict of element -> count from parse_formula_to_atoms
            total_conjugates: total number of conjugate molecules (count * strands)

        Returns:
            (extra_mass, extra_formula_suffix) for all non-{H,C,N,O,P} atoms
            e.g., (31.972, "S1") for a single sulfur atom with 1 conjugate
        """
        core_elements = {'H', 'C', 'N', 'O', 'P'}
        extra_mass = 0.0
        extra_parts = []

        for element, count in sorted(conj_atoms.items()):
            if element in core_elements:
                continue
            total_count = count * total_conjugates
            # Look up monoisotopic mass
            if element in self.MONOISOTOPIC_MASSES:
                elem_mass = self.MONOISOTOPIC_MASSES[element]
            else:
                # Fallback: use PythoMS for unknown elements
                try:
                    mol = IPMolecule(f"{element}1")
                    elem_mass = mol.monoisotopic
                    logger.info(f"Extra element '{element}' mass from PythoMS: {elem_mass:.6f}")
                except Exception:
                    logger.warning(f"Unknown element '{element}' in conjugate formula, skipping")
                    continue
            extra_mass += elem_mass * total_count
            extra_parts.append(f"{element}{total_count}")

        extra_formula = ''.join(extra_parts)
        return extra_mass, extra_formula

    def calculate_dna_composition(self, dna_sequence: str, strands: int = 1) -> tuple[int, int, int, int, int]:
        """
        Calculate DNA composition from sequence (matching MASS.py logic)
        Returns: nH, nC, nN, nO, nP for the DNA
        """
        nH = 0
        nC = 0
        nN = 0
        nO = 0

        # Add atoms for each base
        for base in dna_sequence.upper():
            if base == "C":
                nC += 4
                nH += 4
                nN += 3
                nO += 1
            elif base == "G":
                nC += 5
                nH += 4
                nN += 5
                nO += 1
            elif base == "A":
                nC += 5
                nH += 4
                nN += 5
                nO += 0
            elif base == "T":
                nC += 5
                nH += 5
                nN += 2
                nO += 2

        # DNA backbone
        # OH ends
        nH += 1
        nO += 2
        # Phosphodiester bonds
        length = len(dna_sequence)
        nP = length - 1
        nO += nP * 4
        # Deoxyriboses
        nC += length * 5
        nH += length * 8
        nO += length

        # Multiply by number of strands
        nH_total = nH * strands
        nC_total = nC * strands
        nN_total = nN * strands
        nO_total = nO * strands
        nP_total = nP * strands

        return nH_total, nC_total, nN_total, nO_total, nP_total

    def calculate_dna_composition_with_conjugate(self, dna_sequence: str, strands: int = 1,
                                                  conjugate_name: Optional[str] = None,
                                                  conjugate_count: int = 0) -> tuple[int, int, int, int, int, float, str, float, str]:
        """
        Calculate DNA composition including conjugate atoms

        Args:
            dna_sequence: DNA sequence
            strands: Number of strands
            conjugate_name: Name of conjugate (e.g., 'BCN')
            conjugate_count: Number of conjugate molecules per strand

        Returns: (nH, nC, nN, nO, nP, conjugate_mass, notation_prefix, extra_conj_mass, extra_conj_formula)
            - nH, nC, nN, nO, nP: atom counts including conjugate (core elements only)
            - conjugate_mass: total conjugate mass added
            - notation_prefix: string like "(DNA-BCN)" or "(DNA-2BCN)" for notation
            - extra_conj_mass: mass from non-HCNOP elements in conjugate
            - extra_conj_formula: formula suffix for non-HCNOP elements (e.g., "S1")
        """
        # Get base DNA composition
        nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, strands)

        conjugate_mass = 0.0
        notation_prefix = "DNA"  # Default notation
        extra_conj_mass = 0.0
        extra_conj_formula = ""

        if conjugate_name and conjugate_count > 0:
            # Get conjugate info
            conjugate = None
            for adduct in self.custom_adducts:
                if adduct['name'] == conjugate_name:
                    conjugate = adduct
                    break

            if conjugate:
                # Get conjugate atoms
                atoms = self.parse_formula_to_atoms(conjugate.get('formula'))
                if atoms:
                    # Add conjugate atoms × count × strands
                    total_conjugates = conjugate_count  # Total conjugates (not per-strand)
                    nH += atoms.get('H', 0) * total_conjugates
                    nC += atoms.get('C', 0) * total_conjugates
                    nN += atoms.get('N', 0) * total_conjugates
                    nO += atoms.get('O', 0) * total_conjugates
                    nP += atoms.get('P', 0) * total_conjugates
                    # Handle non-HCNOP elements
                    extra_conj_mass, extra_conj_formula = self._get_extra_conjugate_contribution(atoms, total_conjugates)
                    conjugate_mass = conjugate['mass'] * total_conjugates

                    # Build notation prefix
                    if conjugate_count == 1:
                        notation_prefix = f"(DNA-{conjugate_name})"
                    else:
                        notation_prefix = f"(DNA-{conjugate_count}{conjugate_name})"

                    logger.debug(f"Added conjugate: {total_conjugates}x {conjugate_name}, mass={conjugate_mass:.4f}, extra_mass={extra_conj_mass:.4f}, extra_formula={extra_conj_formula}")
                else:
                    # No formula, just add mass
                    conjugate_mass = conjugate['mass'] * conjugate_count  # Total (not per-strand)
                    if conjugate_count == 1:
                        notation_prefix = f"(DNA-{conjugate_name})"
                    else:
                        notation_prefix = f"(DNA-{conjugate_count}{conjugate_name})"

        return nH, nC, nN, nO, nP, conjugate_mass, notation_prefix, extra_conj_mass, extra_conj_formula

    def calculate_mass_from_formula(self, formula: str) -> float:
        """
        Calculate monoisotopic mass from a chemical formula using PythoMS
        Returns: monoisotopic mass in Daltons (consistent with isotope pattern generation)
        """
        try:
            # Use charge=1 so the monoisotopic mass is computed consistently with isotope-pattern generation (singly charged ion mass)
            mol = IPMolecule(formula, charge=1, verbose=False)
            return mol.monoisotopic_mass
        except Exception as e:
            raise ValueError(f"Could not calculate mass from formula '{formula}': {e}")

    def parse_txt_spectrum(self, txt_content: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Parse mass spectrum from txt file
        Expected format: two columns (m/z, intensity) separated by whitespace or comma
        """
        lines = txt_content.strip().replace('\r\n', '\n').replace('\r', '\n').split('\n')
        mz_values = []
        intensity_values = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Try different delimiters
            parts = None
            if '\t' in line:
                parts = line.split('\t')
            elif ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()

            if len(parts) >= 2:
                try:
                    mz = float(parts[0])
                    intensity = float(parts[1])
                    mz_values.append(mz)
                    intensity_values.append(intensity)
                except ValueError:
                    continue

        return np.array(mz_values), np.array(intensity_values)

    def generate_compositions_for_peak(self, peak_mz: float, charge: int, dna_sequence: str,
                                        resolution: int = 20000, detected_centroid: Optional[float] = None) -> list[dict]:
        """
        Generate compositions for a peak with user-specified charge state
        This is used when the user manually sets the charge state

        Parameters:
        - peak_mz: m/z value for initial search
        - charge: charge state
        - dna_sequence: DNA sequence
        - resolution: instrument resolution
        - detected_centroid: if provided, use X0-based matching instead of m/z matching
        """
        return self.calculate_dna_silver_composition(peak_mz, charge, dna_sequence, detected_centroid=detected_centroid, resolution=resolution)

    def calculate_composition_estimates(self, peak_mz: float, z_observed: int, dna_sequence: str,
                                         custom_xna: Optional[dict] = None, max_strands: int = MAX_STRANDS * 2,
                                         is_complex: bool = False) -> list[dict]:
        """
        Calculate estimated nAg for each strand count based on observed m/z.
        Used to show users what compositions might match when automatic search fails.

        Note: Only nAg is estimated from mass. Qcl and N0 require isotope pattern analysis
        and cannot be reliably estimated from mass alone.

        For complex mode, ns represents number of complexes (1 complex = 2 strands).

        Returns: list of dicts with ns, nAg, status, and warning
        """
        estimates: list[dict[str, Any]] = []

        if not dna_sequence and not custom_xna:
            return estimates

        # Calculate observed neutral mass from m/z
        # Use simple m/z * z (same as detected_centroid * z used in composition search)
        observed_mass = peak_mz * z_observed

        # For complex mode: iterate by complexes (1, 2, 3 = 2, 4, 6 strands)
        # For other modes: iterate by strands (1, 2, 3, 4, 5, 6)
        if is_complex:
            max_complexes = max_strands // 2
            strand_values = [nd * 2 for nd in range(1, max_complexes + 1)]  # [2, 4, 6]
        else:
            strand_values = list(range(1, max_strands + 1))  # [1, 2, 3, 4, 5, 6]

        for num_strands in strand_values:
            # Calculate DNA mass for this strand count
            if custom_xna and custom_xna.get('formula'):
                # Use full formula mass (handles all elements, not just H,C,N,O,P)
                xna_mass = self.calculate_mass_from_formula(custom_xna['formula'])
                if is_complex:
                    # For complex mode, formula is for 1 complex (2 strands)
                    num_complexes = num_strands // 2
                    dna_mass = xna_mass * num_complexes
                else:
                    dna_mass = xna_mass * num_strands
            else:
                # DNA mode or complex DNA mode (no XNA formula) - calculate from sequence
                nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, num_strands)
                dna_mass = self.m_p * nH + self.mC * nC + self.mN * nN + self.mO * nO + self.mP * nP

            # Estimate nAg: observed_mass ≈ dna_mass + nAg * mAg
            # Note: This ignores Qcl contribution which is small compared to Ag mass
            remaining_mass = observed_mass - dna_mass
            nAg_estimate_raw = remaining_mass / self.mAg
            nAg_estimate = int(round(nAg_estimate_raw))

            # Determine status and warning based on nAg only
            if nAg_estimate < 0:
                status = 'invalid'
                warning = 'nAg < 0 (DNA mass exceeds observed mass)'
            elif nAg_estimate > MAX_SILVER * 2:
                status = 'high'
                warning = 'nAg too high'
            elif nAg_estimate > MAX_SILVER:
                status = 'possible'
                warning = 'High nAg'
            else:
                status = 'valid'
                warning = None

            # For complex mode, ns represents complexes; for other modes, strands
            ns_value = num_strands // 2 if is_complex else num_strands

            estimates.append({
                'ns': ns_value,  # Complexes for complex mode, strands for other modes
                'nAg': nAg_estimate,
                'status': status,
                'warning': warning
            })

            # Smart truncation: if nAg < 0, higher strand counts will also be invalid
            # (more DNA mass = even more negative nAg), so stop searching
            if nAg_estimate < 0:
                break

        return estimates

    def find_compositions(self, peak_mz: float, dna_sequence: str, charge: Optional[int] = None,
                         strand_range: tuple[int, int] = (1, MAX_STRANDS), silver_range: tuple[int, int] = (0, MAX_SILVER),
                         ppm_threshold: int = 200, detected_centroid: Optional[float] = None) -> list[dict]:
        """
        Find possible compositions for a given peak m/z value.
        If detected_centroid is provided, use X0-based matching instead of mass-based matching.

        Parameters:
        - peak_mz: m/z value of the peak
        - dna_sequence: DNA sequence string
        - charge: charge state (if known), otherwise searches multiple charge states
        - strand_range: tuple of (min_strands, max_strands) to search
        - silver_range: tuple of (min_ag, max_ag) number of silver atoms
        - ppm_threshold: maximum mass error in ppm

        Returns: list of composition dictionaries
        """
        compositions: list[dict[str, Any]] = []

        if not dna_sequence:
            return compositions

        min_strands, max_strands = strand_range
        min_ag, max_ag = silver_range

        for num_strands in range(min_strands, max_strands + 1):
            for num_ag in range(min_ag, max_ag + 1):
                # Calculate DNA composition
                nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, num_strands)

                # Calculate masses for each element
                mH_total = self.m_p * nH
                mC_total = self.mC * nC
                mN_total = self.mN * nN
                mO_total = self.mO * nO
                mP_total = self.mP * nP
                mAg_total = self.mAg * num_ag

                # Try different z values
                z_values = [charge] if charge else [1, 2, 3, 4, 5, 6, 7, 8]

                for z_test in z_values:
                    if z_test is None or z_test <= 0:
                        continue

                    for qcl in range(0, num_ag + 1):
                        n0_valence = num_ag - qcl

                        if n0_valence < 0:
                            continue

                        # Calculate mass: DNA + Ag - (Qcl + z) * mH
                        mass = mP_total + mH_total + mC_total + mN_total + mO_total + mAg_total - (qcl + z_test) * self.m_p
                        expected_mz = mass / z_test

                        mass_error_ppm = abs((expected_mz - peak_mz) / peak_mz * 1e6)

                        # If using X0-based matching, generate ALL N0 values without PPM filter
                        # Otherwise use traditional PPM filtering
                        if detected_centroid is not None or mass_error_ppm < ppm_threshold:
                            neutral_formula = f"C{nC}H{nH}N{nN}O{nO}P{nP}Ag{num_ag}"

                            nH_ion = nH - (qcl + z_test)
                            ion_formula = f"C{nC}H{nH_ion}N{nN}O{nO}P{nP}Ag{num_ag}"

                            compositions.append({
                                'type': 'nanocluster',
                                'num_strands': num_strands,
                                'num_silver': num_ag,
                                'qcl': qcl,
                                'n0': n0_valence,
                                'z': z_test,
                                'formula': neutral_formula,
                                'ion_formula': ion_formula,
                                'neutral_formula': neutral_formula,
                                'adduct': '',
                                'full_notation': f"{neutral_formula}-{qcl+z_test}H (z={z_test}, Qcl={qcl}, N0={n0_valence})",
                                'expected_mz': expected_mz,
                                'mass_error_ppm': mass_error_ppm,
                                'x0_error': 999.0,
                                'abs_x0_error': 999.0,
                                'pattern_score': 0.0,
                                'nH': nH, 'nC': nC, 'nN': nN, 'nO': nO, 'nP': nP,
                                '_base_match': True
                            })

        return compositions

    def calculate_dna_silver_composition(self, mz: float, z_observed: int, dna_sequence: Optional[str] = None,
                                          detected_centroid: Optional[float] = None, resolution: int = 20000,
                                          custom_xna: Optional[dict] = None,
                                          conjugate_name: Optional[str] = None,
                                          conjugate_count: int = 0) -> list[dict]:
        """
        Calculate possible DNA-silver compositions including:
        1. DNA-stabilized silver nanoclusters (following MASS.py)
        2. DNA + Ag+ ions (no cluster)
        3. Various adducts (NH4+, Na+, K+, Cl-, etc.)

        Parameters:
        - mz: observed m/z value
        - z_observed: charge state (from isotope spacing)
        - dna_sequence: DNA sequence string
        - detected_centroid: frontend-calculated centroid (if provided, bypass PPM filter)
        - resolution: instrument resolution for isotope pattern generation (default 20000 is fallback when webapp cannot parse from uploaded data)
        - custom_xna: dict with custom XNA nucleotide (name, formula) - mass calculated from formula
        - conjugate_name: Name of conjugate (e.g., 'BCN') attached to DNA before silver binding
        - conjugate_count: Number of conjugate molecules per strand

        Returns: list of possible compositions with scoring
        """
        compositions: list[dict[str, Any]] = []

        # For XNA mode, dna_sequence is not required (we use formula to calculate mass)
        if not custom_xna and not dna_sequence:
            return compositions  # Need DNA sequence for accurate calculation (unless using XNA)

        # Pre-calculate XNA composition if using custom XNA
        # Parse once to avoid recalculating in every loop iteration
        xna_composition_one = None
        xna_strand1_composition = None
        xna_strand2_composition = None
        xna_mass_one = None  # Full formula mass (handles all elements, not just H,C,N,O,P)
        xna_strand1_mass = None
        xna_strand2_mass = None
        is_complex_mode = False
        same_strands = False

        if custom_xna:
            # Check if this is complex mode with individual strand formulas
            is_complex_mode = custom_xna.get('is_complex', False)
            same_strands = custom_xna.get('same_strands', False)

            # Only process XNA formulas if formula field exists (DNA-only Complex has no formula)
            if custom_xna.get('formula'):
                if is_complex_mode:
                    # Complex mode: parse strand1, strand2, and combined formulas
                    strand1_formula = custom_xna.get('strand1_formula', '')
                    strand2_formula = custom_xna.get('strand2_formula', '')

                    if strand1_formula:
                        xna_strand1_composition = composition_from_formula(strand1_formula)
                        xna_strand1_mass = self.calculate_mass_from_formula(strand1_formula)
                    if strand2_formula and not same_strands:
                        xna_strand2_composition = composition_from_formula(strand2_formula)
                        xna_strand2_mass = self.calculate_mass_from_formula(strand2_formula)

                    # Combined complex formula
                    xna_composition_one = composition_from_formula(custom_xna['formula'])
                    xna_mass_one = self.calculate_mass_from_formula(custom_xna['formula'])
                    logger.info(f"COMPLEX MODE: strand1={strand1_formula}, strand2={strand2_formula or strand1_formula}, combined={custom_xna['formula']}, same_strands={same_strands}")
                else:
                    # Regular XNA mode: calculate from formula (same as DNA mode)
                    xna_composition_one = composition_from_formula(custom_xna['formula'])
                    xna_mass_one = self.calculate_mass_from_formula(custom_xna['formula'])
                    logger.info(f"XNA MODE: formula={custom_xna['formula']}, mass={xna_mass_one:.4f} Da")
            elif is_complex_mode:
                # DNA-only Complex mode (no XNA formula) - will use DNA sequence below
                logger.info(f"COMPLEX DNA MODE: Using DNA sequence for mass calculation (no XNA formula)")

        # PART 1: DNA-stabilized silver nanoclusters
        # Formula from MASS.py (exact implementation):
        # mass = mP + mH + mC + mN + mO + mAg - (Qcl + z)
        # m/z = mass / z
        # Where: N0 + Qcl = nAg (relationship between valence electrons and cluster charge)

        # Dynamically determine strand range based on initial results
        # Start with 1-3, but will expand if all have N0 > 20
        max_strands = MAX_STRANDS

        # Track best composition for adduct analysis (even if outside threshold)
        best_overall_error = float('inf')
        best_overall_params = None

        # Pre-calculate z values once
        z_values = [z_observed] if z_observed else [1, 2, 3, 4, 5, 6, 7, 8]

        # OPTIMIZATION: Estimate nAg range from m/z to reduce search space
        # Formula: observed_mass ≈ DNA_mass + nAg * mAg
        # So: nAg ≈ (observed_mass - DNA_mass) / mAg
        # Use peak_mz * z as rough mass estimate
        if detected_centroid is not None and z_observed:
            observed_mass_estimate = detected_centroid * z_observed
            # Estimate nAg for each strand configuration and find reasonable range
            nAg_estimates = []
            if is_complex_mode and custom_xna:
                # For complex mode, estimate nAg for each configuration
                test_configs = []
                if xna_strand1_mass:
                    test_configs.append(('strand1', xna_strand1_mass))
                if xna_strand2_mass and not same_strands:
                    test_configs.append(('strand2', xna_strand2_mass))
                if xna_mass_one:
                    test_configs.append(('complex', xna_mass_one))

                # For DNA-only Complex mode (no XNA formula), use DNA sequence
                if not test_configs and dna_sequence:
                    # Calculate one complex mass (2 strands)
                    nH_1d, nC_1d, nN_1d, nO_1d, nP_1d = self.calculate_dna_composition(dna_sequence, 2)
                    one_complex_mass = self.m_p * nH_1d + self.mC * nC_1d + self.mN * nN_1d + self.mO * nO_1d + self.mP * nP_1d

                    # COMPLEX nAg estimation: subtract complex masses until remaining < one_complex_mass
                    # Then divide remaining by mAg to get baseline nAg
                    remaining_mass = observed_mass_estimate
                    nd_estimate = 0
                    while remaining_mass >= one_complex_mass and nd_estimate < 10:
                        remaining_mass -= one_complex_mass
                        nd_estimate += 1

                    # remaining_mass ≈ nAg × mAg
                    nAg_estimate = remaining_mass / self.mAg
                    logger.info(f"COMPLEX DNA nAg estimation: observed={observed_mass_estimate:.2f}, one_complex={one_complex_mass:.2f}, nd={nd_estimate}, remaining={remaining_mass:.2f}, nAg≈{nAg_estimate:.1f}")

                    if nAg_estimate >= -5:
                        nAg_estimates.append(int(round(nAg_estimate)))
                else:
                    logger.debug(f"COMPLEX XNA nAg estimation (observed_mass={observed_mass_estimate:.2f} Da)")
                    for config_name, test_dna_mass in test_configs:
                        nAg_estimate = (observed_mass_estimate - test_dna_mass) / self.mAg
                        logger.debug(f"  {config_name}: DNA_mass={test_dna_mass:.2f}, nAg_estimate={nAg_estimate:.1f}")
                        # Only add reasonable estimates (not too negative)
                        if nAg_estimate >= -5:
                            nAg_estimates.append(int(round(nAg_estimate)))
            elif custom_xna and xna_mass_one is not None:
                # Regular XNA mode - use pre-calculated full formula mass
                for test_strands in range(1, max_strands + 1):
                    test_dna_mass = xna_mass_one * test_strands
                    nAg_estimate = (observed_mass_estimate - test_dna_mass) / self.mAg
                    nAg_estimates.append(int(round(nAg_estimate)))
            elif dna_sequence:
                # DNA mode - include conjugate mass if present
                conj_mass_per_strand = 0.0
                if conjugate_name and conjugate_count > 0:
                    for adduct in self.custom_adducts:
                        if adduct['name'] == conjugate_name:
                            conj_mass_per_strand = adduct['mass'] * conjugate_count
                            break

                for test_strands in range(1, max_strands + 1):
                    nH_t, nC_t, nN_t, nO_t, nP_t = self.calculate_dna_composition(dna_sequence, test_strands)
                    test_dna_mass = self.m_p * nH_t + self.mC * nC_t + self.mN * nN_t + self.mO * nO_t + self.mP * nP_t
                    # Add conjugate mass (per strand × number of strands)
                    test_dna_mass += conj_mass_per_strand * test_strands
                    nAg_estimate = (observed_mass_estimate - test_dna_mass) / self.mAg
                    nAg_estimates.append(int(round(nAg_estimate)))

            # Set search range: min estimate - 5 to max estimate + 5, clamped to [0, MAX_SILVER]
            if nAg_estimates:
                nAg_min = max(0, min(nAg_estimates) - 5)
                nAg_max = min(MAX_SILVER + 1, max(nAg_estimates) + 5 + 1)  # +1 for range() to include max
                # When conjugate is present, always include nAg=0 (DNA-conjugate without silver is valid)
                if conjugate_name and conjugate_count > 0:
                    nAg_min = 0
                    logger.info(f"CONJUGATE MODE: Forcing nAg_min=0 to allow DNA-{conjugate_name} without silver")
                logger.info(f"BASELINE nAg range: [{nAg_min}, {nAg_max-1}] (estimates={nAg_estimates}, exp_x0={detected_centroid:.2f}, z={z_observed})")
            else:
                nAg_min, nAg_max = 0, MAX_SILVER + 1
        else:
            nAg_min, nAg_max = 0, MAX_SILVER + 1

        # For complex mode, use special strand configurations
        if is_complex_mode and custom_xna:
            # Build list of strand configurations to search:
            # - strand1 (single strand 1 - NEW: allows detecting single strands with Ag)
            # - strand2 (single strand 2, if different from strand1)
            # - nd=1 (1 complex = 2 strands)
            # - nd=2 (2 complexes = 4 strands)
            # - nd=3 (3 complexes = 6 strands)
            # Note: In complex mode, ns (number of complexes) = num_strands / 2
            complex_configs = []

            # FIRST: Search individual single strands (for detecting single strand + Ag)
            if xna_strand1_composition:
                complex_configs.append(('strand1', 1, xna_strand1_composition))
            if xna_strand2_composition and not same_strands:
                complex_configs.append(('strand2', 1, xna_strand2_composition))

            # THEN: Search for 1, 2, 3 complexes (2, 4, 6 strands)
            for num_complexes in range(1, MAX_COMPLEXES + 1):
                num_strands_total = num_complexes * 2
                # Scale composition by number of complexes
                # For XNA Complex: use xna_composition_one
                # For DNA-only Complex: use 'DNA' marker (will calculate from sequence below)
                comp_marker = xna_composition_one if xna_composition_one is not None else 'DNA'
                complex_configs.append((f'nd={num_complexes}', num_strands_total, comp_marker))
            logger.info(f"COMPLEX MODE: Searching {len(complex_configs)} configurations: {[c[0] for c in complex_configs]}")
            strand_loop = complex_configs
        else:
            strand_loop = [(f"{i}strand", i, None) for i in range(1, max_strands + 1)]

        for strand_config in strand_loop:
            strand_label, num_strands, complex_comp = strand_config

            # OPTIMIZATION: Calculate DNA/XNA composition ONCE per strand count (moved outside num_ag loop)
            if custom_xna:
                if is_complex_mode:
                    # COMPLEX MODE: Handle both single strands (strand1/strand2) and complexes (nd=X)
                    comp_to_use = complex_comp
                    is_single_strand = strand_label in ['strand1', 'strand2']

                    if comp_to_use == 'DNA':
                        # DNA-only Complex mode: calculate from DNA sequence
                        if not dna_sequence:
                            logger.warning(f"COMPLEX DNA: SKIPPING {strand_label} - no DNA sequence!")
                            continue
                        nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, num_strands)
                        mH_total = self.m_p * nH
                        mC_total = self.mC * nC
                        mN_total = self.mN * nN
                        mO_total = self.mO * nO
                        mP_total = self.mP * nP
                        mDNA_total = mH_total + mC_total + mN_total + mO_total + mP_total
                        num_complexes = num_strands // 2
                        extra_conj_mass = 0.0
                        extra_conj_formula = ""
                        logger.info(f"COMPLEX DNA: {strand_label} (strands={num_strands}, complexes={num_complexes}, DNA_mass={mDNA_total:.4f} Da, seq={dna_sequence[:10]}...)")
                    elif comp_to_use is None:
                        logger.warning(f"COMPLEX: SKIPPING {strand_label} - composition is None!")
                        continue
                    elif is_single_strand:
                        # SINGLE STRAND in complex mode: use composition directly (no scaling)
                        nH = comp_to_use.get('H', 0)
                        nC = comp_to_use.get('C', 0)
                        nN = comp_to_use.get('N', 0)
                        nO = comp_to_use.get('O', 0)
                        nP = comp_to_use.get('P', 0)

                        # Calculate element masses for isotope pattern generation
                        mH_total = self.m_p * nH
                        mC_total = self.mC * nC
                        mN_total = self.mN * nN
                        mO_total = self.mO * nO
                        mP_total = self.mP * nP

                        # Use single strand mass
                        if strand_label == 'strand1' and xna_strand1_mass:
                            mDNA_total = xna_strand1_mass
                        elif strand_label == 'strand2' and xna_strand2_mass:
                            mDNA_total = xna_strand2_mass
                        else:
                            continue  # Skip if no valid mass
                        extra_conj_mass = 0.0
                        extra_conj_formula = ""
                        logger.debug(f"COMPLEX XNA: searching {strand_label} (single strand, mass={mDNA_total:.2f} Da)")
                    else:
                        # XNA Complex mode (nd=X): use pre-calculated formula composition
                        # Extract element counts and scale by number of complexes
                        num_complexes = num_strands // 2
                        nH = comp_to_use.get('H', 0) * num_complexes
                        nC = comp_to_use.get('C', 0) * num_complexes
                        nN = comp_to_use.get('N', 0) * num_complexes
                        nO = comp_to_use.get('O', 0) * num_complexes
                        nP = comp_to_use.get('P', 0) * num_complexes

                        # Calculate element masses for isotope pattern generation
                        mH_total = self.m_p * nH
                        mC_total = self.mC * nC
                        mN_total = self.mN * nN
                        mO_total = self.mO * nO
                        mP_total = self.mP * nP

                        # Scale the formula mass by number of complexes
                        if xna_mass_one is not None:
                            mDNA_total = xna_mass_one * num_complexes
                        else:
                            continue  # Skip if no valid mass
                        extra_conj_mass = 0.0
                        extra_conj_formula = ""
                        logger.debug(f"COMPLEX XNA: searching {strand_label} (num_strands={num_strands}, num_complexes={num_complexes}, formula_mass={mDNA_total:.2f} Da)")
                elif xna_composition_one is not None and xna_mass_one is not None:
                    # Regular XNA mode: multiply formula by num_strands
                    nH = xna_composition_one.get('H', 0) * num_strands
                    nC = xna_composition_one.get('C', 0) * num_strands
                    nN = xna_composition_one.get('N', 0) * num_strands
                    nO = xna_composition_one.get('O', 0) * num_strands
                    nP = xna_composition_one.get('P', 0) * num_strands

                    # Calculate element masses for isotope pattern generation
                    mH_total = self.m_p * nH
                    mC_total = self.mC * nC
                    mN_total = self.mN * nN
                    mO_total = self.mO * nO
                    mP_total = self.mP * nP

                    # Use pre-calculated full formula mass (handles all elements)
                    mDNA_total = xna_mass_one * num_strands
                    extra_conj_mass = 0.0
                    extra_conj_formula = ""
                else:
                    continue  # Skip if XNA composition not available
            elif dna_sequence:
                # Calculate standard DNA composition
                nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, num_strands)

                # Add conjugate atoms if present (conjugate attaches to DNA before silver)
                extra_conj_mass = 0.0
                extra_conj_formula = ""
                if conjugate_name and conjugate_count > 0:
                    conj_atoms = None
                    conj_mass = 0.0
                    for adduct in self.custom_adducts:
                        if adduct['name'] == conjugate_name:
                            conj_atoms = self.parse_formula_to_atoms(adduct.get('formula'))
                            conj_mass = adduct['mass']
                            break
                    if conj_atoms:
                        # Add conjugate atoms per strand
                        total_conjugates = conjugate_count  # Total conjugates (not per-strand)
                        nH += conj_atoms.get('H', 0) * total_conjugates
                        nC += conj_atoms.get('C', 0) * total_conjugates
                        nN += conj_atoms.get('N', 0) * total_conjugates
                        nO += conj_atoms.get('O', 0) * total_conjugates
                        nP += conj_atoms.get('P', 0) * total_conjugates
                        # Handle non-HCNOP elements (e.g., S in biotin)
                        extra_conj_mass, extra_conj_formula = self._get_extra_conjugate_contribution(conj_atoms, total_conjugates)
                        logger.debug(f"CONJUGATE: Added {total_conjugates}x {conjugate_name} atoms to {num_strands} strands, extra_mass={extra_conj_mass:.4f}, extra_formula={extra_conj_formula}")

                # Calculate masses for each element
                mH_total = self.m_p * nH
                mC_total = self.mC * nC
                mN_total = self.mN * nN
                mO_total = self.mO * nO
                mP_total = self.mP * nP
                mDNA_total = mH_total + mC_total + mN_total + mO_total + mP_total + extra_conj_mass
            else:
                continue  # Skip if no valid sequence/formula

            for num_ag in range(nAg_min, nAg_max + 1):  # Use optimized range instead of 0-30
                mAg_total = self.mAg * num_ag

                for z_test in z_values:
                    if z_test is None or z_test <= 0:
                        continue

                    # If detected_centroid is provided, first check if ANY Qcl gives m/z close to centroid
                    if detected_centroid is not None:
                        # OPTIMIZATION: Direct calculation of optimal qcl instead of loop
                        # From: mass = mDNA + mAg - (qcl + z) * mH, mz = mass / z
                        # So: qcl = (mDNA + mAg - mz*z) / mH - z
                        qcl_raw = (mDNA_total + mAg_total - detected_centroid * z_test) / self.m_p - z_test
                        # COMPLEX MODE: Qcl must equal nAg (N0 = 0), so use num_ag directly
                        # Otherwise, use algebraic optimal Qcl (clamped to [0, num_ag])
                        if is_complex_mode:
                            best_qcl_for_debug = num_ag
                        else:
                            best_qcl_for_debug = max(0, min(num_ag, round(qcl_raw)))
                        # Calculate actual m/z for best qcl
                        mass_test = mDNA_total + mAg_total - (best_qcl_for_debug + z_test) * self.m_p
                        best_mz_for_debug = mass_test / z_test
                        min_mz_error = abs(best_mz_for_debug - detected_centroid)

                        # Track the overall best for adduct fallback (even if outside threshold)
                        if min_mz_error < best_overall_error:
                            best_overall_error = min_mz_error
                            best_overall_params = (num_strands, num_ag, z_test, nH, nC, nN, nO, nP,
                                                   mH_total, mC_total, mN_total, mO_total, mP_total, mAg_total, mDNA_total)

                        # Use 5.0 m/z threshold for baseline search
                        if min_mz_error < 5.0:
                            logger.info(f"BASELINE: Testing {strand_label} nAg={num_ag}, z={z_test} (mz_error={min_mz_error:.4f}, best_Qcl={best_qcl_for_debug}, best_mz={best_mz_for_debug:.2f})")
                            # Pass strand_label for complex mode (strand1, strand2, or complex)
                            complex_strand_label = strand_label if is_complex_mode else None
                            smart_comps = self.smart_n0_search(
                                num_strands, num_ag, z_test, dna_sequence, detected_centroid,
                                nH, nC, nN, nO, nP, mH_total, mC_total, mN_total, mO_total, mP_total, mAg_total, resolution,
                                custom_xna=custom_xna, strand_label=complex_strand_label,
                                conjugate_name=conjugate_name, conjugate_count=conjugate_count,
                                extra_conj_mass=extra_conj_mass, extra_conj_formula=extra_conj_formula
                            )
                            compositions.extend(smart_comps)
                            if smart_comps:
                                x0_err = smart_comps[0].get('abs_x0_error', 999)
                                logger.info(f"BASELINE: {strand_label} nAg={num_ag} -> X₀ error={x0_err:.4f}")
                        else:
                            # Log skipped nAg values in Complex mode for debugging
                            if is_complex_mode and num_ag >= nAg_min and num_ag <= min(nAg_max, 25):
                                logger.debug(f"BASELINE: SKIP {strand_label} nAg={num_ag} (mz_error={min_mz_error:.4f} >= 5.0)")
                    else:
                        # Traditional PPM-based filtering (old behavior)
                        # Try different Qcl values (cluster charge)
                        # Relationship: N0 (valence electrons) + Qcl = nAg

                        # COMPLEX MODE: For complex nucleic acid complexes, N0 = 0 always
                        # This means Qcl = nAg, so only test that single value
                        # Complex labels are "nd=1", "nd=2", "nd=3" (or legacy "complex")
                        is_complex_label = strand_label and (strand_label.startswith('nd=') or strand_label == 'complex')
                        if is_complex_mode or is_complex_label:
                            qcl_range: list[int] = [num_ag]  # Only Qcl = nAg (N0 = 0)
                            logger.debug(f"COMPLEX PPM MODE ({strand_label}): N0 = 0 only, Qcl = {num_ag}")
                        else:
                            qcl_range = list(range(0, num_ag + 1))

                        for qcl in qcl_range:
                            n0_valence = num_ag - qcl

                            # N0 can be 0 (DNA + Ag+ ions, no nanocluster)
                            # N0 >= 2 forms nanoclusters
                            if n0_valence < 0:
                                continue

                            # Calculate mass exactly as in MASS.py:
                            # mass = mP + mH + mC + mN + mO + mAg - (Qcl + z) * mH
                            # (For XNA: mDNA_total already contains custom XNA mass)
                            mass = mDNA_total + mAg_total - (qcl + z_test) * self.m_p
                            expected_mz = mass / z_test

                            mass_error_ppm = abs((expected_mz - mz) / mz * 1e6)

                            if mass_error_ppm < 200:
                                # Formula using element composition (for isotope pattern generation)
                                # This works for both DNA and XNA since we parsed the XNA formula to get element counts
                                neutral_formula_chem = f"C{nC}H{nH}N{nN}O{nO}P{nP}{extra_conj_formula}Ag{num_ag}"
                                nH_ion = nH - (qcl + z_test)
                                ion_formula = f"C{nC}H{nH_ion}N{nN}O{nO}P{nP}{extra_conj_formula}Ag{num_ag}"

                                # For display, use custom XNA name if provided
                                # Get strand_label for complex mode
                                complex_strand_label = strand_label if is_complex_mode else None
                                if custom_xna:
                                    xna_name = custom_xna['name']
                                    # For complex mode, show strand type in formula
                                    if complex_strand_label and complex_strand_label in ['strand1', 'strand2', 'complex']:
                                        if complex_strand_label == 'complex':
                                            neutral_formula = f"({xna_name}-complex)Ag{to_subscript(num_ag)}"
                                        else:
                                            neutral_formula = f"({xna_name}-{complex_strand_label})Ag{to_subscript(num_ag)}"
                                    else:
                                        neutral_formula = f"({xna_name}){to_subscript(num_strands)}Ag{to_subscript(num_ag)}"
                                else:
                                    neutral_formula = neutral_formula_chem

                                compositions.append({
                                    'type': 'nanocluster',
                                    'num_strands': num_strands,
                                    'strand_type': complex_strand_label,  # 'strand1', 'strand2', 'complex', or None
                                    'num_silver': num_ag,
                                    'qcl': qcl,
                                    'n0': n0_valence,
                                    'z': z_test,
                                    'formula': neutral_formula,  # Display neutral formula (like MASS.py)
                                    'ion_formula': ion_formula,  # Use deprotonated formula for isotope pattern
                                    'neutral_formula': neutral_formula,
                                    'adduct': '',
                                    'full_notation': f"{neutral_formula}-{qcl+z_test}H (z={z_test}, Qcl={qcl}, N0={n0_valence})",
                                    'expected_mz': expected_mz,
                                    'mass_error_ppm': mass_error_ppm,
                                    'x0_error': 999.0,
                                    'abs_x0_error': 999.0,
                                    'pattern_score': 0.0,
                                    'nH': nH, 'nC': nC, 'nN': nN, 'nO': nO, 'nP': nP,
                                    'custom_xna': custom_xna
                                })

        # PART 2: DNA/XNA-only compositions (no silver)
        # These are important when N0 would be unrealistically high
        # For complex mode, search strand1, strand2 (if different), and complex
        if is_complex_mode and custom_xna:
            # Complex mode: search individual strands and combined
            noag_configs = []
            if xna_strand1_composition:
                noag_configs.append(('strand1', 1, xna_strand1_composition))
            if xna_strand2_composition and not same_strands:
                noag_configs.append(('strand2', 1, xna_strand2_composition))
            noag_configs.append(('complex', 2, xna_composition_one))
        else:
            # Regular mode: search 1-3 strands
            noag_configs = [(f"{i}strand", i, None) for i in range(1, MAX_STRANDS + 1)]

        for config_label, num_strands, config_comp in noag_configs:
            if custom_xna:
                if is_complex_mode and config_comp is not None:
                    # Complex mode: use pre-configured composition
                    comp_to_use = config_comp
                    nH = comp_to_use.get('H', 0)
                    nC = comp_to_use.get('C', 0)
                    nN = comp_to_use.get('N', 0)
                    nO = comp_to_use.get('O', 0)
                    nP = comp_to_use.get('P', 0)

                    # Use pre-calculated full formula mass (handles all elements)
                    if config_label == 'strand1' and xna_strand1_mass:
                        mDNA = xna_strand1_mass
                    elif config_label == 'strand2' and xna_strand2_mass:
                        mDNA = xna_strand2_mass
                    elif xna_mass_one is not None:  # complex
                        mDNA = xna_mass_one
                    else:
                        continue  # Skip if no valid mass
                elif xna_composition_one is not None and xna_mass_one is not None:
                    # Regular XNA mode: multiply formula by num_strands
                    nH = xna_composition_one.get('H', 0) * num_strands
                    nC = xna_composition_one.get('C', 0) * num_strands
                    nN = xna_composition_one.get('N', 0) * num_strands
                    nO = xna_composition_one.get('O', 0) * num_strands
                    nP = xna_composition_one.get('P', 0) * num_strands

                    # Use pre-calculated full formula mass (handles all elements)
                    mDNA = xna_mass_one * num_strands
                else:
                    continue  # Skip if XNA composition not available
            elif dna_sequence:
                # Calculate standard DNA composition
                nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, num_strands)
                mH_total = self.m_p * nH
                mC_total = self.mC * nC
                mN_total = self.mN * nN
                mO_total = self.mO * nO
                mP_total = self.mP * nP
                mDNA = mH_total + mC_total + mN_total + mO_total + mP_total
            else:
                continue  # Skip if no valid sequence/formula

            z_values = [z_observed] if z_observed else [1, 2, 3, 4, 5, 6]

            for z_test in z_values:
                if z_test is None or z_test <= 0:
                    continue

                # DNA/XNA-only (no silver, no adducts)
                # Use relaxed threshold - these peaks may be asymmetric/non-Gaussian
                expected_mz = (mDNA - z_test * self.m_p) / z_test
                mass_error_ppm = abs((expected_mz - mz) / mz * 1e6)

                if mass_error_ppm < 1000:
                    # Use element composition for both DNA and XNA (for isotope pattern generation)
                    neutral_formula_chem = f"C{nC}H{nH}N{nN}O{nO}P{nP}"
                    nH_ion = nH - z_test
                    ion_formula = f"C{nC}H{nH_ion}N{nN}O{nO}P{nP}"

                    # For display, use custom XNA name if provided
                    if custom_xna:
                        xna_name = custom_xna['name']
                        # For complex mode, show strand type
                        if is_complex_mode and config_label in ['strand1', 'strand2', 'complex']:
                            if config_label == 'complex':
                                neutral_formula = f"({xna_name}-complex)"
                            else:
                                neutral_formula = f"({xna_name}-{config_label})"
                        else:
                            neutral_formula = f"({xna_name}){to_subscript(num_strands)}"
                    else:
                        neutral_formula = neutral_formula_chem

                    compositions.append({
                        'type': 'XNA Only' if custom_xna else 'DNA Only',
                        'num_strands': num_strands,
                        'strand_type': config_label if is_complex_mode else None,
                        'num_silver': 0,
                        'qcl': 0,
                        'n0': 0,
                        'z': z_test,
                        'formula': neutral_formula,
                        'ion_formula': ion_formula,
                        'neutral_formula': neutral_formula,
                        'adduct': '',
                        'full_notation': f"{neutral_formula} (z={z_test})",
                        'expected_mz': expected_mz,
                        'mass_error_ppm': mass_error_ppm,
                        'x0_error': 999.0,
                        'abs_x0_error': 999.0,
                        'pattern_score': 0.0,
                        'nH': nH, 'nC': nC, 'nN': nN, 'nO': nO, 'nP': nP,
                        'custom_xna': custom_xna
                    })

        # PART 3: DNA/XNA + Ag+ ions (no nanocluster) with adducts
        for num_strands in range(1, MAX_STRANDS + 1):  # 1-MAX_STRANDS DNA/XNA strands
            if custom_xna and xna_composition_one is not None and xna_mass_one is not None:
                # Use pre-parsed XNA composition (calculated from formula, same as DNA)
                nH = xna_composition_one.get('H', 0) * num_strands
                nC = xna_composition_one.get('C', 0) * num_strands
                nN = xna_composition_one.get('N', 0) * num_strands
                nO = xna_composition_one.get('O', 0) * num_strands
                nP = xna_composition_one.get('P', 0) * num_strands

                # Use pre-calculated full formula mass (handles all elements)
                mDNA = xna_mass_one * num_strands
            elif dna_sequence:
                # Calculate standard DNA composition
                nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, num_strands)

                mH_total = self.m_p * nH
                mC_total = self.mC * nC
                mN_total = self.mN * nN
                mO_total = self.mO * nO
                mP_total = self.mP * nP
                mDNA = mH_total + mC_total + mN_total + mO_total + mP_total
            else:
                continue  # Skip if no valid sequence/formula

            # Try different numbers of Ag+ ions coordinated to DNA
            for num_ag_ions in range(1, 6):  # 1-5 Ag+ ions
                z_values = [z_observed] if z_observed else [1, 2, 3, 4, 5, 6]

                for z_test in z_values:
                    if z_test is None or z_test <= 0:
                        continue

                    # Try different adduct combinations
                    # Extract mass and charge from (mass, charge) tuples
                    adduct_combinations = [
                        ('', 0, 0),  # No adduct (name, mass, charge)
                        ('NH4', self.adducts['NH4'][0], self.adducts['NH4'][1]),
                        ('2NH4', self.adducts['2NH4'][0], self.adducts['2NH4'][1]),
                        ('Na', self.adducts['Na'][0], self.adducts['Na'][1]),
                        ('2Na', self.adducts['2Na'][0], self.adducts['2Na'][1]),
                        ('Cl', self.adducts['Cl'][0], self.adducts['Cl'][1]),      # Adding Cl- anion
                        ('2Cl', self.adducts['2Cl'][0], self.adducts['2Cl'][1]),  # Adding 2 Cl- anions
                    ]
                    ag_mass = self.adducts['Ag'][0]

                    for adduct_name, adduct_mass, adduct_charge in adduct_combinations:
                        # Calculate mass: DNA + Ag+ ions + adducts - protons_removed
                        total_mass = mDNA + (num_ag_ions * ag_mass) + adduct_mass

                        # Protons removed = Qcl + z + adduct_charge
                        protons_removed = num_ag_ions + z_test + adduct_charge

                        # m/z = (total_mass - protons_removed * mH) / z
                        expected_mz = (total_mass - protons_removed * self.m_p) / z_test

                        mass_error_ppm = abs((expected_mz - mz) / mz * 1e6)

                        # Use relaxed threshold for DNA/XNA+Ag ions (non-cluster, may be asymmetric)
                        if mass_error_ppm < 1000:
                            # Build ion formula using element composition (for isotope pattern generation)
                            # This works for both DNA and XNA since we parsed the XNA formula to get element counts
                            nH_ion = nH - protons_removed
                            ion_formula = f"C{nC}H{nH_ion}N{nN}O{nO}P{nP}"
                            if num_ag_ions > 0:
                                ion_formula += f"Ag{num_ag_ions}"
                            # Add adducts to ion formula (convert "2Cl" -> "Cl2", "NH4" -> "NH4", etc.)
                            if adduct_name:
                                adduct_formula = self.adduct_name_to_formula(adduct_name)
                                ion_formula += adduct_formula

                            # Build neutral formula for display (consistent Ag{n} format with subscript)
                            if custom_xna:
                                # XNA + Ag formula
                                xna_name = custom_xna['name']
                                neutral_formula = f"({xna_name}){to_subscript(num_strands)}Ag{to_subscript(num_ag_ions)}"
                                if adduct_name:
                                    neutral_formula += f"+{adduct_name}"
                            else:
                                # DNA + Ag formula (use subscript for Ag count)
                                neutral_formula = f"C{nC}H{nH}N{nN}O{nO}P{nP}Ag{to_subscript(num_ag_ions)}"
                                if adduct_name:
                                    neutral_formula += f"+{adduct_name}"

                            # For display: displayed_qcl = qcl + adduct_charge
                            displayed_qcl = num_ag_ions + adduct_charge

                            compositions.append({
                                'type': 'XNA+Ag ion' if custom_xna else 'DNA+Ag ion',
                                'num_strands': num_strands,
                                'num_silver': num_ag_ions,
                                'qcl': num_ag_ions,  # Internal Qcl (N₀ + Qcl = nAg always)
                                'displayed_qcl': displayed_qcl,  # For display: qcl + adduct_charge
                                'n0': 0,             # No valence electrons (not a nanocluster)
                                'z': z_test,
                                'formula': neutral_formula,  # Display neutral formula
                                'ion_formula': ion_formula,  # Use for isotope pattern
                                'neutral_formula': neutral_formula,
                                'adduct': adduct_name,
                                'adduct_charge': adduct_charge,
                                'full_notation': f"{neutral_formula} (z={z_test}, Qcl={displayed_qcl})",
                                'expected_mz': expected_mz,
                                'mass_error_ppm': mass_error_ppm,
                                'x0_error': 999.0,
                                'abs_x0_error': 999.0,
                                'nH': nH, 'nC': nC, 'nN': nN, 'nO': nO, 'nP': nP,
                                'custom_xna': custom_xna
                            })

        # Sort by mass error (preliminary; final ranking uses |X0 error| after isotope matching)
        compositions.sort(key=lambda x: x['mass_error_ppm'])

        # For compositions marked as base matches, add ALL possible N0 values
        # This ensures we can find the best X0 match across all N0 values
        additional_compositions = []
        processed_formulas = set()  # Track which formulas we've expanded

        for comp in compositions:
            if comp['type'] == 'nanocluster' and comp.get('_base_match'):
                num_strands = comp['num_strands']
                num_ag = comp['num_silver']
                z_test = comp['z']
                qcl = comp['qcl']

                # Create a unique key for this formula (independent of Qcl/N0)
                formula_key = (num_strands, num_ag, z_test)
                if formula_key in processed_formulas:
                    continue  # Already expanded this formula
                processed_formulas.add(formula_key)

                logger.debug(f"Expanding all N0 for: strands={num_strands}, nAg={num_ag}, z={z_test}")

                # Add ALL Qcl values (all possible N0) for X0-based comparison
                # COMPLEX MODE: For complex, N0 = 0 always, so only test Qcl = nAg
                if is_complex_mode:
                    qcl_expand_range = [num_ag]  # Only Qcl = nAg (N0 = 0)
                    logger.debug(f"COMPLEX MODE: Skipping N0 expansion, only N0=0 (Qcl={num_ag})")
                else:
                    qcl_expand_range = range(0, num_ag + 1)

                for qcl_neighbor in qcl_expand_range:
                    # Check if this Qcl already exists
                    exists = any(
                        c['type'] == 'nanocluster' and
                        c['num_strands'] == num_strands and
                        c['num_silver'] == num_ag and
                        c['z'] == z_test and
                        c['qcl'] == qcl_neighbor
                        for c in compositions
                    )
                    if exists:
                        continue  # Already have this one

                    # Check if this neighbor already exists
                    exists = any(
                        c['type'] == 'nanocluster' and
                        c['num_strands'] == num_strands and
                        c['num_silver'] == num_ag and
                        c['z'] == z_test and
                        c['qcl'] == qcl_neighbor
                        for c in compositions
                    )

                    if not exists:
                        # Generate this neighbor composition
                        n0_valence = num_ag - qcl_neighbor
                        if n0_valence < 0:  # N0 can be 0 (DNA + Ag+)
                            continue
                        if not dna_sequence:
                            continue  # Skip if no DNA sequence

                        nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, num_strands)
                        mH_total = self.m_p * nH
                        mC_total = self.mC * nC
                        mN_total = self.mN * nN
                        mO_total = self.mO * nO
                        mP_total = self.mP * nP
                        mAg_total = self.mAg * num_ag

                        mass = mP_total + mH_total + mC_total + mN_total + mO_total + mAg_total - (qcl_neighbor + z_test) * self.m_p
                        expected_mz = mass / z_test
                        mass_error_ppm = abs((expected_mz - mz) / mz * 1e6)

                        neutral_formula = f"C{nC}H{nH}N{nN}O{nO}P{nP}Ag{num_ag}"
                        nH_ion = nH - (qcl_neighbor + z_test)
                        ion_formula = f"C{nC}H{nH_ion}N{nN}O{nO}P{nP}Ag{num_ag}"

                        additional_compositions.append({
                            'type': 'nanocluster',
                            'num_strands': num_strands,
                            'num_silver': num_ag,
                            'qcl': qcl_neighbor,
                            'n0': n0_valence,
                            'z': z_test,
                            'formula': neutral_formula,
                            'ion_formula': ion_formula,
                            'neutral_formula': neutral_formula,
                            'adduct': '',
                            'full_notation': f"{neutral_formula}-{qcl_neighbor+z_test}H (z={z_test}, Qcl={qcl_neighbor}, N0={n0_valence})",
                            'expected_mz': expected_mz,
                            'mass_error_ppm': mass_error_ppm,
                            'x0_error': 999.0,
                            'abs_x0_error': 999.0,
                            'nH': nH, 'nC': nC, 'nN': nN, 'nO': nO, 'nP': nP,
                            'custom_xna': custom_xna
                        })

        # Add the neighbors to our compositions list
        compositions.extend(additional_compositions)

        # Debug: Show all N0 values generated
        if detected_centroid is not None:
            n0_values = sorted(set([c['n0'] for c in compositions if c['type'] == 'nanocluster']))
            logger.debug(f"Generated {len(compositions)} total compositions with N0 values: {n0_values}")
            logger.debug(f"Using X0-based comparison (detected_centroid={detected_centroid:.4f})")

        # Print summary of strand numbers found
        strand_counts: dict[int, int] = {}
        for comp in compositions:
            if comp['type'] == 'nanocluster':
                strands = comp.get('num_strands', 0)
                strand_counts[strands] = strand_counts.get(strands, 0) + 1

        if strand_counts:
            logger.debug("Nanocluster compositions found by strand number:")
            for strands in sorted(strand_counts.keys()):
                logger.debug(f"  Strands={strands}: {strand_counts[strands]} compositions")
        else:
            logger.debug("No nanocluster compositions found")

        # Return all compositions - Qcl±1 selection will happen after isotope matching
        # in refine_compositions_with_isotope_matching()
        return compositions

    def smart_n0_search(self, num_strands: int, num_ag: int, z_test: int, dna_sequence: str,
                        detected_centroid: float, nH: int, nC: int, nN: int, nO: int, nP: int,
                        mH_total: float, mC_total: float, mN_total: float, mO_total: float,
                        mP_total: float, mAg_total: float, resolution: int = 20000,
                        custom_xna: Optional[dict] = None, strand_label: Optional[str] = None,
                        conjugate_name: Optional[str] = None, conjugate_count: int = 0,
                        extra_conj_mass: float = 0.0, extra_conj_formula: str = "") -> list[dict]:
        """
        Smart search for best N0 by calculating X0 error and stopping when it increases.

        Strategy:
        1. Find starting Qcl based on m/z proximity
        2. Calculate X0 for that N0
        3. Search in both directions (increasing/decreasing N0)
        4. Stop when X0 error increases in both directions

        Returns: list of composition dictionaries with calculated X0 errors (pattern scores calculated later)
        """
        compositions = []

        # Calculate total DNA/XNA mass (both calculated from element composition)
        mDNA_total = mH_total + mC_total + mN_total + mO_total + mP_total + extra_conj_mass
        mode_label = "XNA" if custom_xna else "DNA"
        logger.debug(f"{mode_label} MODE: Calculated from elements = {mDNA_total:.2f} Da (extra_conj_mass={extra_conj_mass:.4f})")

        # MASS VALIDATION: For no-adduct compositions, check if nAg is reasonable
        # The observed mass (m/z × z) includes hydrogen loss: observed = DNA + nAg×Ag - (Qcl + z)×H
        # For complex mode: Qcl = nAg, so remaining = observed + (nAg + z)×H - DNA - nAg×Ag ≈ 0
        # For regular mode: Qcl varies, so we use a looser check
        observed_mass = detected_centroid * z_test

        # Check if we're in complex mode
        is_complex_flag = custom_xna and custom_xna.get('is_complex', False)
        is_complex_label = strand_label and (strand_label.startswith('nd=') or strand_label == 'complex')
        is_complex_validation = is_complex_flag or is_complex_label

        if is_complex_validation:
            # For complex mode: account for hydrogen loss (Qcl = nAg)
            # remaining = observed + (nAg + z)×H - DNA - nAg×Ag
            remaining_mass = observed_mass + (num_ag + z_test) * self.m_p - mDNA_total - mAg_total
            # Allow some tolerance for mass measurement error
            if remaining_mass < -50:  # Allow 50 Da tolerance
                logger.debug(f"COMPLEX MASS VALIDATION FAILED: remaining={remaining_mass:.2f} < -50 (nAg={num_ag})")
                return compositions
        else:
            # For regular mode: account for hydrogen loss
            # observed_mass = (neutral - protons_removed×H)
            # protons_removed = qcl + z, where qcl can range from 0 to nAg
            # Check with max protons_removed (qcl=nAg) since that gives most tolerance
            max_protons_removed = num_ag + z_test
            remaining_mass = observed_mass + max_protons_removed * self.m_p - mDNA_total - mAg_total
            # Allow tolerance for mass measurement and isotope pattern width
            if remaining_mass < -30:  # Allow 30 Da tolerance for adducts
                logger.debug(f"MASS VALIDATION FAILED: remaining={remaining_mass:.2f} < -30 (nAg={num_ag}, DNA={mDNA_total:.2f}, Ag={mAg_total:.2f})")
                return compositions  # Return empty list - impossible composition

        # Calculate best Qcl directly using algebra (no loop needed!)
        # m/z = (neutral_mass - (qcl + z) * mH) / z
        # Solving for qcl: qcl = (neutral_mass - m/z * z) / mH - z
        neutral_mass = mDNA_total + mAg_total
        best_qcl_float = (neutral_mass - detected_centroid * z_test) / self.m_p - z_test
        best_qcl = int(round(best_qcl_float))
        best_qcl = max(0, min(num_ag, best_qcl))  # Clamp to valid range [0, num_ag]

        # Calculate actual m/z error for this Qcl
        mass = neutral_mass - (best_qcl + z_test) * self.m_p
        expected_mz = mass / z_test
        best_mz_error = abs(expected_mz - detected_centroid)

        logger.debug(f"Best Qcl = {best_qcl} (calculated directly, no loop)")
        logger.debug(f"expected_mz={expected_mz:.4f}, detected={detected_centroid:.4f}, error={best_mz_error:.4f}")

        # Now search in both directions from best_qcl
        # Direction 1: Decrease Qcl (increase N0)
        # Direction 2: Increase Qcl (decrease N0)

        def calculate_x0_for_composition(qcl):
            """Helper to calculate X0 for a given Qcl"""
            n0_valence = num_ag - qcl
            if n0_valence < 0:
                return None

            # Calculate expected m/z (same formula for DNA and XNA)
            mass = mDNA_total + mAg_total - (qcl + z_test) * self.m_p
            expected_mz = mass / z_test
            mass_error_ppm = abs((expected_mz - detected_centroid) / detected_centroid * 1e6)

            # Generate formulas using element composition (for isotope pattern generation)
            # This works for both DNA and XNA since we parsed the XNA formula to get element counts
            if num_ag > 0:
                neutral_formula_chem = f"C{nC}H{nH}N{nN}O{nO}P{nP}{extra_conj_formula}Ag{num_ag}"
                nH_ion = nH - (qcl + z_test)
                ion_formula = f"C{nC}H{nH_ion}N{nN}O{nO}P{nP}{extra_conj_formula}Ag{num_ag}"
            else:
                # No silver - DNA/conjugate only
                neutral_formula_chem = f"C{nC}H{nH}N{nN}O{nO}P{nP}{extra_conj_formula}"
                nH_ion = nH - (qcl + z_test)  # For nAg=0, qcl=0, so protons_removed = z
                ion_formula = f"C{nC}H{nH_ion}N{nN}O{nO}P{nP}{extra_conj_formula}"

            # For display, use custom XNA name if provided
            if custom_xna:
                xna_name = custom_xna['name']
                # For complex mode, show nd (number of complexes) in formula
                if strand_label and (strand_label.startswith('nd=') or strand_label in ['strand1', 'strand2', 'complex']):
                    if strand_label.startswith('nd='):
                        # Extract nd value: "nd=1" -> 1, "nd=2" -> 2
                        nd_value = strand_label.split('=')[1]
                        neutral_formula = f"({xna_name}){to_subscript(int(nd_value))}Ag{to_subscript(num_ag)}"
                    elif strand_label == 'complex':
                        neutral_formula = f"({xna_name})Ag{to_subscript(num_ag)}"
                    else:
                        neutral_formula = f"({xna_name}-{strand_label})Ag{to_subscript(num_ag)}"
                else:
                    neutral_formula = f"({xna_name}){to_subscript(num_strands)}Ag{to_subscript(num_ag)}"
            else:
                # Build DNA formula with conjugate notation if present
                if conjugate_name and conjugate_count > 0:
                    unconjugated = num_strands - conjugate_count
                    if unconjugated > 0:
                        # Mixed: some strands have conjugate, some don't
                        unconj_part = f"(DNA){to_subscript(unconjugated)}" if unconjugated > 1 else "(DNA)"
                        conj_part = f"(DNA-{conjugate_name}){to_subscript(conjugate_count)}" if conjugate_count > 1 else f"(DNA-{conjugate_name})"
                        strand_part = f"{unconj_part}{conj_part}"
                    else:
                        # All strands conjugated — each strand has 1 conjugate
                        strand_part = f"(DNA-{conjugate_name}){to_subscript(num_strands)}"
                    if num_ag > 0:
                        neutral_formula = f"{strand_part}[Ag{to_subscript(num_ag)}]"
                    else:
                        # DNA-conjugate only (no silver)
                        neutral_formula = f"{strand_part}"
                else:
                    neutral_formula = neutral_formula_chem

            # Generate isotope pattern to get X0 (same for DNA and XNA)
            try:
                pattern = self.generate_isotope_pattern(ion_formula, z_test, resolution=resolution)
                if pattern and 'gaussian_mz' in pattern and len(pattern['gaussian_mz']) > 0:
                    # Use smooth Gaussian pattern for theo_x0 (same method as exp_x0)
                    theo_mz_gaussian = np.array(pattern['gaussian_mz'])
                    theo_int_gaussian = np.array(pattern['gaussian_intensity'])

                    if len(theo_mz_gaussian) > 0 and np.sum(theo_int_gaussian) > 0:
                        # Fit Gaussian to smooth theoretical pattern to extract x0 parameter
                        # This matches how exp_x0 is calculated from experimental data
                        theo_fit_result = self.gaussian_fit_centroid(theo_mz_gaussian, theo_int_gaussian)
                        if theo_fit_result and theo_fit_result[0] is not None:
                            theo_x0 = theo_fit_result[0]
                        else:
                            # Fallback to weighted average if Gaussian fit fails
                            theo_x0 = np.sum(theo_mz_gaussian * theo_int_gaussian) / np.sum(theo_int_gaussian)
                        logger.debug(f"Theo X0 (Gaussian fit): {theo_x0:.4f}")

                        # X0 error = |theo_x0 - exp_x0|
                        abs_x0_error = abs(theo_x0 - detected_centroid)
                        logger.debug(f"Exp X0: {detected_centroid:.4f}, |X0 error|: {abs_x0_error:.4f} m/z")
                    else:
                        return None
                else:
                    return None
            except Exception as e:
                logger.warning(f"Could not generate pattern for {ion_formula}: {e}")
                return None

            # Build full notation
            full_notation = f"{neutral_formula}-{qcl+z_test}H (z={z_test}, Qcl={qcl}, N0={n0_valence})"

            comp_type = self.determine_composition_type(
                num_ag, n0_valence, strand_label=strand_label,
                custom_xna=custom_xna, conjugate_name=conjugate_name,
                conjugate_count=conjugate_count)

            return {
                'type': comp_type,
                'num_strands': num_strands,
                'strand_type': strand_label,  # 'strand1', 'strand2', 'complex', or None
                'num_silver': num_ag,
                'qcl': qcl,
                'n0': n0_valence,
                'z': z_test,
                'formula': neutral_formula,
                'ion_formula': ion_formula,
                'neutral_formula': neutral_formula,
                'adduct': '',
                'adduct_mass': 0.0,
                'adduct_charge': 0,
                'conjugate': conjugate_name if conjugate_name and conjugate_count > 0 else None,
                'conjugate_count': conjugate_count if conjugate_name and conjugate_count > 0 else 0,
                'dna_neutral_mass': mDNA_total,  # For adduct validation
                'full_notation': full_notation,
                'expected_mz': expected_mz,
                'mass_error_ppm': mass_error_ppm,
                'x0_error': abs_x0_error,  # |theo_x0 - exp_x0|
                'abs_x0_error': abs_x0_error,
                'theo_x0': theo_x0,
                'exp_x0': detected_centroid,  # Store exp_x0 used in x0_error calculation
                'pattern_score': 0.0,
                'theo_mz': pattern.get('gaussian_mz', []),
                'theo_intensity': pattern.get('gaussian_intensity', []),
                'nH': nH, 'nC': nC, 'nN': nN, 'nO': nO, 'nP': nP,
                'extra_conj_mass': extra_conj_mass,
                'extra_conj_formula': extra_conj_formula,
                'custom_xna': custom_xna
            }

        # COMPLEX MODE: For complex nucleic acid complexes, N0 = 0 always
        # This means Qcl = nAg (all silver atoms are Ag+, no reduced Ag0)
        # Skip the N0 search and only test Qcl = nAg
        # Complex labels are "nd=1", "nd=2", "nd=3" (or legacy "complex")
        # Also check custom_xna.is_complex for DNA-only Complex mode (no strand_label)
        is_complex_label = strand_label and (strand_label.startswith('nd=') or strand_label == 'complex')
        is_complex_flag = custom_xna and custom_xna.get('is_complex', False)
        is_complex = is_complex_label or is_complex_flag
        if is_complex:
            logger.debug(f"COMPLEX MODE ({strand_label}): N0 = 0 (skipping N0 search), Qcl = nAg = {num_ag}")
            complex_comp = calculate_x0_for_composition(num_ag)  # Qcl = nAg means N0 = 0
            if complex_comp:
                compositions.append(complex_comp)
                logger.debug(f"COMPLEX: N0=0, Qcl={num_ag}, |X0_error|={complex_comp['x0_error']:.4f}")
            return compositions

        # Calculate for starting point
        center_comp = calculate_x0_for_composition(best_qcl)
        if center_comp:
            compositions.append(center_comp)
            logger.debug(f"Starting N0={center_comp['n0']}, |X0_error|={center_comp['x0_error']:.4f}")
        else:
            return compositions

        # OPTIMIZED BIDIRECTIONAL SEARCH with early stopping
        # Stop IMMEDIATELY when X0 error increases (matches fast IsoSpecPy version)
        # This reduces patterns from 20+ to ~3-5, making analysis much faster
        best_error = center_comp['abs_x0_error']

        # Search left (decreasing Qcl, increasing N0)
        prev_error = best_error
        for qcl in range(best_qcl - 1, -1, -1):
            comp = calculate_x0_for_composition(qcl)
            if comp is None:
                break
            compositions.append(comp)
            logger.debug(f"<- N0={comp['n0']}, |X0_error|={comp['x0_error']:.4f}")
            # Stop IMMEDIATELY if error is increasing
            if comp['abs_x0_error'] > prev_error:
                logger.debug(f"<- Early stop: error increasing")
                break
            prev_error = comp['abs_x0_error']
            if comp['abs_x0_error'] < best_error:
                best_error = comp['abs_x0_error']

        # Search right (increasing Qcl, decreasing N0)
        prev_error = center_comp['abs_x0_error']  # Reset for right search
        for qcl in range(best_qcl + 1, num_ag + 1):
            comp = calculate_x0_for_composition(qcl)
            if comp is None:
                break
            compositions.append(comp)
            logger.debug(f"-> N0={comp['n0']}, |X0_error|={comp['x0_error']:.4f}")
            # Stop IMMEDIATELY if error is increasing
            if comp['abs_x0_error'] > prev_error:
                logger.debug(f"-> Early stop: error increasing")
                break
            prev_error = comp['abs_x0_error']
            if comp['abs_x0_error'] < best_error:
                best_error = comp['abs_x0_error']

        logger.debug(f"Optimized search: found {len(compositions)} compositions, best error={best_error:.4f}")

        return compositions

    def smart_n0_search_with_adduct(self, num_strands: int, num_ag: int, z_test: int, dna_sequence: str,
                                      detected_centroid: float, nH: int, nC: int, nN: int, nO: int, nP: int,
                                      mH_total: float, mC_total: float, mN_total: float, mO_total: float,
                                      mP_total: float, mAg_total: float, adduct_name: str, adduct_mass: float,
                                      adduct_charge: int, resolution: int = 20000, custom_xna: Optional[dict] = None,
                                      strand_label: Optional[str] = None,
                                      conjugate_name: Optional[str] = None, conjugate_count: int = 0,
                                      extra_conj_mass: float = 0.0, extra_conj_formula: str = "") -> list[dict]:
        """
        Smart N0 search WITH ADDUCT using same approach as baseline:

        Step 1: Find best starting Qcl by theoretical m/z (FAST - no isotope patterns!)
        Step 2: Calculate X0 for that best Qcl (1 isotope pattern)
        Step 3: Search bidirectionally, STOP when X0 error increases

        Result: Only ~3-5 isotope patterns instead of 10+, making it much faster!

        Returns: list of composition dictionaries with calculated X0 errors
        """
        compositions = []

        # Calculate total DNA/XNA mass (both calculated from element composition)
        mDNA_total = mH_total + mC_total + mN_total + mO_total + mP_total + extra_conj_mass
        mode_label = "XNA" if custom_xna else "DNA"
        logger.debug(f"{mode_label} MODE: Calculated from elements = {mDNA_total:.2f} Da (extra_conj_mass={extra_conj_mass:.4f})")

        # STEP 1: Calculate best Qcl directly using algebra (no loop needed!)
        # m/z = (neutral_mass - (qcl + z + adduct_charge) * mH) / z
        # Solving for qcl: qcl = (neutral_mass - m/z * z) / mH - z - adduct_charge
        neutral_mass = mDNA_total + mAg_total + adduct_mass
        best_qcl_float = (neutral_mass - detected_centroid * z_test) / self.m_p - z_test - adduct_charge
        best_qcl = int(round(best_qcl_float))
        # N₀ = nAg - Qcl must be ≥ 0, so max_qcl = nAg
        max_qcl = num_ag
        best_qcl = max(0, min(max_qcl, best_qcl))  # Clamp to valid range [0, nAg]

        # Calculate actual m/z error for this Qcl
        mass = neutral_mass - (best_qcl + z_test + adduct_charge) * self.m_p
        expected_mz = mass / z_test
        best_mz_error = abs(expected_mz - detected_centroid)

        logger.debug(f"Best Qcl (with adduct {adduct_name}) = {best_qcl} (calculated directly, no loop)")
        logger.debug(f"expected_mz={expected_mz:.4f}, detected={detected_centroid:.4f}, error={best_mz_error:.4f}")

        # STEP 2: Now search bidirectionally from best_qcl, calculating X0 only when needed
        def calculate_x0_for_composition(qcl):
            """Helper to calculate X0 for a given Qcl with adduct"""
            # Formula: N₀ + Qcl = nAg (always, regardless of adducts)
            # Therefore: N₀ = nAg - Qcl
            # Adduct charge affects protons_removed formula, not the N₀ relationship
            n0_valence = num_ag - qcl
            if n0_valence < 0:
                return None

            # Calculate expected m/z (same formula for DNA and XNA)
            # protons_removed = Qcl + z + adduct_charge
            mass = mDNA_total + mAg_total + adduct_mass - (qcl + z_test + adduct_charge) * self.m_p
            expected_mz = mass / z_test
            mass_error_ppm = abs((expected_mz - detected_centroid) / detected_centroid * 1e6)

            # Build formulas with adduct using element composition (for isotope pattern generation)
            # This works for both DNA and XNA since we parsed the XNA formula to get element counts
            # Special case: when nAg=0, don't include Ag in formula
            if num_ag > 0:
                neutral_formula_chem = f"C{nC}H{nH}N{nN}O{nO}P{nP}{extra_conj_formula}Ag{num_ag}+{adduct_name}"
                ag_part = f"Ag{num_ag}"
            else:
                neutral_formula_chem = f"C{nC}H{nH}N{nN}O{nO}P{nP}{extra_conj_formula}+{adduct_name}"
                ag_part = ""

            protons_removed = qcl + z_test + adduct_charge
            nH_ion = nH - protons_removed

            # Ion formula for isotope pattern
            # Convert adduct name (e.g., '2Cl') to chemical formula (e.g., 'Cl2')
            adduct_formula = self.adduct_name_to_formula(adduct_name)
            if num_ag > 0:
                ion_formula = f"C{nC}H{nH_ion}N{nN}O{nO}P{nP}{extra_conj_formula}Ag{num_ag}{adduct_formula}"
            else:
                ion_formula = f"C{nC}H{nH_ion}N{nN}O{nO}P{nP}{extra_conj_formula}{adduct_formula}"

            # For display, use custom XNA name if provided
            if custom_xna:
                xna_name = custom_xna['name']
                # For complex mode, show strand type in formula
                if strand_label and strand_label in ['strand1', 'strand2', 'complex']:
                    strand_suffix = '-complex' if strand_label == 'complex' else f'-{strand_label}'
                    if num_ag > 0:
                        neutral_formula = f"({xna_name}{strand_suffix})Ag{to_subscript(num_ag)}+{adduct_name}"
                    else:
                        neutral_formula = f"({xna_name}{strand_suffix})+{adduct_name}"
                else:
                    if num_ag > 0:
                        neutral_formula = f"({xna_name}){to_subscript(num_strands)}Ag{to_subscript(num_ag)}+{adduct_name}"
                    else:
                        neutral_formula = f"({xna_name}){to_subscript(num_strands)}+{adduct_name}"
            else:
                # Build DNA formula with conjugate notation if present
                if conjugate_name and conjugate_count > 0:
                    unconjugated = num_strands - conjugate_count
                    # Format adduct name: "2Cl" -> "Cl₂"
                    formatted_adduct = format_adduct_name(adduct_name)
                    # Calculate displayed_qcl for formula
                    displayed_qcl_formula = qcl + adduct_charge
                    if unconjugated > 0:
                        # Mixed: some strands have conjugate, some don't
                        unconj_part = f"(DNA){to_subscript(unconjugated)}" if unconjugated > 1 else "(DNA)"
                        conj_part = f"(DNA-{conjugate_name}){to_subscript(conjugate_count)}" if conjugate_count > 1 else f"(DNA-{conjugate_name})"
                        strand_part = f"{unconj_part}{conj_part}"
                    else:
                        # All strands conjugated — each strand has 1 conjugate
                        strand_part = f"(DNA-{conjugate_name}){to_subscript(num_strands)}"
                    if num_ag > 0:
                        neutral_formula = f"{strand_part}[Ag{to_subscript(num_ag)}{formatted_adduct}]{to_superscript(str(displayed_qcl_formula) + '+')}"
                    else:
                        neutral_formula = f"{strand_part}+{formatted_adduct}"
                else:
                    neutral_formula = neutral_formula_chem

            # Generate isotope pattern to get X0 (same for DNA and XNA)
            pattern_generated = False
            gaussian_mz_shifted = []
            gaussian_intensity = []
            theo_x0_final = None
            try:
                pattern = self.generate_isotope_pattern(ion_formula, z_test, resolution=resolution)
                if pattern and 'gaussian_mz' in pattern and len(pattern['gaussian_mz']) > 0:
                    # Use smooth Gaussian pattern for theo_x0 (same method as exp_x0)
                    theo_mz_gaussian = np.array(pattern['gaussian_mz'])
                    theo_int_gaussian = np.array(pattern['gaussian_intensity'])

                    if len(theo_mz_gaussian) > 0 and np.sum(theo_int_gaussian) > 0:
                        # Fit Gaussian to smooth theoretical pattern to extract x0 parameter
                        # This matches how exp_x0 is calculated from experimental data
                        theo_fit_result = self.gaussian_fit_centroid(theo_mz_gaussian, theo_int_gaussian)
                        if theo_fit_result and theo_fit_result[0] is not None:
                            theo_x0 = theo_fit_result[0]
                        else:
                            # Fallback to weighted average if Gaussian fit fails
                            theo_x0 = np.sum(theo_mz_gaussian * theo_int_gaussian) / np.sum(theo_int_gaussian)

                        # X0 error = |theo_x0 - exp_x0|
                        abs_x0_error = abs(theo_x0 - detected_centroid)
                        theo_x0_final = theo_x0
                        gaussian_mz = pattern.get('gaussian_mz', [])
                        gaussian_intensity = pattern.get('gaussian_intensity', [])
                        logger.debug(f"Theo X0: {theo_x0:.4f}, Exp X0: {detected_centroid:.4f}, X0 error: {abs_x0_error:.4f} m/z")
                        pattern_generated = True
                        # No early stopping - search ALL N₀ values to find global minimum
            except Exception as e:
                pass  # Will use m/z error as fallback

            # If pattern failed, use m/z error as proxy for X0 error
            if not pattern_generated:
                logger.debug(f"Pattern failed for qcl={qcl}, using m/z error as X0 proxy")
                # Use expected_mz error as approximation |theo_x0 - exp_x0|
                abs_x0_error = abs(expected_mz - detected_centroid)
                theo_x0_final = expected_mz
                gaussian_mz = []
                gaussian_intensity = []
                # No early stopping - search ALL N₀ values to find global minimum

            # For display: displayed_qcl = qcl + adduct_charge
            displayed_qcl = qcl + adduct_charge

            # Build full notation
            full_notation = f"{neutral_formula}-{protons_removed}H (z={z_test}, Qcl={displayed_qcl}, N0={n0_valence})"

            comp_type = self.determine_composition_type(
                num_ag, n0_valence, strand_label=strand_label,
                custom_xna=custom_xna)

            return {
                'type': comp_type,
                'num_strands': num_strands,
                'strand_type': strand_label,  # 'strand1', 'strand2', 'complex', or None
                'num_silver': num_ag,
                'qcl': qcl,  # Internal Qcl (N₀ + Qcl = nAg always)
                'displayed_qcl': displayed_qcl,  # For display: qcl + adduct_charge
                'n0': n0_valence,
                'z': z_test,
                'formula': neutral_formula,
                'ion_formula': ion_formula,
                'neutral_formula': neutral_formula,
                'adduct': adduct_name,
                'conjugate': conjugate_name if conjugate_name and conjugate_count > 0 else None,
                'conjugate_count': conjugate_count if conjugate_name and conjugate_count > 0 else 0,
                'dna_neutral_mass': mDNA_total,  # For adduct validation
                'full_notation': full_notation,
                'expected_mz': expected_mz,
                'mass_error_ppm': mass_error_ppm,
                'x0_error': abs_x0_error,  # Absolute X0 error = |theo_x0 - exp_x0|
                'abs_x0_error': abs_x0_error,
                'exp_x0': detected_centroid,  # Store exp_x0 used in x0_error calculation
                'pattern_score': 0.0,
                'theo_x0': theo_x0_final,  # Shifted theo_x0 (after mass correction)
                # Store theoretical pattern (mass-corrected, no additional alignment shift)
                'theo_mz': gaussian_mz,
                'theo_intensity': gaussian_intensity,
                'nH': nH, 'nC': nC, 'nN': nN, 'nO': nO, 'nP': nP,
                'extra_conj_mass': extra_conj_mass,
                'extra_conj_formula': extra_conj_formula,
                # Store custom_xna and adduct_mass for later use in refine_compositions_with_isotope_matching
                'custom_xna': custom_xna,
                'adduct_mass': adduct_mass,
                'adduct_charge': adduct_charge
            }

        # COMPLEX MODE: For complex nucleic acid complexes, N0 = 0 always
        # This means Qcl = nAg (all silver atoms are Ag+, no reduced Ag0)
        # Skip the N0 search and only test Qcl = nAg
        # Complex labels are "nd=1", "nd=2", "nd=3" (or legacy "complex")
        # Also check custom_xna.is_complex flag for DNA-only Complex mode
        is_complex_label = strand_label and (strand_label.startswith('nd=') or strand_label == 'complex')
        is_complex_flag = custom_xna and custom_xna.get('is_complex', False)
        is_complex = is_complex_label or is_complex_flag
        if is_complex:
            logger.debug(f"COMPLEX MODE (with {adduct_name}, {strand_label}): N0 = 0 (skipping N0 search), Qcl = nAg = {num_ag}")
            complex_comp = calculate_x0_for_composition(num_ag)  # Qcl = nAg means N0 = 0
            if complex_comp:
                compositions.append(complex_comp)
                logger.debug(f"[Adduct {adduct_name}] COMPLEX: N0=0, Qcl={num_ag}, |X0_error|={complex_comp['x0_error']:.4f}")
            return compositions

        # Calculate for starting point (best Qcl by m/z)
        logger.debug(f"smart_n0_search_with_adduct: num_ag={num_ag}, best_qcl={best_qcl}, adduct={adduct_name}")
        center_comp = calculate_x0_for_composition(best_qcl)
        if center_comp:
            compositions.append(center_comp)
            logger.debug(f"[Adduct {adduct_name}] Starting N0={center_comp['n0']}, |X0_error|={center_comp['x0_error']:.4f}")
        else:
            logger.debug(f"calculate_x0_for_composition returned None for qcl={best_qcl}")
            return compositions

        # OPTIMIZED BIDIRECTIONAL SEARCH with early stopping
        # Stop IMMEDIATELY when X0 error increases (matches fast IsoSpecPy version)
        # This reduces patterns from 20+ to ~3-5, making XNA analysis much faster
        best_error = center_comp['abs_x0_error']

        # Search left (decreasing Qcl, increasing N0)
        prev_error = best_error
        for qcl in range(best_qcl - 1, -1, -1):
            comp = calculate_x0_for_composition(qcl)
            if comp is None:
                break
            compositions.append(comp)
            logger.debug(f"[Adduct {adduct_name}] <- N0={comp['n0']}, |X0_error|={comp['x0_error']:.4f}")
            # Stop IMMEDIATELY if error is increasing
            if comp['abs_x0_error'] > prev_error:
                logger.debug(f"[Adduct {adduct_name}] <- Early stop: error increasing")
                break
            prev_error = comp['abs_x0_error']
            if comp['abs_x0_error'] < best_error:
                best_error = comp['abs_x0_error']

        # Search right (increasing Qcl, decreasing N0)
        prev_error = center_comp['abs_x0_error']  # Reset for right search
        for qcl in range(best_qcl + 1, num_ag + 1):
            comp = calculate_x0_for_composition(qcl)
            if comp is None:
                break
            compositions.append(comp)
            logger.debug(f"[Adduct {adduct_name}] -> N0={comp['n0']}, |X0_error|={comp['x0_error']:.4f}")
            # Stop IMMEDIATELY if error is increasing
            if comp['abs_x0_error'] > prev_error:
                logger.debug(f"[Adduct {adduct_name}] -> Early stop: error increasing")
                break
            prev_error = comp['abs_x0_error']
            if comp['abs_x0_error'] < best_error:
                best_error = comp['abs_x0_error']

        logger.debug(f"[Adduct {adduct_name}] Optimized search: found {len(compositions)} compositions, best error={best_error:.4f}")

        return compositions

    def calculate_dna_silver_composition_with_adduct(self, mz: float, z_observed: int, dna_sequence: str,
                                                      adduct_name: str, adduct_mass: float, adduct_charge: int,
                                                      detected_centroid: Optional[float] = None, resolution: int = 20000,
                                                      mz_values: Optional[npt.NDArray[np.float64]] = None,
                                                      intensity_values: Optional[npt.NDArray[np.float64]] = None,
                                                      nAg_center: Optional[int] = None, nAg_range: int = 3,
                                                      num_strands: int = 1, custom_xna: Optional[dict] = None,
                                                      strand_label: Optional[str] = None,
                                                      conjugate_name: Optional[str] = None, conjugate_count: int = 0) -> list[dict]:
        """
        Calculate DNA-silver compositions WITH a specific adduct.

        This is a specialized version that forces a specific adduct to be used.
        Formula: mass = (DNA + Ag + adduct_mass) - (Qcl + z + adduct_charge) * mH

        Note: N₀ + Qcl = nAg always (valence electron balance, unchanged by adducts)
        Adduct charge affects protons_removed formula: protons_removed = Qcl + z + adduct_charge

        Args:
            mz: Peak m/z value
            z_observed: Charge state (observed in mass spec)
            dna_sequence: DNA sequence string
            adduct_name: Name of adduct (e.g., 'NH4', '2Cl', '2Na')
            adduct_mass: Mass of adduct in Da
            adduct_charge: Charge of adduct (e.g., +1 for NH4+, -1 for Cl-)
            detected_centroid: Experimental X0 centroid
            resolution: MS resolution
            mz_values: Spectrum m/z array
            intensity_values: Spectrum intensity array
            nAg_center: Center nAg value from baseline (if None, search all 8-30)
            nAg_range: Range around center to search (default ±3)
            num_strands: Number of DNA strands (from baseline composition)

        Returns:
            List of composition dictionaries with adduct information
        """
        compositions = []

        # Get DNA/XNA composition for the specified number of strands
        if custom_xna and custom_xna.get('formula'):
            # Parse XNA formula to get element-level composition (for isotope patterns)
            try:
                xna_composition = composition_from_formula(custom_xna['formula'])
                nH = xna_composition.get('H', 0) * num_strands
                nC = xna_composition.get('C', 0) * num_strands
                nN = xna_composition.get('N', 0) * num_strands
                nO = xna_composition.get('O', 0) * num_strands
                nP = xna_composition.get('P', 0) * num_strands

                mH_total = self.m_p * nH
                mC_total = self.mC * nC
                mN_total = self.mN * nN
                mO_total = self.mO * nO
                mP_total = self.mP * nP

                # Use user-provided molecular weight if available
                mXNA_one = custom_xna.get('molecular_weight')
                if mXNA_one is None:
                    mXNA_one = self.calculate_mass_from_formula(custom_xna['formula'])
                mDNA_total = mXNA_one * num_strands
                extra_conj_mass = 0.0
                extra_conj_formula = ""
            except Exception as e:
                logger.error(f"Error parsing XNA formula '{custom_xna['formula']}': {e}")
                # Use user-provided molecular weight if available
                mXNA_one = custom_xna.get('molecular_weight')
                if mXNA_one is None:
                    mXNA_one = self.calculate_mass_from_formula(custom_xna['formula'])
                mDNA_total = mXNA_one * num_strands
                nH = nC = nN = nO = nP = 0
                mH_total = mC_total = mN_total = mO_total = mP_total = 0
                extra_conj_mass = 0.0
                extra_conj_formula = ""
        else:
            # Calculate standard DNA composition
            nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, strands=num_strands)

            # Add conjugate atoms if present
            extra_conj_mass = 0.0
            extra_conj_formula = ""
            if conjugate_name and conjugate_count > 0:
                for adduct in self.custom_adducts:
                    if adduct['name'] == conjugate_name:
                        conj_atoms = self.parse_formula_to_atoms(adduct.get('formula'))
                        if conj_atoms:
                            total_conjugates = conjugate_count  # Total conjugates (not per-strand)
                            nH += conj_atoms.get('H', 0) * total_conjugates
                            nC += conj_atoms.get('C', 0) * total_conjugates
                            nN += conj_atoms.get('N', 0) * total_conjugates
                            nO += conj_atoms.get('O', 0) * total_conjugates
                            nP += conj_atoms.get('P', 0) * total_conjugates
                            extra_conj_mass, extra_conj_formula = self._get_extra_conjugate_contribution(conj_atoms, total_conjugates)
                        break

            # Calculate masses for each element
            mH_total = self.m_p * nH
            mC_total = self.mC * nC
            mN_total = self.mN * nN
            mO_total = self.mO * nO
            mP_total = self.mP * nP
            mDNA_total = mH_total + mC_total + mN_total + mO_total + mP_total + extra_conj_mass

        # Determine search range for nAg
        if nAg_center is not None:
            # Search around specified center ±range
            # For low-nAg searches (center < 6), allow nAg_min to be 0
            # For regular cluster searches (center >= 8), enforce nAg_min >= 8
            if nAg_center < 6:
                nAg_min = max(0, nAg_center - nAg_range)
            else:
                nAg_min = max(8, nAg_center - nAg_range)
            nAg_max = min(MAX_SILVER + 1, nAg_center + nAg_range + 1)
            logger.debug(f"Searching nAg range: {nAg_min}-{nAg_max-1} (center={nAg_center} ±{nAg_range})")
        else:
            # Search full range (default for clusters)
            nAg_min = 8
            nAg_max = MAX_SILVER + 1

        # Get strand_label for complex mode detection
        effective_strand_label = strand_label
        if effective_strand_label is None and custom_xna:
            effective_strand_label = custom_xna.get('strand_label', None)
        # Complex labels are "nd=1", "nd=2", "nd=3" (or legacy "complex")
        # Also check custom_xna.is_complex flag for DNA-only Complex mode
        is_complex_label = effective_strand_label and (effective_strand_label.startswith('nd=') or effective_strand_label == 'complex')
        is_complex_flag = custom_xna and custom_xna.get('is_complex', False)
        is_complex = is_complex_label or is_complex_flag

        # Try different numbers of silver atoms with SAME efficiency as baseline
        for num_ag in range(nAg_min, nAg_max):
            mAg_total = self.mAg * num_ag

            # SAME APPROACH AS BASELINE: Quick check if this nAg is promising
            # Find the Qcl that gives m/z closest to target (with adduct)
            min_mz_error = float('inf')
            best_qcl_for_check = None
            best_mz_for_check = None

            # COMPLEX MODE: For complex, N0 = 0 always, so only test Qcl = nAg
            if is_complex:
                qcl_range: list[int] = [num_ag]  # Only Qcl = nAg
            else:
                qcl_range = list(range(0, num_ag + 1))

            for qcl_test in qcl_range:
                # Calculate mass WITH adduct using mDNA_total (correct for XNA!)
                # protons_removed = Qcl + z + adduct_charge
                neutral_mass = mDNA_total + mAg_total + adduct_mass
                mass_test = neutral_mass - (qcl_test + z_observed + adduct_charge) * self.m_p
                mz_test = mass_test / z_observed
                mz_error = abs(mz_test - detected_centroid)

                if mz_error < min_mz_error:
                    min_mz_error = mz_error
                    best_qcl_for_check = qcl_test
                    best_mz_for_check = mz_test

            # Only proceed if this nAg+adduct formula is close (within 10 m/z)
            if min_mz_error < 10.0:
                logger.debug(f"Testing nAg={num_ag} (error={min_mz_error:.2f} < 10.0)")
                # This nAg is promising! Use smart_n0_search with adduct
                # (effective_strand_label already determined above for complex detection)
                smart_comps = self.smart_n0_search_with_adduct(
                    num_strands, num_ag, z_observed, dna_sequence, detected_centroid,
                    nH, nC, nN, nO, nP, mH_total, mC_total, mN_total, mO_total, mP_total, mAg_total,
                    adduct_name, adduct_mass, adduct_charge, resolution, custom_xna=custom_xna,
                    strand_label=effective_strand_label,
                    conjugate_name=conjugate_name, conjugate_count=conjugate_count,
                    extra_conj_mass=extra_conj_mass, extra_conj_formula=extra_conj_formula
                )
                logger.debug(f"nAg={num_ag} returned {len(smart_comps)} compositions")
                compositions.extend(smart_comps)
            else:
                logger.debug(f"SKIP nAg={num_ag} (error={min_mz_error:.2f} >= 10.0)")

        # Return compositions (already have X0 calculated from smart search)
        return compositions

    def analyze_peak_with_smart_adduct_search(self, peak_mz: float, charge: int, dna_sequence: str,
                                               exp_x0: float, resolution: int = 20000,
                                               mz_values: Optional[npt.NDArray[np.float64]] = None,
                                               intensity_values: Optional[npt.NDArray[np.float64]] = None,
                                               custom_xna: Optional[dict] = None,
                                               conjugate_name: Optional[str] = None,
                                               conjugate_count: int = 0,
                                               **kwargs) -> list[dict]:
        """
        Smart adduct search that explores (nAg±3, adduct) combinations.

        Strategy:
        1. Analyze without adduct first (baseline) - get best nAg value
        2. If X0 error > 0.5, test common adducts around baseline nAg ±3
        3. Each adduct searches nAg range: [baseline_nAg-3, baseline_nAg+3]
        4. Keep no-adduct as candidate (may still be best!)
        5. Return best match overall from all tested combinations

        Args:
            peak_mz: Peak m/z value
            charge: Charge state
            dna_sequence: DNA sequence string
            exp_x0: Experimental X0 centroid
            resolution: MS resolution
            mz_values: Spectrum m/z array (for isotope matching)
            intensity_values: Spectrum intensity array (for isotope matching)
            conjugate_name: Name of conjugate (e.g., 'BCN') attached to DNA before silver binding
            conjugate_count: Number of conjugate molecules attached per strand (0 = no conjugate)

        Returns:
            List of best compositions (may or may not have adduct)
        """
        # Version marker for debugging (change this to force new output)
        VERSION = "v2.2-2025-02-05-conjugate"
        logger.info(f"SMART ADDUCT SEARCH {VERSION}")
        logger.info(f"Smart Adduct Search for m/z {peak_mz:.4f} (z={charge})")

        # Check for prioritized conjugate (charge 0 custom adduct marked as prioritized)
        conjugate_counts_to_try = [0]  # Always try without conjugate
        if conjugate_name is None or conjugate_count == 0:
            conjugate = self.get_prioritized_conjugate()
            if conjugate:
                conjugate_name = conjugate['name']
                conjugate_count = 2
                conjugate_counts_to_try = [0, 'all']  # 'all' resolves to 1..num_strands in loops
                logger.info(f"CONJUGATE DETECTED: {conjugate_name}, will try counts {conjugate_counts_to_try}")
            else:
                conjugate_name = None
                conjugate_count = 0
        else:
            conjugate_counts_to_try = [conjugate_count]  # Use explicitly provided count

        logger.debug(f"Custom adducts list: {self.custom_adducts}")
        logger.debug(f"Custom adduct names: {[a['name'] for a in self.custom_adducts]}")
        logger.debug(f"All adducts dict keys: {sorted(list(self.adducts.keys()))}")

        # STEP 1: Baseline analysis (no adduct) - try each conjugate count
        logger.info("Step 1: Analyzing without adduct (baseline)...")
        compositions_no_adduct = []
        # Resolve 'all' marker: try all-conjugated (num_strands) first
        # Only try mixed conjugation (1..num_strands-1) if all-conjugated baseline has high X0 error
        resolved_counts = []
        for conj_count in conjugate_counts_to_try:
            if conj_count == 'all':
                resolved_counts.append(2)  # All strands conjugated first
            else:
                resolved_counts.append(conj_count)
        for conj_count in resolved_counts:
            conj_name_try = conjugate_name if conj_count > 0 else None
            comps = self.calculate_dna_silver_composition(
                peak_mz, charge, dna_sequence, detected_centroid=exp_x0, resolution=resolution,
                custom_xna=custom_xna, conjugate_name=conj_name_try, conjugate_count=conj_count
            )
            if comps:
                logger.info(f"Baseline with {conj_count}x {conjugate_name or 'none'}: {len(comps)} compositions")
                compositions_no_adduct.extend(comps)
        logger.debug(f"Baseline total: {len(compositions_no_adduct)} compositions")

        # If conjugate is present and baseline X0 is still high, also try mixed conjugation
        if conjugate_name and len(compositions_no_adduct) > 0:
            best_baseline = min(compositions_no_adduct, key=lambda c: abs(c.get('x0_error', 999.0)))
            baseline_x0 = abs(best_baseline.get('x0_error', 999.0))
            if baseline_x0 > 0.5:
                logger.info(f"All-conjugated baseline X0={baseline_x0:.4f} > 0.5: also trying mixed conjugation")
                for mixed_count in range(1, 2):  # Try 1 conjugate on 2 strands
                    comps_mixed = self.calculate_dna_silver_composition(
                        peak_mz, charge, dna_sequence, detected_centroid=exp_x0, resolution=resolution,
                        custom_xna=custom_xna, conjugate_name=conjugate_name, conjugate_count=mixed_count
                    )
                    if comps_mixed:
                        logger.info(f"Mixed conjugation {mixed_count}x {conjugate_name}: {len(comps_mixed)} compositions")
                        compositions_no_adduct.extend(comps_mixed)

        if not compositions_no_adduct:
            logger.warning("No baseline compositions found within threshold")

            # COMPLEX MODE: Try direct no-adduct search first before adduct fallback
            is_complex_fallback = custom_xna and custom_xna.get('is_complex', False)
            if is_complex_fallback:
                logger.info("COMPLEX FALLBACK: Trying direct no-adduct search first...")
                # For Complex DNA mode, try nd=1, nd=2, nd=3 with direct nAg calculation
                complex_no_adduct_candidates = []
                for num_complexes in range(1, MAX_COMPLEXES + 1):
                    num_strands_total = num_complexes * 2
                    strand_label = f'nd={num_complexes}'

                    # Calculate DNA mass for this complex configuration
                    nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, strands=num_strands_total)

                    # Add conjugate atoms if present
                    extra_conj_mass_d = 0.0
                    extra_conj_formula_d = ""
                    if conjugate_name and conjugate_count > 0:
                        for adduct in self.custom_adducts:
                            if adduct['name'] == conjugate_name:
                                conj_atoms = self.parse_formula_to_atoms(adduct.get('formula'))
                                if conj_atoms:
                                    total_conjugates = conjugate_count  # Total conjugates (not per-strand)_total
                                    nH += conj_atoms.get('H', 0) * total_conjugates
                                    nC += conj_atoms.get('C', 0) * total_conjugates
                                    nN += conj_atoms.get('N', 0) * total_conjugates
                                    nO += conj_atoms.get('O', 0) * total_conjugates
                                    nP += conj_atoms.get('P', 0) * total_conjugates
                                    extra_conj_mass_d, extra_conj_formula_d = self._get_extra_conjugate_contribution(conj_atoms, total_conjugates)
                                break

                    base_mass = self.m_p * nH + self.mC * nC + self.mN * nN + self.mO * nO + self.mP * nP + extra_conj_mass_d

                    # Calculate optimal nAg directly: nAg = (observed_mass - DNA_mass) / (Ag_mass - H_mass)
                    # For Complex: Qcl = nAg, so mass = DNA + nAg*Ag - (nAg + z)*H
                    nAg_float = (exp_x0 * charge - base_mass + charge * self.m_p) / (self.mAg - self.m_p)
                    optimal_nAg = int(round(nAg_float))
                    optimal_nAg = max(0, min(30, optimal_nAg))

                    logger.info(f"COMPLEX FALLBACK: {strand_label}, DNA_mass={base_mass:.4f}, exp_x0={exp_x0}, z={charge}, nAg_float={nAg_float:.4f}, optimal_nAg={optimal_nAg}")

                    # Test nAg ± 5 range (expanded for better coverage)
                    for test_nAg in range(max(0, optimal_nAg - 5), min(MAX_SILVER + 1, optimal_nAg + 6)):
                        mAg_total = self.mAg * test_nAg
                        comps = self.smart_n0_search(
                            num_strands_total, test_nAg, charge, dna_sequence, exp_x0,
                            nH, nC, nN, nO, nP,
                            self.m_p * nH, self.mC * nC, self.mN * nN, self.mO * nO, self.mP * nP, mAg_total,
                            resolution, custom_xna=custom_xna, strand_label=strand_label,
                            conjugate_name=conjugate_name, conjugate_count=conjugate_count,
                            extra_conj_mass=extra_conj_mass_d, extra_conj_formula=extra_conj_formula_d
                        )
                        if comps:
                            x0_err = comps[0].get('abs_x0_error', 999)
                            logger.info(f"COMPLEX FALLBACK: {strand_label} nAg={test_nAg}, X₀ error={x0_err:.4f}")
                        complex_no_adduct_candidates.extend(comps)

                if complex_no_adduct_candidates:
                    # Refine with pattern matching
                    if mz_values is not None and intensity_values is not None:
                        refined, _, _, _, _, _, _, _, _ = self.refine_compositions_with_isotope_matching(
                            complex_no_adduct_candidates, mz_values, intensity_values, peak_mz,
                            resolution=resolution, detected_centroid=exp_x0
                        )
                        complex_no_adduct_candidates = refined if refined else complex_no_adduct_candidates

                    # Sort by X₀ error
                    complex_no_adduct_candidates.sort(key=lambda x: x.get('abs_x0_error', 999.0))
                    best_complex = complex_no_adduct_candidates[0]
                    logger.info(f"COMPLEX FALLBACK: Found no-adduct composition with X₀ error {best_complex.get('abs_x0_error', 999):.4f}")

                    # Return no-adduct results for Complex mode
                    return complex_no_adduct_candidates
                else:
                    logger.warning("COMPLEX FALLBACK: No no-adduct compositions found, will try adducts")

            logger.info("Fallback: Smart adduct search (no baseline available)...")

            # When baseline fails, test strands 1-3
            # For each strand, find the absolute best adduct/nAg combination
            all_adduct_candidates = []
            # Include base adducts (single AND double), plus custom adducts
            # Must match post-baseline adduct list to ensure consistent results
            common_adducts = ['NH4', '2NH4', 'Na', '2Na', 'Cl', '2Cl'] + [a['name'] for a in self.custom_adducts]

            # Track ALL promising adducts (not just the best one)
            promising_adducts = []  # List of (error, strands, nAg, adduct_name)

            # Build strand configurations based on mode
            # For complex mode: test nd=1, nd=2, nd=3 (1, 2, 3 complexes = 2, 4, 6 strands)
            # For regular XNA/DNA: test 1, 2, 3 strands
            if custom_xna and custom_xna.get('is_complex', False) and custom_xna.get('formula'):
                # COMPLEX XNA MODE: Test multiple complexes (nd=1, 2, 3) with XNA formula
                strand_configs = []
                complex_mass = self.calculate_mass_from_formula(custom_xna['formula'])
                # Test 1, 2, 3 complexes (2, 4, 6 strands)
                for num_complexes in range(1, MAX_COMPLEXES + 1):
                    num_strands_total = num_complexes * 2
                    total_mass = complex_mass * num_complexes
                    strand_configs.append((f'nd={num_complexes}', num_strands_total, total_mass, custom_xna['formula']))

                logger.info(f"COMPLEX XNA FALLBACK: Testing {len(strand_configs)} configurations: {[c[0] for c in strand_configs]}")
            elif custom_xna and custom_xna.get('is_complex', False):
                # COMPLEX DNA MODE: Test multiple complexes using DNA sequence (no XNA formula)
                strand_configs = []
                for num_complexes in range(1, MAX_COMPLEXES + 1):
                    num_strands_total = num_complexes * 2
                    nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, strands=num_strands_total)
                    extra_conj_mass_fb = 0.0
                    if conjugate_name and conjugate_count > 0:
                        for adduct in self.custom_adducts:
                            if adduct['name'] == conjugate_name:
                                conj_atoms = self.parse_formula_to_atoms(adduct.get('formula'))
                                if conj_atoms:
                                    total_conj = conjugate_count  # Total conjugates (not per-strand)
                                    nH += conj_atoms.get('H', 0) * total_conj
                                    nC += conj_atoms.get('C', 0) * total_conj
                                    nN += conj_atoms.get('N', 0) * total_conj
                                    nO += conj_atoms.get('O', 0) * total_conj
                                    nP += conj_atoms.get('P', 0) * total_conj
                                    extra_conj_mass_fb, _ = self._get_extra_conjugate_contribution(conj_atoms, total_conj)
                                break
                    base_mass = self.m_p * nH + self.mC * nC + self.mN * nN + self.mO * nO + self.mP * nP + extra_conj_mass_fb
                    strand_configs.append((f'nd={num_complexes}', num_strands_total, base_mass, None))
                logger.info(f"COMPLEX DNA FALLBACK: Testing {len(strand_configs)} configurations: {[c[0] for c in strand_configs]}")
            elif custom_xna and custom_xna.get('formula'):
                # Regular XNA mode: 1, 2, 3 strands with same formula
                mXNA_one = custom_xna.get('molecular_weight')
                if mXNA_one is None:
                    mXNA_one = self.calculate_mass_from_formula(custom_xna['formula'])
                strand_configs = [(f'{i}strand', i, mXNA_one * i, custom_xna['formula']) for i in [1, 2, 3]]
            else:
                # DNA mode: calculate mass from sequence
                strand_configs = []
                for i in [1, 2, 3]:
                    nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, strands=i)
                    extra_conj_mass_fb = 0.0
                    if conjugate_name and conjugate_count > 0:
                        for adduct in self.custom_adducts:
                            if adduct['name'] == conjugate_name:
                                conj_atoms = self.parse_formula_to_atoms(adduct.get('formula'))
                                if conj_atoms:
                                    total_conj = conjugate_count  # Total conjugates (not per-strand)
                                    nH += conj_atoms.get('H', 0) * total_conj
                                    nC += conj_atoms.get('C', 0) * total_conj
                                    nN += conj_atoms.get('N', 0) * total_conj
                                    nO += conj_atoms.get('O', 0) * total_conj
                                    nP += conj_atoms.get('P', 0) * total_conj
                                    extra_conj_mass_fb, _ = self._get_extra_conjugate_contribution(conj_atoms, total_conj)
                                break
                    base_mass = self.m_p * nH + self.mC * nC + self.mN * nN + self.mO * nO + self.mP * nP + extra_conj_mass_fb
                    strand_configs.append((f'{i}strand', i, base_mass, None))

            for strand_label, test_strands, base_mass, strand_formula in strand_configs:
                logger.debug(f"Testing {strand_label} (strands={test_strands}, mass={base_mass:.2f})...")

                for adduct_name in common_adducts:
                    if adduct_name not in self.adducts:
                        continue

                    adduct_mass, adduct_charge = self.adducts[adduct_name]

                    # OPTIMIZED: Use direct algebra to find optimal nAg, then test ±2
                    # est_mass = base_mass + nAg * mAg + adduct_mass - (nAg + z + adduct_charge) * mH
                    # est_mz = est_mass / z = peak_mz (target)
                    # Solve: nAg = (peak_mz * z - base_mass - adduct_mass + (z + adduct_charge) * mH) / (mAg - mH)
                    # Note: protons_removed = Qcl + z + adduct_charge (where Qcl = nAg for DNA-only)
                    nAg_float = (peak_mz * charge - base_mass - adduct_mass + (charge + adduct_charge) * self.m_p) / (self.mAg - self.m_p)
                    nAg_center = int(round(nAg_float))
                    nAg_center = max(0, min(30, nAg_center))  # Clamp to valid range

                    # Test 5 nAg values: center-2, center-1, center, center+1, center+2
                    best_nAg = nAg_center
                    best_error = float('inf')
                    for test_nAg in range(nAg_center - 2, nAg_center + 3):  # -2 to +2
                        if test_nAg < 0 or test_nAg > 30:
                            continue
                        est_mass = base_mass + test_nAg * self.mAg + adduct_mass - (test_nAg + charge + adduct_charge) * self.m_p
                        est_mz = est_mass / charge
                        mz_error = abs(est_mz - peak_mz)
                        if mz_error < best_error:
                            best_error = mz_error
                            best_nAg = test_nAg

                    logger.debug(f"Adduct={adduct_name}: best nAg={best_nAg} (from ±2 of {nAg_center}), error={best_error:.4f} m/z")

                    # Track ALL promising adducts within threshold
                    # Include strand_label and strand_formula for complex mode
                    if best_error < 10.0:  # Within 10 m/z threshold
                        promising_adducts.append((best_error, test_strands, best_nAg, adduct_name, strand_label, strand_formula))

            # ADDUCT MASS VALIDATION: Filter and rank by remaining mass match
            # Simple: remaining = observed_mass - DNA_mass - nAg × Ag_mass
            # Valid adducts: remaining >= expected_adduct_mass (can't have negative contribution)
            # Rank by |remaining - expected| (smallest difference = best match)
            if promising_adducts:
                # Build lookup for DNA mass by strand configuration
                strand_mass_lookup = {label: mass for label, _, mass, _ in strand_configs}

                validated_adducts = []
                for mz_error, test_strands, best_nAg, adduct_name, strand_label, strand_formula in promising_adducts:
                    adduct_mass, adduct_charge = self.adducts[adduct_name]
                    dna_mass = strand_mass_lookup.get(strand_label, 0.0)

                    # Simple calculation: observed_mass - DNA - Ag = remaining
                    # This should approximately equal the adduct mass
                    observed_mass = peak_mz * charge
                    remaining = observed_mass - dna_mass - best_nAg * self.mAg

                    # Validation: remaining must be >= expected (with 5% tolerance)
                    MIN_RATIO = 0.95
                    is_valid = remaining >= adduct_mass * MIN_RATIO

                    mass_match_error = abs(remaining - adduct_mass)

                    logger.debug(f"Adduct mass check: {adduct_name}, remaining={remaining:.2f}, expected={adduct_mass:.2f}, valid={is_valid}")

                    if is_valid:
                        validated_adducts.append((mass_match_error, mz_error, test_strands, best_nAg, adduct_name, strand_label, strand_formula))
                    else:
                        logger.debug(f"  REJECTED: {adduct_name} (remaining {remaining:.2f} < expected {adduct_mass:.2f})")

                # If we have validated adducts, use them; otherwise fall back to all promising
                if validated_adducts:
                    # Don't sort by mass error - let X0 error decide among valid adducts
                    # Just pass all valid adducts through for isotope pattern testing
                    adducts_to_test = [(mz_err, strands, nAg, adduct, s_label, s_formula)
                                       for mass_err, mz_err, strands, nAg, adduct, s_label, s_formula in validated_adducts]
                    logger.info(f"Testing {len(adducts_to_test)} adducts (validated by mass match, will rank by X0):")
                else:
                    # Fallback: use old method if validation filters out everything
                    promising_adducts.sort(key=lambda x: x[0])
                    best_error = promising_adducts[0][0]
                    adducts_to_test = [a for a in promising_adducts if a[0] <= best_error + 2.0]
                    logger.warning(f"No adducts passed mass validation, falling back to {len(adducts_to_test)} by m/z error")

                # OPTIMIZATION: Limit to top 5 adducts to speed up XNA analysis
                # Sort by m/z error to prioritize most promising adducts
                adducts_to_test.sort(key=lambda x: x[0])  # Sort by m/z error
                MAX_ADDUCTS_TO_TEST = 5
                adducts_to_test = adducts_to_test[:MAX_ADDUCTS_TO_TEST]

                for err, strands, nAg, adduct, s_label, s_formula in adducts_to_test:
                    logger.debug(f"{adduct}: {s_label} (strands={strands}), nAg={nAg}, error={err:.2f} m/z")

                for err, test_strands, test_nAg, adduct_name, strand_label, strand_formula in adducts_to_test:
                    logger.debug(f"Testing isotope pattern for {adduct_name} ({strand_label}, strands={test_strands}, nAg={test_nAg})...")
                    adduct_mass, adduct_charge = self.adducts[adduct_name]

                    # For complex mode, create a temporary custom_xna with the specific strand's formula
                    # This ensures calculate_dna_silver_composition_with_adduct uses the correct formula
                    temp_custom_xna: dict[str, Any] | None = None
                    if custom_xna and custom_xna.get('is_complex', False) and strand_formula:
                        # Create modified custom_xna for this specific strand configuration
                        temp_custom_xna = {
                            'formula': strand_formula,
                            'molecular_weight': self.calculate_mass_from_formula(strand_formula),
                            # Preserve other fields but mark as non-complex for the adduct function
                            # (it doesn't need complex logic since we're providing the specific formula)
                            'is_complex': False,
                            'strand_label': strand_label  # For debugging
                        }
                    else:
                        temp_custom_xna = custom_xna

                    comps = self.calculate_dna_silver_composition_with_adduct(
                        peak_mz, charge, dna_sequence,
                        adduct_name, adduct_mass, adduct_charge,
                        detected_centroid=exp_x0, resolution=resolution,
                        mz_values=mz_values, intensity_values=intensity_values,
                        nAg_center=test_nAg, nAg_range=1,  # Use ±1 range
                        num_strands=test_strands,
                        custom_xna=temp_custom_xna,
                        strand_label=strand_label,  # Pass strand_label for complex mode detection
                        conjugate_name=conjugate_name, conjugate_count=conjugate_count
                    )
                    if comps:
                        # Add strand_label to each composition for clarity
                        for comp in comps:
                            if strand_label:
                                comp['strand_config'] = strand_label
                        all_adduct_candidates.extend(comps)
                        logger.info(f"Found {len(comps)} candidates for {adduct_name} ({strand_label})")

                        # EARLY TERMINATION: If we found an excellent match, stop searching
                        best_comp = min(comps, key=lambda x: x.get('abs_x0_error', 999.0))
                        if best_comp.get('abs_x0_error', 999.0) < 0.15:
                            logger.info(f"Early termination: excellent match found (X0 error={best_comp['abs_x0_error']:.4f})")
                            break

            if all_adduct_candidates:
                if mz_values is not None and intensity_values is not None:
                    # Refine with pattern matching
                    logger.info("Refining fallback compositions with pattern matching...")
                    refined_fallback, _, _, _, _, _, _, _, _ = self.refine_compositions_with_isotope_matching(
                        all_adduct_candidates,
                        mz_values, intensity_values,
                        peak_mz,  # Add missing peak_mz argument
                        resolution=resolution,
                        detected_centroid=exp_x0
                    )
                    all_adduct_candidates = refined_fallback if refined_fallback else all_adduct_candidates

                # Sort by pattern score (primary), X₀ error (tiebreaker) for adduct-only candidates
                all_adduct_candidates.sort(key=lambda x: (
                    -x.get('pattern_score', 0.0),
                    x.get('abs_x0_error', 999.0)
                ))
                logger.info(f"Found {len(all_adduct_candidates)} adduct candidates (no baseline needed)")
                logger.info(f"Best: {all_adduct_candidates[0]['formula']}, pattern={all_adduct_candidates[0].get('pattern_score', 0.0):.2f}, X0 error={all_adduct_candidates[0].get('abs_x0_error', 999.0):.4f}")
                return all_adduct_candidates
            else:
                logger.warning("No adduct candidates found either")
                dimer_result = self._try_dimer_fallback(peak_mz, charge, dna_sequence, exp_x0,
                    resolution, mz_values, intensity_values, custom_xna, conjugate_name, conjugate_count, kwargs)
                return dimer_result if dimer_result else []

        # REFINE BASELINE WITH PATTERN MATCHING for fair comparison with fallback
        if mz_values is not None and intensity_values is not None:
            logger.info("Refining BASELINE compositions with pattern matching...")
            refined_baseline, _, _, _, _, _, _, _, _ = self.refine_compositions_with_isotope_matching(
                compositions_no_adduct,
                mz_values, intensity_values,
                peak_mz,  # Add missing peak_mz argument
                resolution=resolution,
                detected_centroid=exp_x0
            )
            compositions_no_adduct = refined_baseline if refined_baseline else compositions_no_adduct

        # OPTIMIZATION: Sort by smallest X0 error to find the best baseline match
        # This ensures we pick the composition closest to experimental data,
        # reducing unnecessary adduct searches
        compositions_no_adduct.sort(key=lambda x: x.get('abs_x0_error', 999.0))
        logger.debug(f"Sorted baseline by X0 error. Top 3: {[(c['formula'], c.get('abs_x0_error', 999)) for c in compositions_no_adduct[:3]]}")

        best_no_adduct = compositions_no_adduct[0]
        baseline_error = best_no_adduct['abs_x0_error']
        baseline_pattern_score = best_no_adduct.get('pattern_score', 0.0)
        baseline_nAg = best_no_adduct['num_silver']  # Get nAg from best baseline composition
        baseline_strands = best_no_adduct['num_strands']  # Get strands from best baseline composition

        logger.info(f"Baseline result: {best_no_adduct['formula']}")
        logger.info(f"Baseline X0 error: {baseline_error:.4f} m/z")
        logger.info(f"Baseline pattern score: {baseline_pattern_score:.2f}")
        logger.info(f"Baseline: {baseline_strands} strands, {baseline_nAg} Ag")

        # COMPLEX MODE: No-adduct is PRIMARY, adduct is FALLBACK only
        # For Complex mode, return no-adduct results unless X₀ error is very high (> 5.0)
        is_complex_mode = custom_xna and custom_xna.get('is_complex', False)
        if is_complex_mode:
            COMPLEX_ADDUCT_FALLBACK_THRESHOLD = 5.0  # Only search adducts if X₀ error > 5.0 m/z
            if baseline_error <= COMPLEX_ADDUCT_FALLBACK_THRESHOLD:
                logger.info(f"COMPLEX MODE: Returning no-adduct result (X₀ error {baseline_error:.4f} <= {COMPLEX_ADDUCT_FALLBACK_THRESHOLD})")
                logger.info(f"COMPLEX MODE: Skipping adduct search (no-adduct is primary for Complex)")
                return compositions_no_adduct
            else:
                logger.info(f"COMPLEX MODE: X₀ error {baseline_error:.4f} > {COMPLEX_ADDUCT_FALLBACK_THRESHOLD}, will search adducts as fallback")

        # For complex mode, get the strand configuration from baseline
        baseline_strand_type = best_no_adduct.get('strand_type', None)  # 'strand1', 'strand2', 'complex', or None
        if baseline_strand_type:
            logger.debug(f"Baseline strand type: {baseline_strand_type}")

        # Prepare custom_xna for adduct testing based on baseline strand type
        # For complex mode, we need to use the specific strand's formula, not the combined formula
        adduct_custom_xna = custom_xna  # Default: use original custom_xna
        if custom_xna and custom_xna.get('is_complex', False) and custom_xna.get('formula') and baseline_strand_type:
            strand1_formula = custom_xna.get('strand1_formula', '')
            strand2_formula = custom_xna.get('strand2_formula', '')
            same_strands = custom_xna.get('same_strands', False)

            if baseline_strand_type == 'strand1' and strand1_formula:
                adduct_custom_xna = {
                    'formula': strand1_formula,
                    'molecular_weight': self.calculate_mass_from_formula(strand1_formula),
                    'is_complex': False,  # Treat as simple XNA for adduct function
                    'strand_label': 'strand1',
                    'name': custom_xna.get('name', 'XNA') + '-strand1'
                }
                logger.debug(f"COMPLEX: Using strand1 formula for adduct testing: {strand1_formula}")
            elif baseline_strand_type == 'strand2' and strand2_formula and not same_strands:
                adduct_custom_xna = {
                    'formula': strand2_formula,
                    'molecular_weight': self.calculate_mass_from_formula(strand2_formula),
                    'is_complex': False,
                    'strand_label': 'strand2',
                    'name': custom_xna.get('name', 'XNA') + '-strand2'
                }
                logger.debug(f"COMPLEX: Using strand2 formula for adduct testing: {strand2_formula}")
            elif baseline_strand_type == 'complex':
                # For complex, use the combined formula (already in custom_xna)
                logger.debug(f"COMPLEX: Using combined complex formula for adduct testing: {custom_xna['formula']}")
                # Keep adduct_custom_xna = custom_xna but mark as non-complex to avoid recursion
                adduct_custom_xna = {
                    'formula': custom_xna['formula'],
                    'molecular_weight': self.calculate_mass_from_formula(custom_xna['formula']),
                    'is_complex': False,
                    'strand_label': 'complex',
                    'name': custom_xna.get('name', 'XNA') + '-complex'
                }
        elif custom_xna and custom_xna.get('is_complex', False) and not custom_xna.get('formula'):
            # DNA-only Complex mode - keep is_complex flag for N0=0 enforcement, but no formula
            adduct_custom_xna = {
                'name': 'Complex',
                'is_complex': True,
                'same_strands': custom_xna.get('same_strands', False)
            }
            logger.debug(f"COMPLEX DNA MODE: No XNA formula, using DNA sequence for adduct testing (is_complex=True)")
        # STEP 2: Only test adducts if baseline X0 error is high (> 0.5 m/z)
        # OPTIMIZATION: Skip adduct search when baseline is good enough
        X0_ERROR_THRESHOLD = 0.5  # m/z - only search adducts if baseline error exceeds this
        if baseline_error <= X0_ERROR_THRESHOLD:
            logger.info(f"SKIP adduct search: baseline X0 error ({baseline_error:.4f}) <= threshold ({X0_ERROR_THRESHOLD})")
            return compositions_no_adduct

        logger.info(f"STEP 2: Testing adducts (baseline X0 error={baseline_error:.4f} > {X0_ERROR_THRESHOLD}, nAg={baseline_nAg} ±1)...")

        # STEP 3: Test ALL adducts with nAg = baseline ±1
        logger.debug(f"Strategy: Test ALL adducts with nAg = {baseline_nAg} ±1")

        # Test most common adducts (including multiples) + custom adducts
        common_adducts = ['NH4', '2NH4', 'Na', '2Na', 'Cl', '2Cl']

        # Add custom adducts with their multiples (1x, 2x)
        custom_adduct_names = self.get_custom_adduct_names()
        common_adducts.extend(custom_adduct_names)

        logger.debug(f"Testing adducts: {common_adducts}")
        logger.debug(f"Custom adducts loaded: {[a['name'] for a in self.custom_adducts]}")
        if custom_adduct_names:
            logger.debug(f"Including {len(self.custom_adducts)} custom adducts with multiples: {', '.join([a['name'] for a in self.custom_adducts])}")

        logger.debug("About to calculate DNA/XNA composition...")

        # Get DNA/XNA composition for baseline strands
        logger.debug(f"custom_xna={custom_xna is not None}, baseline_strands={baseline_strands}")
        logger.debug(f"dna_sequence type={type(dna_sequence)}, value={dna_sequence[:50] if dna_sequence else 'None'}...")

        try:
            if adduct_custom_xna:
                # Parse XNA formula to get element-level composition
                # Use adduct_custom_xna which has the correct formula for complex strand type
                logger.debug(f"XNA mode - parsing formula: {adduct_custom_xna.get('formula', 'N/A')}")
                try:
                    xna_composition = composition_from_formula(adduct_custom_xna['formula'])
                    nH = xna_composition.get('H', 0) * baseline_strands
                    nC = xna_composition.get('C', 0) * baseline_strands
                    nN = xna_composition.get('N', 0) * baseline_strands
                    nO = xna_composition.get('O', 0) * baseline_strands
                    nP = xna_composition.get('P', 0) * baseline_strands

                    mH_total = self.m_p * nH
                    mC_total = self.mC * nC
                    mN_total = self.mN * nN
                    mO_total = self.mO * nO
                    mP_total = self.mP * nP

                    # Use user-provided molecular weight if available
                    mXNA_one = adduct_custom_xna.get('molecular_weight')
                    if mXNA_one is None:
                        mXNA_one = self.calculate_mass_from_formula(adduct_custom_xna['formula'])
                    mDNA_total = mXNA_one * baseline_strands
                except Exception as e:
                    logger.error(f"Error parsing XNA formula '{adduct_custom_xna['formula']}': {e}")
                    # Use user-provided molecular weight if available
                    mXNA_one = adduct_custom_xna.get('molecular_weight')
                    if mXNA_one is None:
                        mXNA_one = self.calculate_mass_from_formula(adduct_custom_xna['formula'])
                    mDNA_total = mXNA_one * baseline_strands
                    nH = nC = nN = nO = nP = 0
                    mH_total = mC_total = mN_total = mO_total = mP_total = 0
            else:
                # Get DNA composition for baseline strands
                logger.debug("DNA mode - calling calculate_dna_composition...")
                nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, strands=baseline_strands)
                logger.debug(f"DNA composition calculated: nH={nH}, nC={nC}, nN={nN}, nO={nO}, nP={nP}")
                mH_total = self.m_p * nH
                mC_total = self.mC * nC
                mN_total = self.mN * nN
                mO_total = self.mO * nO
                mP_total = self.mP * nP
                mDNA_total = mH_total + mC_total + mN_total + mO_total + mP_total
                logger.debug(f"mDNA_total calculated: {mDNA_total:.2f}")
        except Exception as e:
            logger.exception(f"Error in DNA/XNA composition calculation: {e}")
            # Return baseline result if composition calculation fails
            return best_no_adduct

        # Test ALL adducts (no pre-filtering)
        NAG_RANGE = 1  # Test nAg = baseline ± 1

        logger.debug(f"Testing ALL {len(common_adducts)} adducts (no pre-filter), baseline nAg={baseline_nAg}")

        # Build list of all adducts to test with their best nAg
        promising_adducts = []
        for adduct_name in common_adducts:
            if adduct_name not in self.adducts:
                logger.warning(f"{adduct_name} not in adduct library, skipping")
                continue
            # Test all adducts with baseline nAg (will search ±1 later)
            promising_adducts.append((adduct_name, baseline_nAg))
            logger.debug(f"{adduct_name}: will test nAg={baseline_nAg}±1")

        logger.debug(f"Testing {len(promising_adducts)} adducts: {[a[0] for a in promising_adducts]}")

        if not promising_adducts:
            logger.warning("No adducts available to test (adduct library empty?)")

            # If baseline error is VERY large (> 3.0), baseline is probably wrong
            # Run 2-PHASE fallback: fast m/z screening, then top-5 pattern generation
            FALLBACK_THRESHOLD = 3.0
            if baseline_error > FALLBACK_THRESHOLD:
                logger.info(f"Baseline error ({baseline_error:.2f}) > {FALLBACK_THRESHOLD}, trying 2-phase adduct fallback...")

                all_adduct_candidates = []
                # Extended adduct list including multi-adducts + custom adducts
                common_adducts_fallback = ['NH4', '2NH4', 'Na', '2Na', 'Cl', '2Cl'] + self.get_custom_adduct_names()

                # Test BOTH baseline_strands AND baseline_strands + 1
                # For complex mode, we use the correct formula from adduct_custom_xna
                strand_counts_to_test = [baseline_strands, baseline_strands + 1]

                # PHASE 1: Fast m/z screening (no isotope patterns!)
                logger.info("Phase 1: Fast m/z screening...")
                phase1_candidates = []  # List of (mz_error, strands, nAg, adduct_name)

                for test_strands in strand_counts_to_test:
                    # Get base mass once per strand count
                    # Use adduct_custom_xna which has the correct formula for complex strand type
                    if adduct_custom_xna:
                        mXNA_one = adduct_custom_xna.get('molecular_weight')
                        if mXNA_one is None:
                            mXNA_one = self.calculate_mass_from_formula(adduct_custom_xna['formula'])
                        base_mass = mXNA_one * test_strands
                    else:
                        nH, nC, nN, nO, nP = self.calculate_dna_composition(dna_sequence, strands=test_strands)
                        base_mass = self.m_p * nH + self.mC * nC + self.mN * nN + self.mO * nO + self.mP * nP
                        # Add conjugate mass if present (total semantics)
                        # Note: custom['mass'] already includes ALL atoms (H,C,N,O,P,S,etc.)
                        if conjugate_name and conjugate_count > 0:
                            for custom_fb in self.custom_adducts:
                                if custom_fb['name'] == conjugate_name:
                                    base_mass += custom_fb['mass'] * conjugate_count
                                    break

                    for adduct_name in common_adducts_fallback:
                        if adduct_name not in self.adducts:
                            continue

                        adduct_mass, adduct_charge = self.adducts[adduct_name]

                        # Use direct algebra to find approximate best nAg
                        # Note: This assumes Qcl ≈ nAg (rough estimate)
                        nAg_float = (peak_mz * charge - base_mass - adduct_mass + (charge + adduct_charge) * self.m_p) / (self.mAg - self.m_p)
                        center_nAg = int(round(nAg_float))
                        center_nAg = max(0, min(30, center_nAg))  # Clamp to valid range

                        # Test nAg ± 1 around the estimated center
                        for test_nAg in range(max(0, center_nAg - 1), min(30, center_nAg + 1) + 1):
                            # For each nAg, find best Qcl
                            # COMPLEX MODE: For complex, N0 = 0 always, so only test Qcl = nAg
                            is_complex_fallback = custom_xna and custom_xna.get('is_complex', False)
                            if is_complex_fallback:
                                qcl_range_fallback = [test_nAg]  # Only Qcl = nAg (N0 = 0)
                            else:
                                qcl_range_fallback = range(0, test_nAg + 1)

                            best_qcl_error = float('inf')
                            best_qcl = 0
                            for qcl in qcl_range_fallback:
                                est_mass = base_mass + test_nAg * self.mAg + adduct_mass - (qcl + charge + adduct_charge) * self.m_p
                                est_mz = est_mass / charge
                                qcl_error = abs(est_mz - peak_mz)
                                if qcl_error < best_qcl_error:
                                    best_qcl_error = qcl_error
                                    best_qcl = qcl

                            # Calculate with best Qcl
                            est_mass = base_mass + test_nAg * self.mAg + adduct_mass - (best_qcl + charge + adduct_charge) * self.m_p
                            est_mz = est_mass / charge

                            # Estimate isotope centroid shift
                            centroid_shift = test_nAg * 0.97 / charge
                            estimated_centroid = est_mz + centroid_shift

                            # Use centroid-based error for better ranking
                            mz_error = abs(estimated_centroid - peak_mz)

                            phase1_candidates.append((mz_error, test_strands, test_nAg, adduct_name, est_mz, estimated_centroid))

                # Sort by estimated centroid error and take top 10
                phase1_candidates.sort(key=lambda x: x[0])
                top_candidates = phase1_candidates[:10]

                logger.info(f"Phase 1 results (top 10 of {len(phase1_candidates)}) - using estimated centroid:")
                for i, (err, strands, nAg, adduct, est_mz, est_centroid) in enumerate(top_candidates):
                    logger.debug(f"{i+1}. strands={strands}, nAg={nAg}, adduct={adduct}, est_X0={est_centroid:.2f}, error={err:.4f}")

                # PHASE 2: Generate isotope patterns for top 10
                logger.info("Phase 2: Generating isotope patterns for top 10...")
                for mz_error, test_strands, best_nAg, adduct_name, _, _ in top_candidates:
                    adduct_mass, adduct_charge = self.adducts[adduct_name]
                    # For complex mode, generate strand_label
                    phase2_strand_label = None
                    if custom_xna and custom_xna.get('is_complex', False):
                        num_complexes = test_strands // 2
                        phase2_strand_label = f'nd={num_complexes}'
                    # For complex mode, use adduct_custom_xna which has the correct strand formula
                    comps = self.calculate_dna_silver_composition_with_adduct(
                        peak_mz, charge, dna_sequence,
                        adduct_name, adduct_mass, adduct_charge,
                        detected_centroid=exp_x0, resolution=resolution,
                        mz_values=mz_values, intensity_values=intensity_values,
                        nAg_center=best_nAg, nAg_range=1,  # Small range since we already found optimal
                        num_strands=test_strands,
                        custom_xna=adduct_custom_xna,  # Use complex-aware custom_xna
                        strand_label=phase2_strand_label,  # Pass strand_label for complex mode detection
                        conjugate_name=conjugate_name, conjugate_count=conjugate_count
                    )
                    if comps:
                        all_adduct_candidates.extend(comps)

                logger.info(f"Phase 2 found {len(all_adduct_candidates)} fallback candidates")

                # STEP 4: Refine fallback with pattern matching, then compare
                if all_adduct_candidates:
                    if mz_values is not None and intensity_values is not None:
                        # Refine fallback compositions with pattern matching
                        logger.info("Refining fallback compositions with pattern matching...")
                        refined_fallback, _, _, _, _, _, _, _, _ = self.refine_compositions_with_isotope_matching(
                            all_adduct_candidates,
                            mz_values, intensity_values,
                            peak_mz,  # Add missing peak_mz argument
                            resolution=resolution,
                            detected_centroid=exp_x0
                        )
                        all_adduct_candidates = refined_fallback if refined_fallback else all_adduct_candidates

                    # Rank adduct fallback by combined score to compare against baseline
                    all_adduct_candidates.sort(key=lambda x: -(
                        x.get('pattern_score', 0.0) - 0.1 * x.get('abs_x0_error', 999.0)
                    ))
                    best_fallback = all_adduct_candidates[0]
                    fallback_x0_error = best_fallback.get('abs_x0_error', 999.0)
                    fallback_pattern_score = best_fallback.get('pattern_score', 0.0)

                    logger.info("Comparison:")
                    logger.info(f"Baseline: {best_no_adduct['formula']}, X0 error={baseline_error:.4f}, pattern={baseline_pattern_score:.2f}")
                    logger.info(f"Fallback: {best_fallback['formula']}, X0 error={fallback_x0_error:.4f}, pattern={fallback_pattern_score:.2f}")

                    # Compare: Pattern score is MORE important than X₀ error
                    # Use 5% tolerance for pattern score comparison (0.05 difference)
                    pattern_difference = fallback_pattern_score - baseline_pattern_score

                    if pattern_difference > 0.05:
                        # Fallback has significantly better pattern match
                        logger.info(f"Fallback has better pattern match ({fallback_pattern_score:.2f} > {baseline_pattern_score:.2f})! Returning fallback")
                        return all_adduct_candidates
                    elif abs(pattern_difference) <= 0.05 and fallback_x0_error < baseline_error:
                        # Same pattern quality, but fallback has better X₀
                        logger.info(f"Fallback has similar pattern ({fallback_pattern_score:.2f} ~ {baseline_pattern_score:.2f}) but better X0! Returning fallback")
                        return all_adduct_candidates
                    else:
                        # Baseline is better
                        logger.info(f"Baseline is better (pattern={baseline_pattern_score:.2f}, X0={baseline_error:.4f})! Returning baseline composition")
                        return compositions_no_adduct
                else:
                    logger.warning("Fallback found nothing, returning baseline")
                    return compositions_no_adduct
            else:
                return compositions_no_adduct

        logger.debug(f"Testing {len(promising_adducts)} promising adducts: {[a[0] for a in promising_adducts]}")

        # Collect ALL candidates (no-adduct + promising adducts only)
        all_candidates = list(compositions_no_adduct)  # Start with baseline

        # Determine strand counts to test based on baseline X₀ error
        # If baseline error is high (> 0.5), test adjacent strand counts too
        if baseline_error > 0.5:
            strand_counts_to_test = [baseline_strands]
            if baseline_strands > 1:
                strand_counts_to_test.append(baseline_strands - 1)
            if baseline_strands < 3:
                strand_counts_to_test.append(baseline_strands + 1)
            logger.warning(f"Baseline X0 error high ({baseline_error:.2f}), testing strand counts: {strand_counts_to_test}")
        else:
            strand_counts_to_test = [baseline_strands]

        logger.debug(f"ADDUCT LOOP: {len(promising_adducts)} adducts x {len(strand_counts_to_test)} strands x {len(conjugate_counts_to_try)} conj_counts")
        for adduct_name, _ in promising_adducts:
            adduct_mass, adduct_charge = self.adducts[adduct_name]

            for test_strands in strand_counts_to_test:
                # Resolve 'all' marker: all strands conjugated first (mixed only if needed)
                resolved_conj_counts = []
                for c in conjugate_counts_to_try:
                    if c == 'all':
                        resolved_conj_counts.append(test_strands)  # All conjugated
                    else:
                        resolved_conj_counts.append(c)
                # Try each conjugate count for each strand/adduct combination
                for test_conj_count in resolved_conj_counts:
                    test_conj_name = conjugate_name if test_conj_count > 0 else None

                    # Get base mass for this strand count
                    if adduct_custom_xna and adduct_custom_xna.get('formula'):
                        # XNA mode with formula
                        mXNA_one = adduct_custom_xna.get('molecular_weight')
                        if mXNA_one is None:
                            mXNA_one = self.calculate_mass_from_formula(adduct_custom_xna['formula'])
                        test_base_mass = mXNA_one * test_strands
                    else:
                        # DNA mode or DNA-only Complex mode (no formula)
                        nH_t, nC_t, nN_t, nO_t, nP_t = self.calculate_dna_composition(dna_sequence, strands=test_strands)
                        test_base_mass = self.m_p * nH_t + self.mC * nC_t + self.mN * nN_t + self.mO * nO_t + self.mP * nP_t

                        # Add conjugate mass if present (total semantics: conjugate_count IS total, not per-strand)
                        # Note: custom['mass'] already includes ALL atoms (H,C,N,O,P,S,etc.)
                        if test_conj_name and test_conj_count > 0:
                            for custom in self.custom_adducts:
                                if custom['name'] == test_conj_name:
                                    test_base_mass += custom['mass'] * test_conj_count
                                    break

                    # Algebraically compute optimal nAg for this adduct + strand + conjugate combination
                    # Formula: peak_mz * z = base_mass + nAg * mAg + adduct_mass - (Qcl + z + adduct_charge) * mH
                    # Assuming Qcl ≈ nAg: nAg = (peak_mz * z - base_mass - adduct_mass + (z + adduct_charge) * mH) / (mAg - mH)
                    nAg_float = (peak_mz * charge - test_base_mass - adduct_mass + (charge + adduct_charge) * self.m_p) / (self.mAg - self.m_p)
                    optimal_nAg = int(round(nAg_float))
                    optimal_nAg = max(0, min(30, optimal_nAg))  # Clamp to valid range

                    # For complex mode, generate strand_label (nd=1, nd=2, nd=3)
                    adduct_strand_label = None
                    if adduct_custom_xna and adduct_custom_xna.get('is_complex', False):
                        num_complexes = test_strands // 2
                        adduct_strand_label = f'nd={num_complexes}'

                    logger.debug(f"Generating {adduct_name} candidates (strands={test_strands}, conj={test_conj_count}x{test_conj_name or 'none'}, nAg={optimal_nAg}±1, mass: {adduct_mass:.4f} Da, charge: {adduct_charge:+d})...")

                    # Generate candidates with this adduct using smart N0 search
                    compositions_with_adduct = self.calculate_dna_silver_composition_with_adduct(
                        peak_mz, charge, dna_sequence,
                        adduct_name, adduct_mass, adduct_charge,
                        detected_centroid=exp_x0, resolution=resolution,
                        mz_values=mz_values, intensity_values=intensity_values,
                        nAg_center=optimal_nAg, nAg_range=1,
                        num_strands=test_strands,
                        custom_xna=adduct_custom_xna,
                        strand_label=adduct_strand_label,
                        conjugate_name=test_conj_name, conjugate_count=test_conj_count
                    )

                    if compositions_with_adduct:
                        all_candidates.extend(compositions_with_adduct)
                        logger.debug(f"Generated {len(compositions_with_adduct)} candidates for {adduct_name} strands={test_strands} conj={test_conj_count}x{test_conj_name or 'none'} nAg={optimal_nAg}")

        logger.debug(f"Adduct search complete: {len(all_candidates)} total candidates")
        # STEP 4: Refine adduct candidates with PATTERN MATCHING
        if all_candidates and mz_values is not None and intensity_values is not None:
            logger.info("Refining adduct candidates with pattern matching...")
            refined_candidates, _, _, _, _, _, _, _, _ = self.refine_compositions_with_isotope_matching(
                all_candidates,
                mz_values, intensity_values,
                peak_mz,
                resolution=resolution,
                detected_centroid=exp_x0,
                skip_asymmetric_filter=True  # Don't filter clusters in adduct search
            )
            all_candidates = refined_candidates if refined_candidates else all_candidates

        # Sort by smallest |X0 error| (primary ranking criterion)
        logger.debug(f"Sorting {len(all_candidates)} candidates by |X0 error| (ascending)...")
        all_candidates.sort(key=lambda x: x.get('abs_x0_error', 999.0))

        best_x0_err = min((c.get('abs_x0_error', 999.0) for c in all_candidates), default=999.0) if all_candidates else 999.0
        if best_x0_err > 0.5:
            dimer_result = self._try_dimer_fallback(peak_mz, charge, dna_sequence, exp_x0,
                resolution, mz_values, intensity_values, custom_xna, conjugate_name, conjugate_count, kwargs)
            if dimer_result:
                dimer_best_x0 = min(c.get('abs_x0_error', 999.0) for c in dimer_result)
                if dimer_best_x0 < best_x0_err:
                    logger.info(f"Dimer fallback better: X₀={dimer_best_x0:.4f} < {best_x0_err:.4f}")
                    all_candidates = dimer_result
                else:
                    logger.info(f"Dimer fallback not better: X₀={dimer_best_x0:.4f} >= {best_x0_err:.4f}")

        if not all_candidates:
            logger.warning("No compositions found")
            return compositions_no_adduct

        best_comp = all_candidates[0]

        logger.info("=== Result ===")
        if best_comp.get('adduct'):
            logger.info(f"Best match: {best_comp['num_silver']} Ag + {best_comp['adduct']}")
            logger.info(f"Pattern score: {best_comp.get('pattern_score', 0.0):.2f}")
            logger.info(f"X0 error: {best_comp['abs_x0_error']:.4f} m/z")
            logger.info(f"Improvement: {baseline_error - best_comp['abs_x0_error']:.4f} m/z")
            logger.info(f"Formula: {best_comp['formula']}")
        else:
            logger.info("Best match: NO ADDUCT")
            logger.info(f"Pattern score: {best_comp.get('pattern_score', 0.0):.2f}")
            logger.info(f"X0 error: {best_comp['abs_x0_error']:.4f} m/z")
            logger.info(f"Formula: {best_comp['formula']}")

        return all_candidates

    def smooth_gaussian_pattern(self, barip: list[list], fwhm: float,
                                 num_points_per_fwhm: int = 100) -> list[list]:
        """
        Generate smooth Gaussian isotope pattern with guaranteed high resolution sampling.

        Args:
            barip: Bar isotope pattern [[mz_values], [intensities]]
            fwhm: Full width at half maximum
            num_points_per_fwhm: Number of sampling points per FWHM (default: 100 for very smooth curves)

        Returns:
            [[mz_values], [intensities]] - smoothed Gaussian pattern
        """
        if not barip or len(barip[0]) == 0:
            return [[], []]

        mz_bar = np.array(barip[0])
        int_bar = np.array(barip[1])

        # Define m/z range: extend ±3*FWHM from min/max peaks (covers 99.7% of Gaussian)
        mz_min = np.min(mz_bar) - 3 * fwhm
        mz_max = np.max(mz_bar) + 3 * fwhm

        # Calculate step size for smooth curve: FWHM / num_points_per_fwhm
        step = fwhm / num_points_per_fwhm

        # Generate high-resolution m/z grid
        mz_grid = np.arange(mz_min, mz_max + step, step)
        intensity_grid = np.zeros_like(mz_grid)

        # Sigma (standard deviation) from FWHM: FWHM = 2.355 * sigma
        sigma = fwhm / 2.355

        # Sum Gaussian peaks for each isotope
        for center_mz, height in zip(mz_bar, int_bar):
            # Generate normalized Gaussian: exp(-(x-center)^2 / (2*sigma^2))
            gaussian_contrib = np.exp(-0.5 * ((mz_grid - center_mz) / sigma) ** 2)
            # Scale by height
            intensity_grid += gaussian_contrib * height

        # Normalize to 100
        if np.max(intensity_grid) > 0:
            intensity_grid = (intensity_grid / np.max(intensity_grid)) * 100.0

        return [mz_grid.tolist(), intensity_grid.tolist()]

    def generate_isotope_pattern(self, formula: str, charge: int = 1, resolution: int = 20000) -> dict:
        """
        Generate isotope pattern for a given formula.
        Dispatches to either IsoSpecPy (faster) or PythoMS based on ISOTOPE_LIBRARY setting.
        Returns both bar pattern and Gaussian pattern.

        Uses global cache for speed optimization.

        Parameters:
            formula: Chemical formula string
            charge: Charge state
            resolution: MS resolution (default 20000 is fallback when webapp cannot parse from uploaded data)
        """
        global _isotope_pattern_cache, _ISOTOPE_CACHE_MAX_SIZE, ISOTOPE_LIBRARY

        # Check cache first
        cache_key = (formula, charge, resolution)
        if cache_key in _isotope_pattern_cache:
            logger.debug(f"[generate_isotope_pattern] CACHE HIT for {formula[:30]}... (z={charge})")
            return _isotope_pattern_cache[cache_key]

        logger.debug(f"[generate_isotope_pattern] CACHE MISS for {formula[:30]}... (z={charge}) - computing...")
        # Dispatch to appropriate library
        if ISOTOPE_LIBRARY == 'isospec' and ISOSPEC_AVAILABLE:
            result = self._generate_isotope_pattern_isospec(formula, charge, resolution)
        else:
            result = self._generate_isotope_pattern_pythoms(formula, charge, resolution)

        # Cache the result (with size limit) if successful
        if 'error' not in result:
            if len(_isotope_pattern_cache) >= _ISOTOPE_CACHE_MAX_SIZE:
                # Remove oldest entry (first key)
                oldest_key = next(iter(_isotope_pattern_cache))
                del _isotope_pattern_cache[oldest_key]
            _isotope_pattern_cache[cache_key] = result

        return result

    def _consolidate_formula(self, formula: str) -> str:
        """
        Consolidate a formula with duplicate elements into standard form.
        E.g., 'C304H368N128O184P30Ag28N2H8' -> 'C304H376N130O184P30Ag28'

        IsoSpecPy requires each element to appear only once.
        """
        import re
        # Parse formula: find all element-count pairs
        # Matches element symbols (1-2 letters, first uppercase) followed by optional count
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)

        # Consolidate counts for each element
        element_counts: dict[str, int] = {}
        for element, count in matches:
            if element:  # Skip empty matches
                count = int(count) if count else 1
                element_counts[element] = element_counts.get(element, 0) + count

        # Rebuild formula in a standard order (C, H, N, O, P, S, then others alphabetically)
        priority_order = ['C', 'H', 'N', 'O', 'P', 'S']
        result = []

        # Add priority elements first
        for elem in priority_order:
            if elem in element_counts:
                count = element_counts.pop(elem)
                result.append(f"{elem}{count}" if count > 1 else elem)

        # Add remaining elements alphabetically
        for elem in sorted(element_counts.keys()):
            count = element_counts[elem]
            result.append(f"{elem}{count}" if count > 1 else elem)

        return ''.join(result)

    def _generate_isotope_pattern_isospec(self, formula: str, charge: int = 1, resolution: int = 20000) -> dict:
        """
        Generate isotope pattern using IsoSpecPy (faster than PythoMS for large molecules).
        """
        try:
            # Consolidate formula to handle duplicate elements (e.g., from adducts)
            # IsoSpecPy requires each element to appear only once
            consolidated_formula = self._consolidate_formula(formula)

            # prob_to_cover=0.9999 captures 99.99% of the isotope distribution
            iso_result = isospec.IsoTotalProb(formula=consolidated_formula, prob_to_cover=0.9999)

            # IsoSpecPy returns CFFI objects - must convert to list first
            masses = np.array(list(iso_result.masses))
            probs = np.array(list(iso_result.probs))

            if len(masses) == 0:
                return {'error': 'IsoSpecPy returned empty pattern'}

            # Sort by mass first
            sort_idx = np.argsort(masses)
            masses = masses[sort_idx]
            probs = probs[sort_idx]

            # Get monoisotopic mass (first peak after sorting, before any filtering)
            monoisotopic_mass = masses[0]

            # Calculate molecular weight (weighted average of ALL peaks)
            molecular_weight = np.average(masses, weights=probs)

            # Convert to m/z (apply charge)
            # The formula passed is already the ION formula (protons already removed)
            # So we just divide by charge, same as PythoMS does
            mz_values = masses / abs(charge)

            # Bin peaks FIRST to match PythoMS behavior (combine peaks at similar m/z)
            # IsoSpecPy returns fine-grained peaks, PythoMS aggregates by nominal mass
            # Use 0.2 Da bins to match typical isotope spacing
            bin_width = 0.2 / abs(charge)
            min_mz = mz_values.min()
            max_mz = mz_values.max()
            bins = np.arange(min_mz - bin_width/2, max_mz + bin_width, bin_width)

            # Digitize: assign each peak to a bin
            bin_indices = np.digitize(mz_values, bins)

            # Aggregate peaks in each bin
            binned_mz = []
            binned_int = []
            for i in range(1, len(bins)):
                mask = bin_indices == i
                if np.any(mask):
                    # Weighted average for m/z, sum for intensity
                    bin_probs = probs[mask]
                    bin_mzs = mz_values[mask]
                    binned_mz.append(np.average(bin_mzs, weights=bin_probs))
                    binned_int.append(np.sum(bin_probs))

            if not binned_mz:
                return {'error': 'IsoSpecPy: no peaks after binning'}

            mz_values = np.array(binned_mz)
            probs = np.array(binned_int)

            # Normalize AFTER binning to max = 1.0 (like PythoMS)
            probs = probs / np.max(probs)

            # Filter out low intensity peaks (threshold=0.01 like PythoMS)
            threshold = 0.01
            mask = probs >= threshold
            mz_values = mz_values[mask]
            probs = probs[mask]

            if len(mz_values) == 0:
                return {'error': 'IsoSpecPy: all peaks below threshold after filtering'}

            # Create bar isotope pattern in PythoMS format
            barip = [mz_values.tolist(), probs.tolist()]

            # Calculate FWHM for Gaussian smoothing
            if len(mz_values) > 0:
                theoretical_mz = mz_values[0]
            else:
                theoretical_mz = monoisotopic_mass / abs(charge)

            fwhm = theoretical_mz / resolution

            # Generate Gaussian pattern using the same smooth function
            gaussian_pattern = self.smooth_gaussian_pattern(barip, fwhm, num_points_per_fwhm=100)

            # Sort Gaussian pattern by m/z
            if gaussian_pattern and len(gaussian_pattern[0]) > 0:
                gaussian_mz = np.array(gaussian_pattern[0])
                gaussian_int = np.array(gaussian_pattern[1])
                sort_idx = np.argsort(gaussian_mz)
                gaussian_mz_sorted = gaussian_mz[sort_idx].tolist()
                gaussian_int_sorted = gaussian_int[sort_idx].tolist()
            else:
                gaussian_mz_sorted = []
                gaussian_int_sorted = []

            return {
                'mz': barip[0],
                'intensity': barip[1],
                'gaussian_mz': gaussian_mz_sorted,
                'gaussian_intensity': gaussian_int_sorted,
                'monoisotopic_mass': monoisotopic_mass,
                'molecular_weight': molecular_weight,
            }
        except Exception as e:
            # Fall back to PythoMS if IsoSpecPy fails
            logger.warning(f"IsoSpecPy failed for {formula}: {e}, falling back to PythoMS")
            return self._generate_isotope_pattern_pythoms(formula, charge, resolution)

    def _generate_isotope_pattern_pythoms(self, formula: str, charge: int = 1, resolution: int = 20000) -> dict:
        """
        Generate isotope pattern using PythoMS (original implementation).
        """
        try:
            mol = IPMolecule(
                formula,
                charge=charge,
                resolution=resolution,
                verbose=False,
                ipmethod='hybrid',
                dropmethod='threshold',
                threshold=0.01
            )

            # Get bar isotope pattern
            barip = mol.bar_isotope_pattern

            # Calculate theoretical m/z for FWHM calculation
            # Use the first m/z value from bar pattern (monoisotopic peak)
            if len(barip[0]) > 0:
                theoretical_mz = barip[0][0]  # First m/z value = monoisotopic peak
            else:
                # Fallback to old method if bar pattern is empty
                theoretical_mass = mol.monoisotopic_mass
                theoretical_mz = (theoretical_mass - charge * self.m_p) / charge

            # Generate Gaussian pattern using custom smooth function
            fwhm = theoretical_mz / resolution

            # Use custom smooth Gaussian generation instead of PythoMS version
            gaussian_pattern = self.smooth_gaussian_pattern(barip, fwhm, num_points_per_fwhm=100)

            # Sort Gaussian pattern by m/z to prevent zigzag plotting
            if gaussian_pattern and len(gaussian_pattern[0]) > 0:
                gaussian_mz = np.array(gaussian_pattern[0])
                gaussian_int = np.array(gaussian_pattern[1])

                # Sort by m/z
                sort_idx = np.argsort(gaussian_mz)
                gaussian_mz_sorted = gaussian_mz[sort_idx].tolist()
                gaussian_int_sorted = gaussian_int[sort_idx].tolist()
            else:
                gaussian_mz_sorted = []
                gaussian_int_sorted = []

            return {
                'mz': barip[0],
                'intensity': barip[1],
                'gaussian_mz': gaussian_mz_sorted,
                'gaussian_intensity': gaussian_int_sorted,
                'monoisotopic_mass': mol.monoisotopic_mass,
                'molecular_weight': mol.molecular_weight,
            }
        except Exception as e:
            return {'error': str(e)}

    def detect_peak_boundaries(self, mz_array: npt.NDArray[np.float64], int_array: npt.NDArray[np.float64],
                                peak_mz: float) -> tuple[float, float, float]:
        """
        Detect the boundaries of a single isotope envelope from experimental data.

        NEW APPROACH:
        1. Find APEX (highest point) in small window around clicked position
        2. From apex, scan left/right to find valleys (local minima)
        3. This ensures we identify the correct peak without jumping to adjacent ones

        Returns: (left_boundary_mz, right_boundary_mz, apex_mz)
        """
        # Find the index closest to the clicked peak
        peak_idx = np.argmin(np.abs(mz_array - peak_mz))

        # STEP 1: Find the APEX (highest point) in a SMALL window around clicked position
        # Use small window (±10 points) to avoid jumping to adjacent peaks
        search_window = 10  # Small window to stay on same peak
        search_start = max(0, peak_idx - search_window)
        search_end = min(len(int_array), peak_idx + search_window)

        # Find apex within this small region
        local_region_intensities = int_array[search_start:search_end]
        apex_idx_in_region = np.argmax(local_region_intensities)
        apex_idx = search_start + apex_idx_in_region

        apex_mz = mz_array[apex_idx]
        apex_intensity = int_array[apex_idx]

        logger.debug(f"Detecting isotope envelope boundaries around clicked m/z={peak_mz:.4f}")
        logger.debug(f"Clicked at index={peak_idx}, mz={mz_array[peak_idx]:.4f}")
        logger.debug(f"Found APEX at index={apex_idx}, mz={apex_mz:.4f}, intensity={apex_intensity:.0f}")

        # STEP 2: From APEX, scan LEFT to find valley (local minimum)
        left_idx = apex_idx
        min_intensity_left = apex_intensity

        for i in range(apex_idx - 1, max(0, apex_idx - 200), -1):
            current_intensity = int_array[i]

            # Track the minimum intensity as we scan left
            if current_intensity < min_intensity_left:
                min_intensity_left = current_intensity
                left_idx = i

            # Stop if intensity starts rising significantly (found the valley)
            # Look for 2 consecutive points rising by >10%
            if i >= 1:
                if (int_array[i-1] > current_intensity * 1.1 and
                    int_array[i] > int_array[i+1] * 1.1):
                    # Found a valley - intensity is rising on the left
                    logger.debug(f"Left valley at index={left_idx}, mz={mz_array[left_idx]:.4f}, intensity={int_array[left_idx]:.0f}")
                    break

        # If we hit the edge without finding a valley, use the minimum we found
        if left_idx == apex_idx:
            logger.debug(f"Left boundary at edge: index={left_idx}, mz={mz_array[left_idx]:.4f}")

        # STEP 3: From APEX, scan RIGHT to find valley (local minimum)
        right_idx = apex_idx
        min_intensity_right = apex_intensity

        for i in range(apex_idx + 1, min(len(int_array), apex_idx + 200)):
            current_intensity = int_array[i]

            # Track the minimum intensity as we scan right
            if current_intensity < min_intensity_right:
                min_intensity_right = current_intensity
                right_idx = i

            # Stop if intensity starts rising significantly (found the valley)
            # Look for 2 consecutive points rising by >10%
            if i < len(int_array) - 1:
                if (int_array[i+1] > current_intensity * 1.1 and
                    int_array[i] > int_array[i-1] * 1.1):
                    # Found a valley - intensity is rising on the right
                    logger.debug(f"Right valley at index={right_idx}, mz={mz_array[right_idx]:.4f}, intensity={int_array[right_idx]:.0f}")
                    break

        # If we hit the edge without finding a valley, use the minimum we found
        if right_idx == apex_idx:
            logger.debug(f"Right boundary at edge: index={right_idx}, mz={mz_array[right_idx]:.4f}")

        left_boundary_mz = mz_array[left_idx]
        right_boundary_mz = mz_array[right_idx]
        width = right_boundary_mz - left_boundary_mz
        num_points = right_idx - left_idx + 1

        logger.debug(f"Final envelope: [{left_boundary_mz:.4f}, {right_boundary_mz:.4f}] m/z (width={width:.4f}, {num_points} points)")
        logger.debug(f"Apex at {apex_mz:.4f} (Gaussian will use MIDPOINT of boundaries as initial guess)")

        return left_boundary_mz, right_boundary_mz, apex_mz

    def validate_adduct_mass_match(self, comp: dict, peak_mz: float,
                                    tolerance_ppm: float = 2000.0) -> tuple[bool, float, float, float]:
        """
        Validate that the remaining mass after subtracting cluster matches the adduct.

        Logic:
        - observed_mass = peak_mz × z + (Qcl + z + adduct_charge) × mH
        - cluster_mass = DNA_mass + nAg × Ag_mass
        - remaining_mass = observed_mass - cluster_mass

        For valid compositions:
        - With adduct: remaining_mass should be close to adduct_mass AND >= adduct_mass
        - No adduct: remaining_mass should be close to 0

        Args:
            comp: Composition dictionary
            peak_mz: Observed peak m/z
            tolerance_ppm: Allowed error in ppm (default 2000)

        Returns:
            (is_valid, remaining_mass, expected_adduct_mass, error_ppm)
        """
        # Get composition parameters
        z = comp.get('z', comp.get('charge', 1))  # 'z' is used in composition dicts
        qcl = comp.get('qcl', 0)
        nAg = comp.get('num_silver', 0)
        adduct = comp.get('adduct', '')
        adduct_mass = comp.get('adduct_mass', 0.0)
        adduct_charge = comp.get('adduct_charge', 0)

        # Get DNA mass from composition (stored during smart_n0_search)
        dna_mass = comp.get('dna_neutral_mass', 0.0)

        if dna_mass == 0.0:
            # Fallback: calculate from formula if available
            nC = comp.get('nC', 0)
            nH = comp.get('nH', 0)
            nN = comp.get('nN', 0)
            nO = comp.get('nO', 0)
            nP = comp.get('nP', 0)
            dna_mass = nC * self.mC + nH * self.m_p + nN * self.mN + nO * self.mO + nP * self.mP

        if dna_mass == 0.0:
            # Can't validate without DNA mass
            logger.debug(f"Cannot validate adduct: missing DNA mass for {comp.get('ion_formula', 'unknown')}")
            return (True, 0.0, adduct_mass, 0.0)  # Pass by default if we can't validate

        # Calculate observed neutral mass
        # From: expected_mz = (neutral_mass - (Qcl + z + adduct_charge) × mH) / z
        # So: neutral_mass = expected_mz × z + (Qcl + z + adduct_charge) × mH
        observed_neutral_mass = peak_mz * z + (qcl + z + adduct_charge) * self.m_p

        # Calculate cluster mass (DNA + Ag, without adduct)
        # NOTE: dna_neutral_mass already includes all strands (element counts multiplied by num_strands)
        cluster_mass = dna_mass + nAg * self.mAg

        # Remaining mass should equal adduct mass
        remaining_mass = observed_neutral_mass - cluster_mass

        # Expected adduct mass (0 for no-adduct)
        expected_adduct_mass = adduct_mass if adduct else 0.0

        # Calculate error - use neutral mass as denominator since we're comparing neutral masses
        mass_diff = remaining_mass - expected_adduct_mass
        error_ppm = abs(mass_diff) / observed_neutral_mass * 1e6 if observed_neutral_mass > 0 else 999999.0

        # Validation criteria:
        # 1. Error should be within tolerance
        # 2. For adducts: remaining_mass MUST be >= adduct_mass (can't have negative contribution)
        #    If remaining < expected, the adduct is definitely WRONG
        is_within_tolerance = error_ppm < tolerance_ppm

        if adduct:
            # STRICT CHECK: remaining_mass must be >= expected_adduct_mass
            # If remaining < expected, this adduct is impossible
            # Allow small tolerance (5%) for measurement/calibration error
            MIN_REMAINING_RATIO = 0.95
            is_mass_reasonable = remaining_mass >= (expected_adduct_mass * MIN_REMAINING_RATIO)
            is_valid = is_within_tolerance and is_mass_reasonable
        else:
            # No adduct: remaining mass should be close to 0
            is_valid = is_within_tolerance

        logger.debug(f"Adduct validation: {comp.get('ion_formula', 'unknown')}")
        logger.debug(f"  observed_neutral={observed_neutral_mass:.4f}, cluster={cluster_mass:.4f}")
        logger.debug(f"  remaining={remaining_mass:.4f}, expected_adduct={expected_adduct_mass:.4f}")
        logger.debug(f"  error={error_ppm:.1f}ppm, valid={is_valid}")

        return (is_valid, remaining_mass, expected_adduct_mass, error_ppm)

    def refine_compositions_with_isotope_matching(self, compositions: list[dict],
                                                   experimental_mz: npt.NDArray[np.float64],
                                                   experimental_int: npt.NDArray[np.float64],
                                                   peak_mz: float, resolution: int = 20000,
                                                   detected_centroid: Optional[float] = None,
                                                   skip_asymmetric_filter: bool = False) -> tuple:
        """
        Refine composition candidates by matching their theoretical isotope patterns
        to the experimental spectrum around the peak
        Returns: (refined_compositions, experimental_x0, experimental_sigma, has_other_strands, all_compositions, has_odd_n0_warning, exp_mz_gaussian, exp_int_gaussian)
        """
        logger.debug(f"refine_compositions_with_isotope_matching called with resolution={resolution}")

        # Filter out invalid compositions (must be dict, not str or other types)
        valid_compositions = [c for c in compositions if isinstance(c, dict)]
        if len(valid_compositions) != len(compositions):
            logger.warning(f"Filtered out {len(compositions) - len(valid_compositions)} invalid (non-dict) compositions")
        compositions = valid_compositions

        if len(compositions) == 0:
            return compositions, None, None, False, [], False, None, None, False

        # STEP 1: Use fixed window for consistency with custom search
        # Fixed 3.0 m/z window gives more reliable exp_x0 than adaptive boundary detection
        window = 3.0
        mask = (experimental_mz >= peak_mz - window) & (experimental_mz <= peak_mz + window)
        exp_mz_window = experimental_mz[mask]
        exp_int_window = experimental_int[mask]

        if len(exp_mz_window) == 0:
            return compositions, None, None, False, [], False, None, None, False

        # Calculate peak symmetry using the proven calculate_peak_symmetry() method
        symmetry_info = self.calculate_peak_symmetry(
            experimental_mz, experimental_int, peak_mz, window=2.0
        )
        symmetry_percent = symmetry_info.get('symmetry_score', 0.0) * 100
        is_symmetric_peak = symmetry_percent >= 60.0  # Threshold: 60% symmetry or better

        logger.debug(f"Peak symmetry = {symmetry_percent:.1f}%")
        if is_symmetric_peak:
            logger.debug("Peak is SYMMETRIC (>=60%) - will attempt Gaussian fitting")
        else:
            logger.debug("Peak is ASYMMETRIC (<60%) - will use weighted average")

        # FILTER: Asymmetric peaks likely indicate non-cluster compositions
        # Filter out nanocluster compositions when peak is asymmetric
        # Skip this filter for adduct search refinement (STEP 4) where conjugate clusters may look asymmetric
        if not is_symmetric_peak and not skip_asymmetric_filter:
            original_count = len(compositions)
            original_compositions = compositions.copy()  # Backup for fallback
            # Keep non-cluster types including conjugate-only compositions
            non_cluster_types = ['DNA Only', 'XNA Only', 'DNA+Ag ion', 'XNA+Ag ion', 'DNA/XNA Only', 'DNA/XNA+Conjugate']
            compositions = [c for c in compositions
                          if c.get('type') in non_cluster_types]
            filtered_count = len(compositions)

            if filtered_count < original_count and filtered_count > 0:
                logger.warning(f"ASYMMETRIC PEAK FILTER: Removed {original_count - filtered_count} nanocluster compositions")
                logger.info(f"Keeping only {filtered_count} non-cluster compositions (DNA/XNA Only, DNA+Ag ion)")
            elif filtered_count == 0 and original_count > 0:
                logger.warning("Asymmetric peak but no non-cluster compositions found")
                logger.info(f"Keeping original {original_count} nanocluster compositions as fallback")
                # Restore original compositions if we filtered everything out
                compositions = original_compositions

        # Generate smooth Gaussian envelope from experimental data
        exp_mz_gaussian, exp_int_gaussian = self.generate_experimental_gaussian_envelope(
            exp_mz_window, exp_int_window, resolution
        )

        # Use detected_centroid if provided (e.g., from manual Gaussian fit)
        # Otherwise, fit Gaussian from experimental data
        if detected_centroid is not None:
            exp_x0 = detected_centroid
            exp_sigma = None
            logger.debug(f"Using provided detected_centroid: exp_x0={exp_x0:.4f}")
        elif exp_mz_gaussian is not None and exp_int_gaussian is not None:
            fit_result = self.gaussian_fit_centroid(exp_mz_gaussian, exp_int_gaussian)
            if fit_result and fit_result[0] is not None:
                exp_x0 = fit_result[0]
                exp_sigma = fit_result[1]
            else:
                exp_x0, exp_sigma = None, None
        else:
            exp_x0, exp_sigma = None, None

        if exp_x0 is None:
            # Fallback: if envelope generation/fitting fails, use weighted average
            logger.debug("Envelope generation failed, using weighted average")
            exp_x0, exp_sigma = self.weighted_average_centroid(exp_mz_window, exp_int_window)
            exp_mz_gaussian = exp_mz_window
            exp_int_gaussian = exp_int_window
            logger.debug(f"exp_x0={exp_x0:.4f} (weighted average fallback)")

        # Score each composition based on isotope pattern match
        for comp in compositions:
            # Skip invalid compositions (should be dict, not string)
            if not isinstance(comp, dict):
                logger.warning(f"Skipping invalid composition: expected dict, got {type(comp).__name__}")
                continue

            # ALWAYS recalculate theo_x0 from the pattern to ensure it uses the mass-corrected value
            # Previously we were skipping recalculation if theo_x0 was already set, which caused display issues
            theo_x0_already_calculated = False  # Force recalculation

            try:
                # Generate theoretical isotope pattern using ION formula (deprotonated)
                logger.debug(f"Generating isotope pattern for {comp['ion_formula']} (z={comp['z']}, nAg={comp.get('num_silver', '?')}, strands={comp.get('num_strands', '?')})")

                theo_pattern = self.generate_isotope_pattern(
                    comp['ion_formula'],
                    comp['z'],
                    resolution
                )

                if 'error' not in theo_pattern:
                    theo_mz_arr = np.array(theo_pattern['gaussian_mz'])
                    theo_int_arr = np.array(theo_pattern['gaussian_intensity'])
                    comp['theo_mz'] = theo_pattern['gaussian_mz']
                    comp['theo_intensity'] = theo_pattern['gaussian_intensity']

                    # Theoretical X₀ from Gaussian centroid fit
                    theo_x0 = None
                    comp['theo_sigma'] = None
                    if len(theo_mz_arr) > 0 and np.sum(theo_int_arr) > 0:
                        theo_fit_result = self.gaussian_fit_centroid(theo_mz_arr, theo_int_arr)
                        if theo_fit_result and theo_fit_result[0] is not None:
                            theo_x0 = theo_fit_result[0]
                            comp['theo_sigma'] = float(theo_fit_result[2]) if len(theo_fit_result) > 2 and theo_fit_result[2] is not None else None
                        else:
                            theo_x0 = np.sum(theo_mz_arr * theo_int_arr) / np.sum(theo_int_arr)
                    comp['theo_x0'] = float(theo_x0) if theo_x0 is not None else None

                    # X₀ error
                    if not theo_x0_already_calculated:
                        if theo_x0 is not None and exp_x0 is not None:
                            x0_error_calculated = abs(theo_x0 - exp_x0)
                            comp['x0_error'] = x0_error_calculated
                            comp['abs_x0_error'] = x0_error_calculated
                            comp['exp_x0'] = float(exp_x0)
                        else:
                            comp['x0_error'] = 999.0
                            comp['abs_x0_error'] = 999.0
                            logger.warning(f"X0 calc failed for {comp['ion_formula']}: theo_x0={theo_x0}, exp_x0={exp_x0}")

                    # Pattern similarity (theoretical sticks vs experimental apexes)
                    theo_stick_mz = np.array(theo_pattern['mz'])
                    theo_stick_int = np.array(theo_pattern['intensity'])
                    pattern_similarity = self.calculate_pattern_similarity(
                        theo_stick_mz, theo_stick_int, exp_mz_window, exp_int_window
                    )
                    comp['pattern_similarity'] = float(pattern_similarity)
                    comp['pattern_score'] = float(pattern_similarity)

                else:
                    logger.warning(f"Isotope pattern generation returned error for {comp['ion_formula']} (z={comp['z']})")
                    logger.warning(f"Error details: {theo_pattern.get('error', 'Unknown error')}")
                    comp['x0_error'] = 999.0
                    comp['theo_mz'] = []
                    comp['theo_intensity'] = []
                    comp['theo_x0'] = None
                    comp['theo_sigma'] = None
            except Exception as e:
                # Handle case where comp might not be a dict (shouldn't happen, but be defensive)
                if isinstance(comp, dict):
                    ion_formula = comp.get('ion_formula', 'unknown')
                    logger.exception(f"Exception during isotope matching for {ion_formula}: {str(e)}")
                    comp['x0_error'] = 999.0
                    comp['theo_mz'] = []
                    comp['theo_intensity'] = []
                    comp['theo_x0'] = None
                    comp['theo_sigma'] = None
                else:
                    logger.exception(f"Exception during isotope matching (comp is {type(comp).__name__}, not dict): {str(e)}")

        # DEDUPLICATION: Remove duplicate compositions
        # Duplicates occur when same (strands, nAg, z, Qcl) is generated by multiple code paths
        logger.debug(f"Before deduplication: {len(compositions)} compositions")

        dedup_dict: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        for comp in compositions:
            # Skip invalid compositions
            if not isinstance(comp, dict):
                continue
            # Create unique key based on physical composition
            dedup_key = (comp['num_strands'], comp['num_silver'], comp['z'], comp['qcl'])

            if dedup_key not in dedup_dict:
                dedup_dict[dedup_key] = comp
            else:
                # Duplicate found - keep the better one
                existing = dedup_dict[dedup_key]

                # Compare X0 errors
                existing_error = abs(existing['x0_error']) if existing['x0_error'] is not None and existing['x0_error'] != 999.0 else 999.0
                new_error = abs(comp['x0_error']) if comp['x0_error'] is not None and comp['x0_error'] != 999.0 else 999.0

                # If errors are essentially the same, prefer by type
                if abs(existing_error - new_error) < 0.0001:
                    # Type priority for N0=0 cases: DNA+Ag ion > DNA/XNA Only > nanocluster
                    # (N0=0 means no nanocluster, just ionic silver)
                    n0_val = comp.get('n0', 0)
                    if n0_val == 0:
                        type_priority = {'DNA+Ag ion': 3, 'DNA Only': 2, 'XNA Only': 2, 'nanocluster': 1}
                        if type_priority.get(comp['type'], 0) > type_priority.get(existing['type'], 0):
                            dedup_dict[dedup_key] = comp
                            logger.debug(f"Replaced {existing['type']} with {comp['type']} (same error, N0=0)")
                    # For N0>0, prefer nanocluster
                    else:
                        if comp['type'] == 'nanocluster' and existing['type'] != 'nanocluster':
                            dedup_dict[dedup_key] = comp
                            logger.debug(f"Replaced {existing['type']} with nanocluster (N0={n0_val}>0)")
                # Otherwise keep the one with lower error
                elif new_error < existing_error:
                    dedup_dict[dedup_key] = comp
                    logger.debug(f"Replaced (better error: {new_error:.4f} < {existing_error:.4f})")

        compositions = list(dedup_dict.values())
        logger.debug(f"After deduplication: {len(compositions)} compositions")

        # Sort by smallest |X0 error| (centroid match is the primary ranking criterion)
        def x0_sort_key(comp):
            x0_err = comp.get('x0_error', 999.0)
            return abs(x0_err) if x0_err not in [None, 999.0] else 999.0

        compositions.sort(key=x0_sort_key)

        # Debug: Log top 10 compositions with their scores
        logger.debug("Top 10 compositions sorted by |X0 error| (ascending):")
        for i, comp in enumerate(compositions[:10]):
            pattern_sim = comp.get('pattern_similarity', 0.0)
            x0_err = comp.get('x0_error', 999.0)
            x0_err_str = f"{x0_err:.4f}" if x0_err not in [None, 999.0] else "N/A"
            logger.debug(f"{i+1}. X0_err={x0_err_str}, pattern_sim={pattern_sim:.3f}, N0={comp.get('n0', '?')}, Qcl={comp.get('qcl', '?')}, type={comp.get('type', '?')}, strands={comp.get('num_strands', '?')}, nAg={comp.get('num_silver', '?')}")

        # STEP 1: For EACH Qcl, compare no-adduct vs with-adduct, keep ONLY the best one
        # This ensures each Qcl has only ONE composition (the winner)
        best_comp = None
        has_odd_n0_warning = False

        logger.debug("Comparing no-adduct vs with-adduct FOR EACH Qcl individually...")

        # Group compositions by (strands, nAg, Qcl) - compare adducts within each group
        qcl_groups: dict[tuple[int, int, int | None], list[dict[str, Any]]] = {}
        for comp in compositions:
            if comp['type'] == 'nanocluster' and comp.get('x0_error', 999.0) != 999.0:
                group_key: tuple[int, int, int | None] = (comp['num_strands'], comp['num_silver'], comp.get('qcl'))
                if group_key not in qcl_groups:
                    qcl_groups[group_key] = []
                qcl_groups[group_key].append(comp)

        logger.debug(f"Found {len(qcl_groups)} unique (strands, nAg, Qcl) groups")

        # For each group, keep ONLY the best composition (no-adduct vs with-adduct)
        filtered_compositions: list[dict[str, Any]] = []
        for group_key, group_comps in qcl_groups.items():
            strands, nAg, qcl = group_key

            # ADDUCT MASS VALIDATION: Filter out compositions where remaining < expected adduct
            adduct_valid_comps = []
            for c in group_comps:
                is_valid, remaining, expected, error_ppm = self.validate_adduct_mass_match(c, peak_mz)
                if is_valid:
                    adduct_valid_comps.append(c)
                else:
                    adduct_name = c.get('adduct', 'none')
                    logger.debug(f"Adduct validation FAILED: {adduct_name}, remaining={remaining:.2f}, expected={expected:.2f}")

            if adduct_valid_comps:
                best_in_group = min(adduct_valid_comps, key=lambda c: abs(c.get('x0_error', 999.0)))
                skipped = len(group_comps) - len(adduct_valid_comps)
            else:
                best_in_group = min(group_comps, key=lambda c: abs(c.get('x0_error', 999.0)))
                skipped = 0
                logger.warning(f"No compositions passed adduct validation for Qcl={qcl}, nAg={nAg}")
            skip_msg = f" (skipped {skipped} by validation)" if skipped > 0 else ""

            # Check if it's adduct or not
            has_adduct = bool(best_in_group.get('adduct', ''))
            adduct_name = best_in_group.get('adduct', 'none')
            x0_err = best_in_group.get('x0_error', 999.0)
            n0_val = best_in_group.get('n0', '?')

            logger.debug(f"Qcl={qcl}, nAg={nAg}, strands={strands}: {len(group_comps)} candidates -> Best: {'adduct=' + adduct_name if has_adduct else 'no-adduct'}, N0={n0_val}, X0={x0_err:.4f}{skip_msg}")

            filtered_compositions.append(best_in_group)

        # Keep also non-cluster compositions
        non_cluster = [c for c in compositions if c['type'] != 'nanocluster']
        filtered_compositions.extend(non_cluster)

        logger.info(f"After per-Qcl adduct filtering: {len(filtered_compositions)} compositions (reduced from {len(compositions)})")

        # Update compositions list
        compositions = filtered_compositions

        # Check if this is complex mode (N0=0 constraint for all complex strand types)
        # Complex mode labels: 'strand1', 'strand2', 'nd=X', 'complex'
        # Standard mode labels: '1strand', '2strand', '3strand', etc. or None
        is_complex_mode = any(self.is_complex_strand_label(c.get('strand_type')) for c in compositions)
        # Find the overall best composition
        valid_comps = [c for c in compositions if c.get('x0_error', 999.0) != 999.0]

        if valid_comps:
            # Check if conjugate is present — if so, use combined score (pattern + X0)
            # to prevent wrong mixed-conjugation compositions from winning on X0 alone
            has_conjugate = any(c.get('extra_conj_formula') or c.get('extra_conj_mass', 0) > 0 for c in valid_comps)
            if has_conjugate:
                def _conj_combined_score(comp):
                    pattern_sim = comp.get('pattern_similarity', 0.0)
                    x0_err = abs(comp.get('x0_error', 999.0)) if comp.get('x0_error') not in [None, 999.0] else 999.0
                    return -(pattern_sim - x0_err * 0.1)
                sorted_comps = sorted(valid_comps, key=_conj_combined_score)
                logger.info("Conjugate detected: ranking by combined score (pattern similarity + X0 error)")
            else:
                sorted_comps = sorted(valid_comps, key=lambda c: abs(c.get('x0_error', 999.0)))
            best_comp = sorted_comps[0]

            # Check for unrealistic composition: nAg > 20 with (N₀ ≤ 5 OR N₀ > 20)
            nAg = best_comp.get('num_silver', 0)
            n0 = best_comp.get('n0', 0)
            is_unrealistic = nAg > 20 and n0 is not None and (n0 <= 5 or n0 > 20)
            if is_unrealistic:
                reason = f"N0={n0} <= 5" if n0 <= 5 else f"N0={n0} > 20"
                logger.warning(f"Best match has nAg={nAg} > 20 with {reason} (unrealistic)")
                # Look for next best composition that doesn't have this issue
                for alt_comp in sorted_comps[1:]:
                    alt_nAg = alt_comp.get('num_silver', 0)
                    alt_n0 = alt_comp.get('n0', 0)
                    alt_unrealistic = alt_nAg > 20 and alt_n0 is not None and (alt_n0 <= 5 or alt_n0 > 20)
                    if not alt_unrealistic:
                        logger.info(f"Demoting to next best: nAg={alt_nAg}, N0={alt_n0}")
                        best_comp = alt_comp
                        break

            # ADDUCT MASS VALIDATION: Check if remaining mass matches adduct
            # This is the final check - remaining_mass = peak × z - DNA - Ag should ≈ adduct_mass
            # Use higher tolerance (5000 ppm) to account for systematic calibration offsets
            # This validation helps distinguish between clearly wrong adducts (e.g., 2Na vs Cl)
            ADDUCT_VALIDATION_TOLERANCE_PPM = 5000.0
            is_valid, remaining_mass, expected_adduct, error_ppm = self.validate_adduct_mass_match(
                best_comp, peak_mz, ADDUCT_VALIDATION_TOLERANCE_PPM
            )

            if not is_valid:
                adduct_str_check = best_comp.get('adduct', 'none')
                logger.warning(f"ADDUCT VALIDATION FAILED for top candidate: adduct={adduct_str_check}")
                logger.warning(f"  remaining_mass={remaining_mass:.4f}, expected={expected_adduct:.4f}, error={error_ppm:.1f}ppm")

                # Try next candidates until we find one that passes validation
                found_valid = False
                for alt_comp in sorted_comps[1:]:
                    alt_valid, alt_remaining, alt_expected, alt_error = self.validate_adduct_mass_match(
                        alt_comp, peak_mz, ADDUCT_VALIDATION_TOLERANCE_PPM
                    )
                    if alt_valid:
                        alt_adduct = alt_comp.get('adduct', 'none')
                        alt_x0 = alt_comp.get('x0_error', 999.0)
                        logger.info(f"ADDUCT VALIDATION: Selecting alternative with valid adduct mass")
                        logger.info(f"  adduct={alt_adduct}, remaining={alt_remaining:.4f}, expected={alt_expected:.4f}")
                        logger.info(f"  X0_error={alt_x0:.4f} (was {best_comp.get('x0_error', 999.0):.4f})")
                        best_comp = alt_comp
                        found_valid = True
                        break

                if not found_valid:
                    logger.warning("ADDUCT VALIDATION: No valid candidates found, keeping best X0 match")

            logger.info("SELECTED by smallest X0 error (after N0 and adduct validation)")

            n0_val = best_comp.get('n0', 0)
            is_even = n0_val % 2 == 0
            pattern_sim = best_comp.get('pattern_similarity', 0.0)
            x0_err = best_comp.get('x0_error', 999.0)
            has_adduct = bool(best_comp.get('adduct', ''))
            adduct_str = best_comp.get('adduct', 'none')

            logger.info("FINAL SELECTION FOR Qcl FILTERING:")
            logger.info(f"Strands={best_comp['num_strands']}, nAg={best_comp.get('num_silver', '?')}")
            logger.info(f"X0_err={x0_err:.4f}, pattern_sim={pattern_sim:.3f}")
            logger.info(f"N0={n0_val} ({'EVEN' if is_even else 'ODD'}), Qcl={best_comp.get('qcl', '?')}")
            logger.info(f"Adduct: {adduct_str if has_adduct else 'none'}")

            if n0_val > 0 and not is_even:
                has_odd_n0_warning = True
                logger.warning(f"Best match has ODD N0={n0_val} - possible calibration issue!")

        # If no valid composition found, use first composition
        if best_comp is None:
            logger.warning("No valid composition found, using first available")
            best_comp = compositions[0] if len(compositions) > 0 else None

        if best_comp is None:
            return [], None, None, False, [], False, None, None, False

        # Get the best Qcl for filtering
        best_qcl = best_comp.get('qcl') if best_comp['type'] == 'nanocluster' else None
        logger.info(f"Best composition Qcl = {best_qcl}")

        # STEP 2: Filter compositions
        # Keep: (1) non-cluster, (2) within Qcl±3 of best
        # Using ±3 to ensure the actual best N₀±1 is included after frontend's dynamic X₀ recalculation
        good_compositions = [
            comp for comp in compositions
            if (comp['type'] != 'nanocluster') or  # Keep all non-cluster
               (best_qcl is not None and comp.get('qcl') is not None and abs(comp['qcl'] - best_qcl) <= 3)  # Qcl±3 of best
        ]

        if len(good_compositions) == 0:
            logger.warning("No compositions passed filtering. Keeping all.")
            good_compositions = compositions
        else:
            nanocluster_count = len([c for c in good_compositions if c['type'] == 'nanocluster'])
            non_cluster_count = len([c for c in good_compositions if c['type'] != 'nanocluster'])
            logger.info(f"Kept {nanocluster_count} nanoclusters (Qcl±3 of best={best_qcl}) + {non_cluster_count} non-cluster")
            compositions = good_compositions

        # STEP 3: Re-sort filtered compositions by PATTERN SIMILARITY (highest first)
        # Now that we've filtered to the correct Qcl range, rank by how well patterns match
        def final_x0_sort_key(comp: dict[str, Any]) -> float:
            x0_err = comp.get('x0_error', 999.0)
            return abs(x0_err) if x0_err not in [None, 999.0] else 999.0

        compositions.sort(key=final_x0_sort_key)  # Smallest |X0 error| first

        logger.debug("Final ranking (after Qcl filter, sorted by |X0 error|):")
        for i, comp in enumerate(compositions[:5]):
            pattern_sim = comp.get('pattern_similarity', 0.0)
            x0_err = comp.get('x0_error', 999.0)
            logger.debug(f"{i+1}. |X0|={final_x0_sort_key(comp):.4f} (sim={pattern_sim:.3f}, X0={x0_err:.4f}), Qcl={comp.get('qcl', '?')}, N0={comp.get('n0', '?')}, strands={comp.get('num_strands', '?')}")

        # STEP 3: Check if there are good compositions with different strand numbers
        strands_available: dict[int, list[dict[str, Any]]] = {}
        for comp in compositions:
            if comp['type'] == 'nanocluster':
                ns = comp['num_strands']
                if ns not in strands_available:
                    strands_available[ns] = []
                strands_available[ns].append(comp)

        result_compositions = []
        has_other_strands = False

        # Check if other strand numbers have good compositions (X0 error < 0.05)
        if best_comp['type'] == 'nanocluster':
            best_strands = best_comp['num_strands']
            for ns, comps in strands_available.items():
                if ns != best_strands:
                    # Check if any composition with this strand number has good X0 error
                    if any(c['x0_error'] < 0.05 for c in comps):
                        has_other_strands = True
                        break

        # Strategy: Return ALL valid compositions for nanoclusters (all Qcl from 0 to nAg)
        # where "best" means N0 is EVEN (not necessarily minimal error)
        if best_comp['type'] == 'nanocluster':
            best_strands = best_comp['num_strands']
            best_nAg = best_comp['num_silver']
            best_z = best_comp['z']
            best_qcl = best_comp['qcl']

            logger.info(f"Selected best composition with N0={best_comp['n0']} (even), Qcl={best_qcl}, X0 error={best_comp['x0_error']:.4f}")
            if has_other_strands:
                logger.info("Other strand numbers also have good matches (X0 < 0.05)")

            # Create a dictionary to hold compositions by Qcl value
            # IMPORTANT: Only include compositions with the SAME adduct as best composition
            best_adduct = best_comp.get('adduct', '')
            comps_by_qcl: dict[int, dict[str, Any]] = {}
            for comp in compositions:
                if (comp['type'] == 'nanocluster' and
                    comp['num_strands'] == best_strands and
                    comp['num_silver'] == best_nAg and
                    comp['z'] == best_z and
                    comp.get('adduct', '') == best_adduct):  # Must match adduct!
                    qcl = comp['qcl']
                    if qcl is None:
                        continue  # Skip compositions without Qcl
                    # Keep the best X0 error for each Qcl
                    if qcl not in comps_by_qcl or comp['abs_x0_error'] < comps_by_qcl[qcl]['abs_x0_error']:
                        comps_by_qcl[qcl] = comp

            # Get compositions around the best Qcl
            # IMPORTANT: Include Qcl±3 range to ensure the actual best N₀±1 is always shown
            # This is needed because the frontend recalculates X₀ error dynamically,
            # which may identify a slightly different "best" composition
            # COMPLEX MODE: For complex, N0 = 0 always, so only return Qcl = nAg
            if is_complex_mode:
                qcl_values = [best_nAg]  # Only Qcl = nAg (N0 = 0)
                logger.debug(f"COMPLEX MODE: Only returning N0=0 (Qcl={best_nAg})")
            elif best_qcl <= 2:
                # At/near lower boundary: show [0, 1, 2, ..., up to 6]
                qcl_values = list(range(0, min(7, best_nAg + 1)))
            elif best_qcl >= best_nAg - 2:
                # At/near upper boundary: show [nAg-6, ..., nAg]
                qcl_values = list(range(max(0, best_nAg - 6), best_nAg + 1))
            else:
                # Normal case: show [qcl-3, qcl-2, qcl-1, qcl, qcl+1, qcl+2, qcl+3]
                qcl_values = list(range(max(0, best_qcl - 3), min(best_qcl + 4, best_nAg + 1)))

            # Remove duplicates and ensure all are valid
            qcl_values = sorted(list(set([q for q in qcl_values if 0 <= q <= best_nAg])))

            logger.info(f"Showing compositions for Qcl values: {qcl_values}")

            for qcl_val in qcl_values:
                if qcl_val in comps_by_qcl:
                    result_compositions.append(comps_by_qcl[qcl_val])
                else:
                    # Generate missing composition on-the-fly
                    logger.debug(f"Generating missing composition for Qcl={qcl_val}")

                    # Get adduct info from best_comp FIRST (needed for N0 calculation)
                    adduct_name = best_comp.get('adduct', '')
                    adduct_mass = best_comp.get('adduct_mass', 0.0)
                    adduct_charge = best_comp.get('adduct_charge', 0)

                    # Formula: N₀ + Qcl = nAg (always, regardless of adducts)
                    # Therefore: N₀ = nAg - Qcl
                    n0_valence = best_nAg - qcl_val

                    if n0_valence >= 0:  # N0 can be 0 (DNA + Ag+ ions)
                        # Use the DNA composition from best_comp (same DNA for all Qcl variants)
                        # IMPORTANT: Also inherit adduct from best_comp!
                        nH = best_comp['nH']
                        nC = best_comp['nC']
                        nN = best_comp['nN']
                        nO = best_comp['nO']
                        nP = best_comp['nP']

                        mH_total = self.m_p * nH
                        mC_total = self.mC * nC
                        mN_total = self.mN * nN
                        mO_total = self.mO * nO
                        mP_total = self.mP * nP
                        mAg_total = self.mAg * best_nAg

                        # Get extra conjugate contribution from best_comp
                        extra_conj_mass_otf = best_comp.get('extra_conj_mass', 0.0)
                        extra_conj_formula_otf = best_comp.get('extra_conj_formula', '')

                        # Get custom_xna from best_comp for XNA mode detection
                        custom_xna = best_comp.get('custom_xna')

                        # Calculate expected m/z using user-provided mass for XNA, or calculated mass for DNA
                        # protons_removed = Qcl + z + adduct_charge
                        if custom_xna and custom_xna.get('molecular_weight') is not None:
                            # XNA mode: use user-provided mass (include adduct)
                            xna_neutral_mass = custom_xna['molecular_weight'] * best_strands + mAg_total + adduct_mass
                            mass = xna_neutral_mass - (qcl_val + best_z + adduct_charge) * self.m_p
                            expected_mz = mass / best_z
                            adduct_str = f"+{adduct_name}" if adduct_name else ""
                            logger.debug(f"Expected m/z (XNA{adduct_str}, missing Qcl={qcl_val}): Using user mass {custom_xna['molecular_weight']:.2f} Da -> expected_mz = {expected_mz:.4f}")
                        else:
                            # DNA mode: use calculated mass from elements (including extra conjugate mass)
                            dna_neutral_mass = mP_total + mH_total + mC_total + mN_total + mO_total + extra_conj_mass_otf + mAg_total + adduct_mass
                            mass = dna_neutral_mass - (qcl_val + best_z + adduct_charge) * self.m_p
                            expected_mz = mass / best_z
                        mass_error_ppm = abs((expected_mz - peak_mz) / peak_mz * 1e6)

                        # Build formulas - check if XNA mode for proper formatting
                        if custom_xna:
                            # XNA mode: use subscript format
                            xna_name = custom_xna['name']
                            if adduct_name:
                                neutral_formula = f"({xna_name}){to_subscript(best_strands)}Ag{to_subscript(best_nAg)}+{adduct_name}"
                            else:
                                neutral_formula = f"({xna_name}){to_subscript(best_strands)}Ag{to_subscript(best_nAg)}"
                        else:
                            # DNA mode: use element formula
                            if adduct_name:
                                neutral_formula = f"C{nC}H{nH}N{nN}O{nO}P{nP}{extra_conj_formula_otf}Ag{best_nAg}+{adduct_name}"
                            else:
                                neutral_formula = f"C{nC}H{nH}N{nN}O{nO}P{nP}{extra_conj_formula_otf}Ag{best_nAg}"

                        nH_ion = nH - (qcl_val + best_z + adduct_charge)  # protons_removed = Qcl + z + adduct_charge
                        ion_formula = f"C{nC}H{nH_ion}N{nN}O{nO}P{nP}{extra_conj_formula_otf}Ag{best_nAg}"
                        if adduct_name:
                            adduct_formula = self.adduct_name_to_formula(adduct_name)
                            ion_formula += adduct_formula

                        # Generate isotope pattern for this composition
                        try:
                            theo_pattern = self.generate_isotope_pattern(ion_formula, best_z, resolution)
                            if 'error' not in theo_pattern:
                                theo_mz = theo_pattern['gaussian_mz']
                                theo_intensity = theo_pattern['gaussian_intensity']

                                # Use smooth Gaussian pattern for theo_x0 (same method as exp_x0)
                                theo_mz_gaussian = np.array(theo_pattern['gaussian_mz'])
                                theo_int_gaussian = np.array(theo_pattern['gaussian_intensity'])
                                if len(theo_mz_gaussian) > 0 and np.sum(theo_int_gaussian) > 0:
                                    # Fit Gaussian to smooth theoretical pattern to extract x0 parameter
                                    theo_fit_result = self.gaussian_fit_centroid(theo_mz_gaussian, theo_int_gaussian)
                                    if theo_fit_result and theo_fit_result[0] is not None:
                                        theo_x0 = theo_fit_result[0]
                                        # theo_fit_result[2] = fitting uncertainty (not sigma width)
                                        theo_sigma = theo_fit_result[2] if len(theo_fit_result) > 2 else None
                                        logger.debug(f"Theo X0 (Gaussian fit): {theo_x0:.4f}, uncertainty={theo_sigma:.6f}" if theo_sigma else f"Theo X0 (Gaussian fit): {theo_x0:.4f}")
                                    else:
                                        # Fallback to weighted average if Gaussian fit fails
                                        theo_x0 = np.sum(theo_mz_gaussian * theo_int_gaussian) / np.sum(theo_int_gaussian)
                                        theo_sigma = None  # No uncertainty available for weighted average
                                        logger.debug(f"Theo X0 (weighted avg): {theo_x0:.4f}")

                                    # Calculate X0 error as: |theo_x0 - exp_x0|
                                    if theo_x0 is not None and exp_x0 is not None:
                                        x0_error = abs(theo_x0 - exp_x0)
                                        logger.debug(f"Exp X0: {exp_x0:.4f}, X0 error: {x0_error:.4f}")
                                    else:
                                        x0_error = None
                                else:
                                    theo_x0 = None
                                    theo_sigma = None
                                    x0_error = 999.0
                            else:
                                x0_error = 999.0
                                theo_mz = []
                                theo_intensity = []
                                theo_x0 = None
                                theo_sigma = None
                        except:
                            x0_error = 999.0
                            theo_mz = []
                            theo_intensity = []
                            theo_x0 = None
                            theo_sigma = None

                        comp_type = self.determine_composition_type(
                            best_nAg, n0_valence, is_complex=is_complex_mode,
                            custom_xna=custom_xna)

                        # For display: displayed_qcl = qcl + adduct_charge
                        displayed_qcl = qcl_val + adduct_charge

                        result_compositions.append({
                            'type': comp_type,
                            'num_strands': best_strands,
                            'num_silver': best_nAg,
                            'qcl': qcl_val,  # Internal Qcl (N₀ + Qcl = nAg always)
                            'displayed_qcl': displayed_qcl,  # For display: qcl + adduct_charge
                            'n0': n0_valence,
                            'z': best_z,
                            'formula': neutral_formula,
                            'ion_formula': ion_formula,
                            'neutral_formula': neutral_formula,
                            'adduct': adduct_name,
                            'adduct_mass': adduct_mass,
                            'adduct_charge': adduct_charge,
                            'full_notation': f"{neutral_formula}-{qcl_val+best_z}H (z={best_z}, Qcl={displayed_qcl}, N0={n0_valence})",
                            'expected_mz': expected_mz,
                            'mass_error_ppm': mass_error_ppm,
                            'x0_error': x0_error,
                            'abs_x0_error': abs(x0_error) if x0_error is not None else 999.0,
                            'theo_x0': float(theo_x0) if theo_x0 is not None else None,
                            'exp_x0': float(exp_x0) if exp_x0 is not None else None,
                            'theo_sigma': float(theo_sigma) if theo_sigma is not None else None,
                            'theo_mz': theo_mz,
                            'theo_intensity': theo_intensity,
                            'nH': nH, 'nC': nC, 'nN': nN, 'nO': nO, 'nP': nP,
                            'extra_conj_mass': extra_conj_mass_otf,
                            'extra_conj_formula': extra_conj_formula_otf,
                            'custom_xna': best_comp.get('custom_xna')
                        })

            # Sort compositions by Qcl for consistent display order
            result_compositions.sort(key=lambda x: x['qcl'])

            # Log how many compositions we're returning
            actual_qcl_values = sorted([c['qcl'] for c in result_compositions])
            logger.info(f"Returning {len(result_compositions)} compositions for display")
            logger.debug(f"Qcl values: {actual_qcl_values}")
            if len(result_compositions) < len(qcl_values):
                logger.warning(f"Generated {len(result_compositions)}/{len(qcl_values)} expected compositions")
                missing_qcl = [q for q in qcl_values if q not in [c['qcl'] for c in result_compositions]]
                logger.debug(f"Missing Qcl values: {missing_qcl}")

        else:
            # For DNA+Ag ions, just return the best match
            result_compositions = [best_comp]

        # Return: compositions for display, X0, sigma, flag for other strands, all compositions, has_odd_n0_warning, and shifted experimental Gaussian pattern for display
        return result_compositions, exp_x0, exp_sigma, has_other_strands, compositions, has_odd_n0_warning, exp_mz_gaussian, exp_int_gaussian, False

    def group_isotope_envelope(self, peak_mz: npt.NDArray[np.float64], peak_intensity: npt.NDArray[np.float64],
                                charge: Optional[int]) -> Optional[int]:
        """
        Group peaks that belong to the same isotope envelope
        Returns the index of the most intense peak (representative peak)
        """
        if charge is None or charge <= 0:
            return None

        # Expected spacing for this charge state
        spacing = 1.003 / charge

        # Find the most intense peak in this envelope
        return int(np.argmax(peak_intensity))

    def estimate_resolution(self, mz_values: npt.NDArray[np.float64], intensity_values: npt.NDArray[np.float64]) -> int:
        """
        Estimate the resolution of the spectrum using PythoMS methodology
        """
        try:
            res = autoresolution(list(mz_values), list(intensity_values), n=10, v=False)
            if res is None or not np.isfinite(res) or res <= 0:
                return 20000  # Default resolution
            return int(res)
        except:
            return 20000  # Default resolution if estimation fails

    def calculate_fwhm(self, mz: float, resolution: int) -> float:
        """
        Calculate Full Width at Half Maximum for a given m/z and resolution
        FWHM = m/z / resolution
        """
        return mz / resolution

    def find_local_maximum(self, mz_values: npt.NDArray[np.float64], intensity_values: npt.NDArray[np.float64],
                            center_mz: float, lookwithin: Optional[float] = None) -> tuple[Optional[float], Optional[float]]:
        """
        Find the local maximum within a window around center_mz
        Based on PythoMS localmax function
        """
        if lookwithin is None:
            lookwithin = 1.0

        # Find indices within the window
        left_idx = np.searchsorted(mz_values, center_mz - lookwithin, side='left')
        right_idx = np.searchsorted(mz_values, center_mz + lookwithin, side='right')

        if left_idx >= right_idx:
            return None, None

        # Find maximum in the window
        window_intensities = intensity_values[left_idx:right_idx]
        if len(window_intensities) == 0:
            return None, None

        max_intensity = np.max(window_intensities)
        max_idx_in_window = np.argmax(window_intensities)
        max_idx = left_idx + max_idx_in_window

        return mz_values[max_idx], max_intensity

    def find_peak_regions(self, mz_values: npt.NDArray[np.float64], intensity_values: npt.NDArray[np.float64],
                          threshold: float = 0.05, merge_gap: float = 1.5) -> list[tuple[int, int]]:
        """
        Find isotope envelope regions - each ENVELOPE is one region
        Merges nearby regions that are likely part of the same isotope envelope
        merge_gap: merge regions separated by less than this m/z (default 1.5)
        """
        norm_intensity = intensity_values / np.max(intensity_values)

        # Find regions above threshold
        above_threshold = norm_intensity > threshold

        # Find all continuous regions above threshold
        regions = []
        in_region = False
        start_idx = 0

        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_region:
                # Start of new region
                start_idx = i
                in_region = True
            elif not above_threshold[i] and in_region:
                # End of region
                end_idx = i - 1
                regions.append((start_idx, end_idx))
                in_region = False

        # Handle case where last region extends to end
        if in_region:
            regions.append((start_idx, len(above_threshold) - 1))

        # Merge regions that are close together (likely same isotope envelope)
        if len(regions) <= 1:
            return regions

        merged_regions = []
        current_start, current_end = regions[0]

        for i in range(1, len(regions)):
            next_start, next_end = regions[i]

            # Check gap between current region end and next region start
            gap = mz_values[next_start] - mz_values[current_end]

            if gap < merge_gap:
                # Merge: extend current region to include next region
                current_end = next_end
            else:
                # Don't merge: save current region and start new one
                merged_regions.append((current_start, current_end))
                current_start, current_end = next_start, next_end

        # Add the last region
        merged_regions.append((current_start, current_end))

        return merged_regions

    def weighted_centroid(self, mz_values: npt.NDArray[np.float64], intensity_values: npt.NDArray[np.float64],
                          start_idx: int, end_idx: int) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate peak centroid (position of maximum intensity) matching PythoMS isotope overlay method

        This uses the m/z value at maximum intensity for peak position, which is consistent
        with how PythoMS's plot_mass_spectrum and localmax functions work.
        """
        region_mz = mz_values[start_idx:end_idx+1]
        region_int = intensity_values[start_idx:end_idx+1]

        if len(region_mz) == 0 or np.sum(region_int) == 0:
            return None, None

        # Find the m/z at maximum intensity (peak apex)
        # This matches PythoMS isotope overlay behavior
        max_idx = np.argmax(region_int)
        centroid_mz = region_mz[max_idx]
        max_intensity = region_int[max_idx]

        return centroid_mz, max_intensity

    def calculate_peak_symmetry(self, mz_values: npt.NDArray[np.float64], intensity_values: npt.NDArray[np.float64],
                                 center_mz: float, window: float = 2.0) -> dict:
        """
        Calculate symmetry of a peak around its center.
        Returns symmetry score (0-1, where 1 is perfectly symmetric)
        and skewness indicator.

        A symmetric peak suggests a clean, single species (like a nanocluster).
        An asymmetric peak may indicate fragmentation, impurities, or overlapping peaks.
        """
        # Extract region around peak
        mask = (mz_values >= center_mz - window) & (mz_values <= center_mz + window)
        region_mz = mz_values[mask]
        region_int = intensity_values[mask]

        if len(region_mz) < 5:
            return {
                'symmetry_score': 0.0,
                'skewness': 0.0,
                'is_symmetric': False,
                'note': 'Insufficient data points'
            }

        # Find peak apex
        max_idx = np.argmax(region_int)
        apex_mz = region_mz[max_idx]
        max_intensity = region_int[max_idx]

        # Divide into left and right sides from apex
        left_mz = region_mz[:max_idx+1]
        left_int = region_int[:max_idx+1]
        right_mz = region_mz[max_idx:]
        right_int = region_int[max_idx:]

        if len(left_mz) < 2 or len(right_mz) < 2:
            return {
                'symmetry_score': 0.0,
                'skewness': 0.0,
                'is_symmetric': False,
                'note': 'Peak too narrow'
            }

        # Calculate statistical skewness
        mean_mz = np.average(region_mz, weights=region_int)
        variance = np.average((region_mz - mean_mz)**2, weights=region_int)
        std_dev = np.sqrt(variance)

        if std_dev > 0:
            skewness = np.average(((region_mz - mean_mz) / std_dev)**3, weights=region_int)
        else:
            skewness = 0.0

        # Compare left and right sides by mirroring around apex
        max_distance = min(apex_mz - region_mz[0], region_mz[-1] - apex_mz)

        symmetry_scores = []
        symmetry_weights = []
        num_points = min(20, int(max_distance / 0.05))  # Finer sampling for better accuracy

        for i in range(1, num_points + 1):
            offset = (i / num_points) * max_distance

            # Find intensity at left and right positions
            left_pos = apex_mz - offset
            right_pos = apex_mz + offset

            # Interpolate intensities
            left_intensity = np.interp(left_pos, left_mz, left_int, left=0, right=0)
            right_intensity = np.interp(right_pos, right_mz, right_int, left=0, right=0)

            # Calculate local symmetry, weighted by average intensity
            # so high-signal regions near apex matter more than low-signal tails
            avg_intensity = (left_intensity + right_intensity) / 2.0
            if avg_intensity > 0:
                local_asym = abs(left_intensity - right_intensity) / (left_intensity + right_intensity)
                symmetry_scores.append(1.0 - local_asym)
                symmetry_weights.append(avg_intensity)

        # Overall symmetry score (intensity-weighted average)
        if symmetry_scores and symmetry_weights:
            symmetry_score = float(np.average(symmetry_scores, weights=symmetry_weights))
        else:
            symmetry_score = 0.0

        # Determine if peak is symmetric
        is_symmetric = symmetry_score > 0.7 and abs(skewness) < 0.5

        # Generate interpretation
        if symmetry_score > 0.85 and abs(skewness) < 0.3:
            note = 'Highly symmetric - likely clean nanocluster'
        elif symmetry_score > 0.7 and abs(skewness) < 0.5:
            note = 'Moderately symmetric - good quality'
        elif symmetry_score > 0.5:
            note = 'Slightly asymmetric - may have impurities'
        else:
            note = 'Asymmetric - possible fragmentation or overlapping peaks'

        return {
            'symmetry_score': float(symmetry_score),
            'skewness': float(skewness),
            'is_symmetric': bool(is_symmetric),  # Convert numpy bool to Python bool
            'note': note,
            'apex_mz': float(apex_mz)
        }

    def calculate_pattern_similarity(self, theo_mz: npt.NDArray[np.float64], theo_int: npt.NDArray[np.float64],
                                      exp_mz: npt.NDArray[np.float64], exp_int: npt.NDArray[np.float64],
                                      window: float = 3.0) -> float:
        """Mean of cosine similarity and Pearson correlation between matched theo/exp isotope peaks."""
        try:
            if len(theo_mz) == 0 or len(exp_mz) == 0:
                return 0.0

            from scipy.signal import find_peaks

            theo_mz = np.array(theo_mz, dtype=np.float64)
            theo_int = np.array(theo_int, dtype=np.float64)
            exp_mz = np.array(exp_mz, dtype=np.float64)
            exp_int = np.array(exp_int, dtype=np.float64)

            # Filter to significant sticks (>5% of max)
            sig_mask = theo_int > np.max(theo_int) * 0.05
            theo_mz = theo_mz[sig_mask]
            theo_int = theo_int[sig_mask]
            if len(theo_mz) < 2:
                return 0.0

            # Find experimental peak apexes
            spacing = np.median(np.diff(theo_mz))
            min_distance = int(spacing * 0.4 / np.median(np.diff(exp_mz))) if len(exp_mz) > 1 else 2
            peaks_idx, _ = find_peaks(exp_int, distance=max(2, min_distance),
                                      prominence=np.max(exp_int) * 0.02)

            if len(peaks_idx) < 2:
                apex_mz = exp_mz
                apex_int = exp_int
            else:
                apex_mz = exp_mz[peaks_idx]
                apex_int = exp_int[peaks_idx]

            # Match each stick to nearest apex within half-spacing tolerance
            match_tol = spacing * 0.5
            paired_theo = []
            paired_exp = []

            for i in range(len(theo_mz)):
                diffs = np.abs(apex_mz - theo_mz[i])
                nearest_idx = np.argmin(diffs)
                if diffs[nearest_idx] <= match_tol:
                    paired_theo.append(theo_int[i])
                    paired_exp.append(apex_int[nearest_idx])
                else:
                    paired_theo.append(theo_int[i])
                    paired_exp.append(0.0)

            if len(paired_theo) < 2:
                return 0.0

            paired_theo = np.array(paired_theo)
            paired_exp = np.array(paired_exp)

            # Normalize to max = 1
            paired_theo = paired_theo / (np.max(paired_theo) + 1e-10)
            paired_exp = paired_exp / (np.max(paired_exp) + 1e-10)

            # Cosine similarity
            cosine_sim = np.dot(paired_theo, paired_exp) / (
                np.linalg.norm(paired_theo) * np.linalg.norm(paired_exp) + 1e-10)
            cosine_sim = max(0.0, min(1.0, cosine_sim))

            # Pearson correlation
            if np.std(paired_theo) > 0 and np.std(paired_exp) > 0:
                correlation = np.corrcoef(paired_theo, paired_exp)[0, 1]
                correlation = max(0.0, min(1.0, correlation))
            else:
                correlation = 0.0

            return float((cosine_sim + correlation) / 2.0)

        except Exception as e:
            logger.exception(f"[calculate_pattern_similarity] Exception: {str(e)}")
            return 0.0
    def calculate_multi_parameter_fit_score(self, theo_mz: npt.NDArray[np.float64], theo_int: npt.NDArray[np.float64],
                                             exp_mz: npt.NDArray[np.float64], exp_int: npt.NDArray[np.float64],
                                             theo_x0: Optional[float], theo_sigma: Optional[float],
                                             exp_x0: Optional[float], exp_sigma: Optional[float]) -> tuple[float, dict]:
        """
        Calculate comprehensive fit score combining multiple parameters:
        1. X₀ error (centroid position)
        2. σ ratio (width matching)
        3. R² (curve overlap quality)

        Returns a composite score (lower is better) and individual metrics
        """
        try:
            if theo_x0 is None or exp_x0 is None or theo_sigma is None or exp_sigma is None:
                return 999.0, {'x0_error': 999.0, 'sigma_ratio': None, 'r_squared': None}

            # 1. X₀ error (absolute difference in centroid positions)
            x0_error = abs(exp_x0 - theo_x0)

            # 2. σ ratio (how well the widths match)
            # Ratio close to 1.0 means good width match
            sigma_ratio = theo_sigma / exp_sigma if exp_sigma > 0 else None
            sigma_deviation = abs(1.0 - sigma_ratio) if sigma_ratio else 999.0

            # 3. R² (coefficient of determination - curve overlap quality)
            # Need to align theoretical and experimental on same m/z grid
            try:
                # Find overlapping m/z range
                mz_min = max(np.min(theo_mz), np.min(exp_mz))
                mz_max = min(np.max(theo_mz), np.max(exp_mz))

                if mz_max <= mz_min:
                    r_squared = 0.0
                else:
                    # Create common m/z grid for comparison
                    mz_grid = np.linspace(mz_min, mz_max, 200)

                    # Interpolate both patterns onto common grid
                    theo_interp = np.interp(mz_grid, theo_mz, theo_int, left=0, right=0)
                    exp_interp = np.interp(mz_grid, exp_mz, exp_int, left=0, right=0)

                    # Normalize both to max=1 for fair comparison
                    theo_norm = theo_interp / np.max(theo_interp) if np.max(theo_interp) > 0 else theo_interp
                    exp_norm = exp_interp / np.max(exp_interp) if np.max(exp_interp) > 0 else exp_interp

                    # Calculate R² (coefficient of determination)
                    ss_res = np.sum((exp_norm - theo_norm) ** 2)  # Residual sum of squares
                    ss_tot = np.sum((exp_norm - np.mean(exp_norm)) ** 2)  # Total sum of squares
                    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    r_squared = max(0.0, min(1.0, r_squared))  # Clamp to [0, 1]

            except Exception as e:
                logger.error(f"[R-squared calculation failed]: {str(e)}")
                r_squared = 0.0

            # Composite score (weighted combination - lower is better)
            # Weight factors - adjust these based on importance
            w_x0 = 10.0        # X₀ error weight (m/z units)
            w_sigma = 5.0      # σ deviation weight
            w_r2 = 20.0        # R² weight (inverted since higher R² is better)

            composite_score = (
                w_x0 * x0_error +                    # Centroid position error
                w_sigma * sigma_deviation +          # Width mismatch
                w_r2 * (1.0 - r_squared)            # Shape overlap quality (inverted)
            )

            metrics = {
                'x0_error': float(x0_error),
                'sigma_ratio': float(sigma_ratio) if sigma_ratio else None,
                'sigma_deviation': float(sigma_deviation),
                'r_squared': float(r_squared),
                'composite_score': float(composite_score)
            }

            logger.debug(f"[Fit Score] X0_err={x0_error:.4f}, sigma_ratio={sigma_ratio:.3f}, R_squared={r_squared:.4f}, Score={composite_score:.2f}")

            return composite_score, metrics

        except Exception as e:
            logger.exception(f"[calculate_multi_parameter_fit_score] Exception: {str(e)}")
            return 999.0, {'x0_error': 999.0, 'sigma_ratio': None, 'r_squared': None}

    def generate_experimental_gaussian_envelope(self, exp_mz: npt.NDArray[np.float64], exp_int: npt.NDArray[np.float64],
                                                resolution: int) -> tuple[Optional[npt.NDArray], Optional[npt.NDArray]]:
        """
        Generate smooth Gaussian envelope for experimental data.
        Uses Gaussian smoothing with kernel based on instrument resolution.
        This will show the natural asymmetry of the experimental data.
        """
        try:
            logger.debug("GENERATE_EXPERIMENTAL_GAUSSIAN_ENVELOPE CALLED")
            logger.debug(f"Input: {len(exp_mz)} m/z points, resolution={resolution}")

            if len(exp_mz) == 0 or len(exp_int) == 0:
                logger.warning("FAILED: Empty input data")
                return None, None

            # Convert to numpy arrays
            exp_mz = np.array(exp_mz)
            exp_int = np.array(exp_int)

            # Calculate FWHM and sigma from resolution
            peak_center = np.average(exp_mz, weights=exp_int)
            fwhm = peak_center / resolution
            sigma = fwhm / 2.355  # Convert FWHM to sigma

            logger.debug(f"Peak center: {peak_center:.4f}, FWHM: {fwhm:.6f}, sigma: {sigma:.6f}")

            # SMART APPROACH: Find apex (local maximum) of each isotope peak
            # Then use the SAME smooth_gaussian_pattern function as theoretical data
            # This ensures consistent smooth curves!

            from scipy.signal import find_peaks

            # Find local maxima (apex of each isotope peak)
            # Use a small distance to separate isotope peaks (~0.2 Da for typical spacing)
            min_distance = int(0.2 / np.median(np.diff(exp_mz))) if len(exp_mz) > 1 else 2
            peaks_idx, properties = find_peaks(exp_int, distance=max(2, min_distance), prominence=np.max(exp_int) * 0.05)

            if len(peaks_idx) < 3:
                # Not enough peaks found - use all data points
                logger.debug(f"Found only {len(peaks_idx)} apex points, using all data")
                apex_mz = exp_mz
                apex_int = exp_int
            else:
                # Extract apex points
                all_apex_mz = exp_mz[peaks_idx]
                all_apex_int = exp_int[peaks_idx]
                logger.debug(f"Found {len(all_apex_mz)} apex points (local maxima)")

                # FILTER: keep apex points that form a contiguous series with the
                # most-intense apex. Walk left/right until a gap larger than
                # 2.5 × median isotope spacing is encountered (the next envelope).
                # This adapts to envelope width — narrow at low mass, wider at high
                # mass where many Ag atoms broaden the isotope distribution.
                max_apex_idx = int(np.argmax(all_apex_int))
                spacings = np.diff(all_apex_mz)
                median_spacing = float(np.median(spacings)) if len(spacings) >= 1 else 0.334
                gap_threshold = max(median_spacing * 3.0, 0.5)

                start = max_apex_idx
                end = max_apex_idx
                while end + 1 < len(all_apex_mz) and (all_apex_mz[end + 1] - all_apex_mz[end]) <= gap_threshold:
                    end += 1
                while start - 1 >= 0 and (all_apex_mz[start] - all_apex_mz[start - 1]) <= gap_threshold:
                    start -= 1

                apex_mz = all_apex_mz[start:end + 1]
                apex_int = all_apex_int[start:end + 1]

                logger.debug(f"Kept {len(apex_mz)} contiguous apex points "
                             f"[{apex_mz[0]:.4f}, {apex_mz[-1]:.4f}] around max at "
                             f"{all_apex_mz[max_apex_idx]:.4f} (gap_threshold={gap_threshold:.3f})")

                if len(apex_mz) < 3:
                    logger.debug(f"Too few contiguous points, using all {len(all_apex_mz)} apex points")
                    apex_mz = all_apex_mz
                    apex_int = all_apex_int

                # CHECK FOR ALTERNATING INTENSITY PATTERN (same logic as charge detection)
                # At low charge states (z=2), isotope peaks are ~0.5 Da apart with deep
                # valleys; find_peaks picks up both real isotope apexes AND minor peaks
                # in the valleys, creating a high-low-high-low pattern that shifts the
                # smooth envelope centroid. Replace minor peaks' intensities with
                # interpolated values from major peaks to correct the envelope shape
                # while preserving the full m/z range for display.
                if len(apex_int) >= 4:
                    intensity_diffs = np.diff(apex_int)
                    signs = np.sign(intensity_diffs)
                    sign_changes = np.diff(signs)
                    alternation_ratio = np.sum(sign_changes != 0) / len(sign_changes) if len(sign_changes) > 0 else 0

                    if alternation_ratio > 0.8:
                        even_sum = np.sum(apex_int[0::2])
                        odd_sum = np.sum(apex_int[1::2])
                        if even_sum >= odd_sum:
                            major_idx = np.arange(0, len(apex_int), 2)
                            minor_idx = np.arange(1, len(apex_int), 2)
                        else:
                            major_idx = np.arange(1, len(apex_int), 2)
                            minor_idx = np.arange(0, len(apex_int), 2)

                        # Interpolate minor peak intensities from major peaks
                        interp_int = np.interp(apex_mz[minor_idx], apex_mz[major_idx], apex_int[major_idx])
                        apex_int = apex_int.copy()
                        apex_int[minor_idx] = interp_int
                        logger.info(f"Alternating pattern detected (ratio={alternation_ratio:.2f}): "
                                    f"interpolated {len(minor_idx)} minor peaks from {len(major_idx)} major peaks")

            # Create SMOOTH envelope by interpolating apex points + Gaussian smoothing
            # STEP 1: Interpolate apex points to create smooth curve
            # STEP 2: Apply Gaussian smoothing based on instrument resolution
            from scipy.interpolate import UnivariateSpline
            from scipy.ndimage import gaussian_filter1d

            # Create fine m/z grid
            mz_min = np.min(apex_mz)
            mz_max = np.max(apex_mz)
            num_points = int((mz_max - mz_min) / (fwhm / 100)) + 1
            mz_grid = np.linspace(mz_min, mz_max, num_points)

            # STEP 1: Interpolate apex points with cubic spline
            if len(apex_mz) >= 4:
                spline = UnivariateSpline(apex_mz, apex_int, s=0, k=3)  # cubic, no smoothing
                intensity_interp = spline(mz_grid)
                logger.debug(f"STEP 1: Cubic spline through {len(apex_mz)} apex -> {len(mz_grid)} points")
            else:
                intensity_interp = np.interp(mz_grid, apex_mz, apex_int)
                logger.debug(f"STEP 1: Linear interpolation through {len(apex_mz)} apex -> {len(mz_grid)} points")

            # STEP 2: Apply STRONGER Gaussian smoothing for better curve fitting
            mz_step = (mz_max - mz_min) / num_points if num_points > 1 else fwhm / 100
            sigma_pixels = (sigma / mz_step) * 15.0  # 15x stronger smoothing for better Gaussian fit
            intensity_grid = gaussian_filter1d(intensity_interp, sigma=sigma_pixels, mode='nearest')
            logger.debug(f"STEP 2: STRONG Gaussian smoothing (sigma={sigma:.6f} m/z x 15 = {sigma_pixels:.2f} pixels)")

            # Clip negative values (artifacts from edge smoothing)
            intensity_grid = np.maximum(intensity_grid, 0.0)

            # Normalize to 100
            if np.max(intensity_grid) > 0:
                intensity_grid = (intensity_grid / np.max(intensity_grid)) * 100.0

            logger.info(f"SUCCESS: Smooth envelope from {len(apex_mz)} apex points")
            logger.debug(f"Envelope: {len(mz_grid)} points, m/z [{np.min(mz_grid):.4f}, {np.max(mz_grid):.4f}]")

            return mz_grid, intensity_grid

        except Exception as e:
            logger.exception(f"[generate_experimental_gaussian_envelope] Exception: {str(e)}")
            return None, None

    def fit_gaussian_to_smooth_envelope(self, mz_array: Optional[npt.NDArray[np.float64]],
                                         int_array: Optional[npt.NDArray[np.float64]], resolution: int,
                                         context: str = "") -> tuple[Optional[float], Optional[float], bool]:
        """
        Fit Gaussian to pre-smoothed isotope envelope to extract X₀ (centroid) and σ.

        This is used by routes to fit experimental envelopes after they've been
        smoothed by generate_experimental_gaussian_envelope().

        Approach:
        1. Find apex points in the data
        2. Find valley boundaries (left/right) by scanning from center
        3. Fit Gaussian to ALL data points between valleys

        Args:
            mz_array: Pre-smoothed m/z values
            int_array: Pre-smoothed intensity values
            resolution: Instrument resolution (for initial sigma estimate)
            context: Optional context string for debug messages

        Returns:
            (x0, sigma, fit_succeeded): Fitted centroid and width, with success flag
            Falls back to apex values if fit fails (fit_succeeded=False)
        """
        from scipy.optimize import curve_fit
        from scipy.signal import find_peaks

        # Igor Pro-style 4-parameter Gaussian: f(x) = y0 + A × exp(-((x - x₀) / w)²)
        def gaussian(x, y0, A, x0, width):
            return y0 + A * np.exp(-((x - x0) / width)**2)

        if mz_array is None or int_array is None or len(mz_array) <= 3:
            return None, None, False

        mz_array = np.array(mz_array)
        int_array = np.array(int_array)

        try:
            # Step 1: Find apex points to determine valley boundaries
            peaks_idx, _ = find_peaks(int_array, distance=2, prominence=np.max(int_array) * 0.05)

            if len(peaks_idx) >= 5:
                mz_apex = mz_array[peaks_idx]
                int_apex = int_array[peaks_idx]

                # Find center (highest apex)
                center_idx = np.argmax(int_apex)

                # Scan left to find valley
                left_bound_idx = 0
                for i in range(center_idx - 1, 0, -1):
                    if i > 0 and int_apex[i] < int_apex[i-1]:
                        if int_apex[i] < int_apex[i+1] * 0.9:
                            left_bound_idx = i
                            break

                # Scan right to find valley
                right_bound_idx = len(int_apex) - 1
                for i in range(center_idx + 1, len(int_apex)):
                    if i < len(int_apex) - 1 and int_apex[i] < int_apex[i+1]:
                        if int_apex[i] < int_apex[i-1] * 0.9:
                            right_bound_idx = i
                            break

                # Get m/z boundaries from valleys
                left_mz = mz_apex[left_bound_idx]
                right_mz = mz_apex[right_bound_idx]

                # Extract ALL data points between valleys
                mask = (mz_array >= left_mz) & (mz_array <= right_mz)
                mz_fit = mz_array[mask]
                int_fit = int_array[mask]

                if context:
                    logger.debug(f"[{context}] Valley boundaries: [{left_mz:.4f}, {right_mz:.4f}], fitting {len(mz_fit)} points")
            else:
                # Fallback: use all data
                mz_fit = mz_array
                int_fit = int_array
                if context:
                    logger.debug(f"[{context}] Too few apexes ({len(peaks_idx)}), using all {len(mz_fit)} points")

            if len(mz_fit) < 3:
                mz_fit = mz_array
                int_fit = int_array

            # Step 2: Fit Igor-style 4-parameter Gaussian to data between valleys
            max_idx = np.argmax(int_fit)
            A_init = int_fit[max_idx]
            x0_init = mz_fit[max_idx]
            fwhm_estimate = x0_init / resolution
            sigma_init = fwhm_estimate / 2.355
            width_init = sigma_init * np.sqrt(2)
            y0_init = np.min(int_fit)

            popt, pcov = curve_fit(
                gaussian,
                mz_fit,
                int_fit,
                p0=[y0_init, A_init, x0_init, width_init],
                bounds=([-A_init * 0.1, 0, mz_fit[0], 0.001],
                        [A_init * 0.5, A_init * 2, mz_fit[-1], fwhm_estimate * 2 * np.sqrt(2)]),
                maxfev=5000
            )

            x0 = float(popt[2])
            sigma = float(abs(popt[3]) / np.sqrt(2))

            if context:
                logger.debug(f"[{context}] Gaussian fit: X0={x0:.4f} m/z, sigma={sigma:.6f} m/z")

            return x0, sigma, True

        except Exception as e:
            if context:
                logger.warning(f"[{context}] Gaussian fit failed ({str(e)}), using apex fallback")
            # Fallback: use apex of smooth envelope
            max_idx = np.argmax(int_array)
            x0 = float(mz_array[max_idx])
            fwhm = x0 / resolution
            sigma = float(fwhm / 2.355)

            return x0, sigma, False

    def detect_peak_asymmetry_visual(self, mz_array: npt.NDArray[np.float64], int_array: npt.NDArray[np.float64],
                                      threshold_ratio: float = 0.3) -> tuple[bool, int, str]:
        """
        Detect peak asymmetry using visual characteristics:
        - Count local maxima (multiple bumps = asymmetric)
        - Check for shoulders (secondary peaks)
        - Measure envelope smoothness

        Returns: (is_asymmetric, num_maxima, details)
        """
        try:
            if len(mz_array) < 5:
                return False, 1, "Too few points"

            mz_array = np.array(mz_array)
            int_array = np.array(int_array)

            # Normalize intensity
            max_int = np.max(int_array)
            if max_int == 0:
                return False, 1, "Zero intensity"

            int_norm = int_array / max_int

            # Find local maxima (peaks)
            from scipy.signal import find_peaks

            # Detect peaks with minimum height (to avoid noise)
            # Prominence helps identify significant peaks vs noise
            peaks, properties = find_peaks(
                int_norm,
                height=threshold_ratio,  # At least 30% of max height
                prominence=0.1,  # Must be prominent enough
                distance=3  # Separated by at least 3 points
            )

            num_maxima = len(peaks)

            # Determine if asymmetric based on number of significant maxima
            is_asymmetric = num_maxima > 1

            details = f"{num_maxima} local maxima detected"
            if num_maxima > 1:
                peak_positions = [f"{mz_array[p]:.2f}" for p in peaks]
                details += f" at m/z: {', '.join(peak_positions)}"

            logger.debug(f"[Visual asymmetry detection] {details} -> {'ASYMMETRIC' if is_asymmetric else 'SYMMETRIC'}")

            return is_asymmetric, num_maxima, details

        except Exception as e:
            logger.error(f"[detect_peak_asymmetry_visual] Error: {str(e)}")
            return False, 1, f"Error: {str(e)}"

    def calculate_peak_skewness(self, mz_array: npt.NDArray[np.float64],
                                 int_array: npt.NDArray[np.float64]) -> Optional[float]:
        """
        Calculate peak skewness (asymmetry measure).
        Skewness = 0: perfectly symmetric
        Skewness > 0: right-tailed (tailing to higher m/z)
        Skewness < 0: left-tailed (tailing to lower m/z)

        Returns: skewness value
        """
        try:
            mz_array = np.array(mz_array, dtype=float)
            int_array = np.array(int_array, dtype=float)

            if len(mz_array) < 3 or np.sum(int_array) == 0:
                return None

            # Calculate mean (weighted by intensity)
            mean = np.sum(mz_array * int_array) / np.sum(int_array)

            # Calculate standard deviation
            variance = np.sum(int_array * (mz_array - mean)**2) / np.sum(int_array)
            std_dev = np.sqrt(variance)

            if std_dev == 0:
                return 0.0

            # Calculate skewness: (mean - mode) / std_dev
            # Approximate mode as the m/z with maximum intensity
            mode_idx = np.argmax(int_array)
            mode = mz_array[mode_idx]

            skewness = (mean - mode) / std_dev

            return float(skewness)

        except Exception as e:
            logger.error(f"[calculate_peak_skewness] Exception: {str(e)}")
            return None

    def weighted_average_centroid(self, mz_array: npt.NDArray[np.float64],
                                   int_array: npt.NDArray[np.float64]) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate centroid using weighted average method.
        This is used for general centroid calculations.
        Returns x₀ = Σ(m/z × intensity) / Σ(intensity) and σ (weighted std dev)
        """
        try:
            if len(mz_array) == 0 or len(int_array) == 0:
                logger.warning(f"[weighted_average_centroid] Empty arrays")
                return None, None

            # Convert to numpy arrays
            mz_array = np.array(mz_array, dtype=float)
            int_array = np.array(int_array, dtype=float)

            total_intensity = np.sum(int_array)
            if total_intensity == 0 or np.isnan(total_intensity) or np.isinf(total_intensity):
                logger.warning(f"[weighted_average_centroid] Invalid total intensity: {total_intensity}")
                return None, None

            # Weighted average: x₀ = Σ(m/z × intensity) / Σ(intensity)
            x0 = np.sum(mz_array * int_array) / total_intensity

            # Weighted standard deviation: σ = sqrt(Σ(intensity × (m/z - x₀)²) / Σ(intensity))
            sigma = np.sqrt(np.sum(int_array * (mz_array - x0)**2) / total_intensity)

            if np.isnan(x0) or np.isinf(x0):
                return None, None

            logger.debug(f"[weighted_average_centroid] x0={x0:.4f}, sigma={sigma:.4f}")
            return float(x0), float(sigma)

        except Exception as e:
            logger.error(f"[weighted_average_centroid] Exception: {str(e)}")
            return None, None

    def gaussian_fit_centroid(self, mz_array: npt.NDArray[np.float64], int_array: npt.NDArray[np.float64],
                               return_quality: bool = False) -> tuple:
        """
        Fit Gaussian curve to isotope envelope: f(x) = A × exp(-(x - x₀)² / (2σ²))
        This is the STANDARD method used for composition determination (X₀ error calculation).

        Parameters:
        - return_quality: If True, also return R² goodness-of-fit metric

        Returns:
        - (x0, sigma, x0_error) if return_quality=False
        - (x0, sigma, x0_error, r_squared) if return_quality=True
        - x0_error is the standard error of the fitted x₀ parameter from covariance matrix
        """
        try:
            if len(mz_array) == 0 or len(int_array) == 0:
                logger.warning(f"[gaussian_fit_centroid] Empty arrays: mz length={len(mz_array)}, int length={len(int_array)}")
                if return_quality:
                    return None, None, None, None
                else:
                    return None, None, None

            # Convert to numpy arrays
            mz_array = np.array(mz_array, dtype=float)
            int_array = np.array(int_array, dtype=float)

            if len(mz_array) < 3:
                logger.warning(f"[gaussian_fit_centroid] Need at least 3 points for fitting")
                if return_quality:
                    return None, None, None, None
                else:
                    return None, None, None

            total_intensity = np.sum(int_array)
            if total_intensity == 0 or np.isnan(total_intensity) or np.isinf(total_intensity):
                logger.warning(f"[gaussian_fit_centroid] Invalid total intensity: {total_intensity}")
                if return_quality:
                    return None, None, None, None
                else:
                    return None, None, None

            # NEW APPROACH: Find apex envelope, then find valleys in envelope, then fit
            # Step 1: Find apex points of individual isotope peaks
            from scipy.signal import find_peaks

            # Find local maxima (apex of each isotope peak)
            peaks_idx, _ = find_peaks(int_array, distance=2, prominence=np.max(int_array) * 0.05)

            if len(peaks_idx) >= 5:
                # Extract apex points (envelope)
                mz_apex = mz_array[peaks_idx]
                int_apex = int_array[peaks_idx]
                logger.debug(f"[gaussian_fit_centroid] Found {len(peaks_idx)} isotope peak apexes")

                # Step 2: Find the highest apex (center of envelope)
                center_idx = np.argmax(int_apex)

                # Step 3: Find valleys in the envelope (left and right of center)
                # A valley must be both a local dip (relative criterion) AND
                # below 40% of center intensity (absolute criterion) to avoid
                # triggering on noise fluctuations in broad complex envelopes.
                center_intensity = int_apex[center_idx]
                valley_threshold = center_intensity * 0.4

                # Scan left from center to find minimum
                left_bound_idx = 0
                for i in range(center_idx - 1, 0, -1):
                    if i > 0 and int_apex[i] < int_apex[i-1]:
                        if int_apex[i] < int_apex[i+1] * 0.9 and int_apex[i] < valley_threshold:
                            left_bound_idx = i
                            break

                # Scan right from center to find minimum
                right_bound_idx = len(int_apex) - 1
                for i in range(center_idx + 1, len(int_apex)):
                    if i < len(int_apex) - 1 and int_apex[i] < int_apex[i+1]:
                        if int_apex[i] < int_apex[i-1] * 0.9 and int_apex[i] < valley_threshold:
                            right_bound_idx = i
                            break

                # Step 4: Extract apex points BETWEEN envelope valleys
                mz_fit = mz_apex[left_bound_idx:right_bound_idx+1]
                int_fit = int_apex[left_bound_idx:right_bound_idx+1]

                logger.debug(f"[gaussian_fit_centroid] Envelope valleys: left={left_bound_idx}, center={center_idx}, right={right_bound_idx}")
                logger.debug(f"[gaussian_fit_centroid] Fitting to {len(mz_fit)} apex points between envelope valleys")
            else:
                # Fallback: if too few peaks, use all apex points or top 70%
                if len(peaks_idx) >= 3:
                    mz_fit = mz_array[peaks_idx]
                    int_fit = int_array[peaks_idx]
                    logger.warning(f"[gaussian_fit_centroid] Only {len(peaks_idx)} apexes, using all")
                else:
                    logger.warning(f"[gaussian_fit_centroid] Too few apexes, using top 70%")
                    max_intensity = np.max(int_array)
                    threshold = max_intensity * 0.70
                    high_intensity_mask = int_array >= threshold
                    mz_fit = mz_array[high_intensity_mask]
                    int_fit = int_array[high_intensity_mask]

            if len(mz_fit) < 3:
                logger.error(f"[gaussian_fit_centroid] Too few points, using all data")
                mz_fit = mz_array
                int_fit = int_array

            # Initial guesses for Gaussian parameters
            # Amplitude: maximum intensity
            A_guess = np.max(int_fit)

            # Center: m/z of the maximum intensity point (apex)
            max_idx = np.argmax(int_fit)
            x0_guess = mz_fit[max_idx]

            # Width: estimate from data range
            mz_min_fit = np.min(mz_fit)
            mz_max_fit = np.max(mz_fit)
            mz_range_fit = mz_max_fit - mz_min_fit
            sigma_guess = mz_range_fit / 4.0  # Narrower estimate since we're fitting top only

            # Ensure reasonable initial guesses
            if sigma_guess < 0.01:
                sigma_guess = 0.5

            # Overall data range for bounds
            mz_min_all = np.min(mz_array)
            mz_max_all = np.max(mz_array)

            # Baseline guess: minimum intensity in fitting region
            max_int_fit = np.max(int_fit)
            y0_guess = min(np.min(int_fit), max_int_fit * 0.4)

            logger.debug(f"[gaussian_fit_centroid] Initial guesses: x0={x0_guess:.4f} (apex), sigma={sigma_guess:.4f}, A={A_guess:.2e}, y0={y0_guess:.2e}")

            # Igor Pro-style 4-parameter Gaussian: f(x) = y0 + A × exp(-((x - x₀) / w)²)
            # where w = sqrt(2) × σ, so exponent = -(x - x₀)² / (2σ²)
            def gaussian(x, y0, A, x0, width):
                return y0 + A * np.exp(-((x - x0) / width)**2)

            # Fit Gaussian curve to HIGH-INTENSITY data only
            try:
                from scipy.optimize import curve_fit

                width_guess = sigma_guess * np.sqrt(2)

                # Allow x0 to vary within the full valley boundaries
                bounds = (
                    [-max_int_fit * 0.1, 0, mz_min_all, 0.01],  # Lower bounds: [y0_min, A_min, x0_min, width_min]
                    [max_int_fit * 0.5, np.inf, mz_max_all, (mz_max_all - mz_min_all) * 2]  # Upper bounds
                )

                logger.debug(f"[gaussian_fit_centroid] x0 bounds: [{mz_min_all:.4f}, {mz_max_all:.4f}]")

                popt, pcov = curve_fit(
                    gaussian,
                    mz_fit,  # Fit to high-intensity points only
                    int_fit,
                    p0=[y0_guess, A_guess, x0_guess, width_guess],
                    bounds=bounds,
                    maxfev=10000,
                    ftol=1e-10,  # Function tolerance for convergence (more precise)
                    xtol=1e-10   # Parameter tolerance for convergence (more precise)
                )

                y0_fit, A_fit, x0_fit, width_fit = popt
                sigma_fit = width_fit / np.sqrt(2)

                # Calculate standard errors from covariance matrix
                # pcov diagonal gives variance of parameters, sqrt gives standard error
                perr = np.sqrt(np.diag(pcov))
                y0_err, A_err, x0_err, width_err = perr

                # Validate fitted parameters
                if np.isnan(x0_fit) or np.isinf(x0_fit) or np.isnan(sigma_fit) or np.isinf(sigma_fit):
                    raise ValueError("Fit returned invalid parameters")

                # Calculate R² (coefficient of determination) if requested
                if return_quality:
                    # Predicted values from fitted Gaussian
                    y_pred = gaussian(mz_array, y0_fit, A_fit, x0_fit, width_fit)

                    # Calculate R²
                    ss_res = np.sum((int_array - y_pred) ** 2)  # Residual sum of squares
                    ss_tot = np.sum((int_array - np.mean(int_array)) ** 2)  # Total sum of squares
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                    logger.debug(f"[gaussian_fit_centroid] Gaussian fit: x₀={x0_fit:.4f}±{x0_err:.4f}, σ={sigma_fit:.4f}, y0={y0_fit:.2e}, R²={r_squared:.4f}")
                    return float(x0_fit), float(sigma_fit), float(x0_err), float(r_squared)
                else:
                    logger.debug(f"[gaussian_fit_centroid] Gaussian fit: x₀={x0_fit:.4f}±{x0_err:.4f}, σ={sigma_fit:.4f}, y0={y0_fit:.2e}")
                    return float(x0_fit), float(sigma_fit), float(x0_err)

            except Exception as fit_error:
                # If Gaussian fit fails, fall back to weighted average
                logger.warning(f"[gaussian_fit_centroid] Gaussian fit failed ({fit_error}), using weighted average fallback")
                x0_fallback = x0_guess
                sigma_fallback = sigma_guess

                if np.isnan(x0_fallback) or np.isinf(x0_fallback):
                    if return_quality:
                        return None, None, None, None
                    else:
                        return None, None, None

                if return_quality:
                    return float(x0_fallback), float(sigma_fallback), None, 0.0  # No fitting error, R² = 0 indicates fit failed
                else:
                    return float(x0_fallback), float(sigma_fallback), None  # No fitting error available

        except Exception as e:
            logger.exception(f"[gaussian_fit_centroid] Exception: {str(e)}")
            if return_quality:
                return None, None, None, None
            else:
                return None, None, None

    def match_isotope_pattern(self, experimental_mz: npt.NDArray[np.float64], experimental_int: npt.NDArray[np.float64],
                               theoretical_pattern: dict, tolerance: float = 0.5) -> float:
        """
        Match experimental peaks to theoretical isotope pattern using Gaussian fitting
        Compares X0 (centroid) positions between theory and experiment
        Returns the X0 centroid difference in m/z units (error metric)
        """
        if 'error' in theoretical_pattern:
            return 999.0  # Large error if pattern generation failed

        # Use smooth Gaussian pattern for theo_x0 calculation (same method as exp_x0)
        theo_mz = np.array(theoretical_pattern.get('gaussian_mz', theoretical_pattern.get('mz', [])))
        theo_int = np.array(theoretical_pattern.get('gaussian_intensity', theoretical_pattern.get('intensity', [])))

        if len(theo_mz) == 0:
            return 999.0

        # Normalize both patterns
        theo_int_norm = theo_int / np.max(theo_int) * 100
        exp_int_norm = experimental_int / np.max(experimental_int) * 100

        # Calculate Gaussian centroids (X0) for both patterns
        theo_fit_result = self.gaussian_fit_centroid(theo_mz, theo_int_norm)
        exp_fit_result = self.gaussian_fit_centroid(experimental_mz, exp_int_norm)

        theo_x0 = theo_fit_result[0] if theo_fit_result else None
        exp_x0 = exp_fit_result[0] if exp_fit_result else None

        if theo_x0 is None or exp_x0 is None:
            return 999.0

        # Find experimental peak closest to each theoretical peak
        matched_intensities = []
        matched_masses = []

        for t_mz, t_int in zip(theo_mz, theo_int_norm):
            # Find experimental peaks within tolerance
            close_peaks = np.where(np.abs(experimental_mz - t_mz) < tolerance)[0]

            if len(close_peaks) > 0:
                # Find the closest peak
                closest_idx = close_peaks[np.argmin(np.abs(experimental_mz[close_peaks] - t_mz))]
                matched_intensities.append((t_int, exp_int_norm[closest_idx]))
                matched_masses.append((t_mz, experimental_mz[closest_idx]))

        if len(matched_intensities) == 0:
            return 999.0

        # Return X0 centroid difference in m/z units
        # This is the ERROR metric - smaller is better
        # Different Qcl values shift the centroid position
        # The Qcl with smallest X0 error is the correct one

        x0_error = abs(exp_x0 - theo_x0)

        return x0_error

    def detect_charge_state(self, mz_values: npt.NDArray[np.float64], intensity_values: npt.NDArray[np.float64],
                                 target_mz: float, window: float = 3.0) -> dict:
        """
        Detect charge state by anchored isotope-grid scoring.

        For each candidate z in [1..10], build the expected isotope grid
        target_mz + k * (1.003/z) and score how well the experimental peaks fit:
            score = (fraction of strong-peak intensity on the grid)
                  - (fraction of grid positions with no peak nearby)
        Best z = highest score among viable candidates. If best is even and the
        grid-intensity pattern alternates high-low (Ag doublet for DNA-AgN
        clusters), halve z — this distinguishes a true z=N envelope from a
        z=N/2 envelope where 107Ag/109Ag doubles the apparent peak count.

        Returns dict compatible with previous callers:
            'spacing', 'charge', 'confidence', 'num_peaks', 'scores'
        """
        from scipy.signal import find_peaks

        NEUTRON_MASS = 1.003
        CHARGE_RANGE = (1, 10)
        DOUBLET_ALT_THRESHOLD = 0.85

        mask = (mz_values >= target_mz - window) & (mz_values <= target_mz + window)
        region_mz = mz_values[mask]
        region_int = intensity_values[mask]

        if len(region_mz) < 5:
            return {'spacing': None, 'charge': None, 'confidence': 0.0, 'num_peaks': 0, 'scores': {}}

        max_intensity = float(np.max(region_int))
        peaks_idx, _ = find_peaks(region_int, prominence=max_intensity * 0.03, distance=1)

        if len(peaks_idx) < 2:
            return {'spacing': None, 'charge': None, 'confidence': 0.0, 'num_peaks': int(len(peaks_idx)), 'scores': {}}

        peak_mzs = region_mz[peaks_idx]
        peak_ints = region_int[peaks_idx]

        strong_mask = peak_ints >= max_intensity * 0.10
        strong_mzs = peak_mzs[strong_mask]
        strong_ints = peak_ints[strong_mask]
        total_strong_int = float(np.sum(strong_ints)) if len(strong_ints) > 0 else 1.0

        results: dict[int, dict[str, float]] = {}
        for z in range(CHARGE_RANGE[0], CHARGE_RANGE[1] + 1):
            spacing = NEUTRON_MASS / z
            tol = spacing * 0.25
            n_iso = max(1, int(window / spacing))
            ks = np.arange(-n_iso, n_iso + 1)
            grid = target_mz + ks * spacing

            grid_ints = np.zeros(len(grid))
            for i, g in enumerate(grid):
                d = np.abs(peak_mzs - g)
                j = int(np.argmin(d))
                if d[j] <= tol:
                    grid_ints[i] = peak_ints[j]

            matched_int = 0.0
            for smz, sint in zip(strong_mzs, strong_ints):
                if np.min(np.abs(grid - smz)) <= tol:
                    matched_int += float(sint)
            coverage = matched_int / total_strong_int
            gap_frac = float(np.sum(grid_ints == 0)) / len(grid)

            even_vals = grid_ints[(ks % 2 == 0) & (grid_ints > 0)]
            odd_vals = grid_ints[(ks % 2 == 1) & (grid_ints > 0)]
            if len(even_vals) > 0 and len(odd_vals) > 0:
                em, om = float(even_vals.mean()), float(odd_vals.mean())
                alt = (min(em, om) / max(em, om)) if max(em, om) > 0 else 1.0
            else:
                alt = 1.0

            left = sum(1 for k in range(1, n_iso + 1)
                       if np.min(np.abs(peak_mzs - (target_mz - k * spacing))) <= tol)
            right = sum(1 for k in range(1, n_iso + 1)
                        if np.min(np.abs(peak_mzs - (target_mz + k * spacing))) <= tol)

            results[z] = {
                'score': float(coverage - gap_frac),
                'coverage': float(coverage),
                'gap_frac': float(gap_frac),
                'alt': float(alt),
                'left': int(left),
                'right': int(right),
                'viable': bool(left + right + 1 >= 5),
                'spacing': float(spacing),
            }

        viable_zs = [z for z, r in results.items() if r['viable']]
        if viable_zs:
            best_z = max(viable_zs, key=lambda z: results[z]['score'])
        else:
            best_z = max(results.keys(), key=lambda z: results[z]['score'])

        while best_z > 1 and best_z % 2 == 0:
            half = best_z // 2
            if (half in results and results[half]['viable']
                    and results[best_z]['alt'] < DOUBLET_ALT_THRESHOLD):
                logger.info(f"[detect_charge_state] Ag-doublet halving at m/z {target_mz:.4f}: "
                            f"z={best_z} -> z={half} (alt={results[best_z]['alt']:.2f})")
                best_z = half
            else:
                break

        best = results[best_z]
        confidence = max(0.0, min(1.0, best['score']))
        num_matched = best['left'] + best['right'] + 1

        logger.debug(f"[detect_charge_state] target={target_mz:.4f} -> z={best_z} "
                     f"(coverage={best['coverage']:.2f}, gap={best['gap_frac']:.2f}, "
                     f"alt={best['alt']:.2f}, score={best['score']:+.3f})")

        return {
            'spacing': float(best['spacing']),
            'charge': int(best_z),
            'confidence': float(confidence),
            'num_peaks': int(num_matched),
            'scores': results,
        }

    def detect_charge_for_clicked_peak(self, mz_values: npt.NDArray[np.float64], intensity_values: npt.NDArray[np.float64],
                                        target_mz: float, charge_range: tuple[int, int] = (1, 10)) -> dict:
        """
        Determine charge state of a user-clicked peak.

        Primary: isotope-grid scoring via detect_charge_state.
        Fallback: Senko charge assignment on the surrounding envelope.
        Returns dict with 'charge', 'confidence', 'method' (and 'spacing',
        'num_peaks' when produced by the primary method). 'charge' is None
        only when both methods fail.
        """
        logger.info(f"[Charge Detection] Analyzing peak at m/z {target_mz:.4f}")

        result = self.detect_charge_state(mz_values, intensity_values, target_mz, window=3.0)
        charge = result.get('charge')
        if charge is not None and charge_range[0] <= charge <= charge_range[1]:
            num_peaks = int(result.get('num_peaks', 0))
            confidence = float(result.get('confidence', 0.0))
            if num_peaks < 3:
                confidence = min(0.6, confidence)
            logger.info(f"[Charge Detection] z={charge} via grid (conf={confidence*100:.0f}%, {num_peaks} matched)")
            return {
                'charge': int(charge),
                'confidence': float(confidence),
                'method': 'spacing',
                'spacing': float(result['spacing']),
                'num_peaks': num_peaks,
            }

        logger.debug("[Charge Detection] Grid method inconclusive, falling back to Senko")
        try:
            from pythoms.senko_charge_assignment import detect_all_peaks_with_charge
            detected = detect_all_peaks_with_charge(
                mz_values, intensity_values,
                prominence=0.01, charge_range=charge_range,
                method='combination', merge_gap=1.5,
            )
            closest = min(
                (p for p in detected if abs(p['mz'] - target_mz) < 5.0 and p.get('charge') is not None),
                key=lambda p: abs(p['mz'] - target_mz),
                default=None,
            )
            if closest is not None:
                logger.info(f"[Charge Detection] z={closest['charge']} via Senko fallback")
                return {
                    'charge': int(closest['charge']),
                    'confidence': float(closest['confidence']) * 0.8,
                    'method': 'senko_fallback',
                }
        except Exception as e:
            logger.error(f"[Charge Detection] Senko fallback error: {e}")

        logger.warning("[Charge Detection] All methods failed; user input required")
        return {'charge': None, 'confidence': 0.0, 'method': 'user_input_required'}

    def find_composition_in_spectrum(self, formula: str, charge: int, qcl: int,
                                      mz_values: npt.NDArray[np.float64], intensity_values: npt.NDArray[np.float64],
                                      peaks_data: dict, resolution: int = 20000,
                                      adduct_mass: float = 0.0, adduct_charge: int = 0) -> dict:
        """
        Search for a specific composition in the experimental spectrum
        Returns match information including score, location, and mass error
        """
        try:
            logger.debug(f"[find_composition_in_spectrum] About to parse formula='{formula}', charge={charge}, qcl={qcl}")
            logger.debug(f"[find_composition_in_spectrum] Formula type: {type(formula)}, length: {len(formula)}")
            logger.debug(f"[find_composition_in_spectrum] Formula repr: {repr(formula)}")
            logger.debug(f"[find_composition_in_spectrum] Last 5 chars: {repr(formula[-5:])}")

            # Generate theoretical isotope pattern using cached function
            logger.debug(f"[find_composition_in_spectrum] Generating isotope pattern for formula '{formula}'")
            try:
                pattern_result = self.generate_isotope_pattern(formula, charge=charge, resolution=resolution)
                if 'error' in pattern_result:
                    return {'found': False, 'error': pattern_result['error']}
                logger.debug(f"[find_composition_in_spectrum] Successfully generated isotope pattern")
            except Exception as e:
                logger.exception(f"[find_composition_in_spectrum] Error generating isotope pattern: {type(e).__name__}: {str(e)}")
                raise

            # Get monoisotopic mass from pattern result
            theoretical_mass = pattern_result.get('monoisotopic_mass')
            if theoretical_mass is None:
                return {'found': False, 'error': 'Could not get monoisotopic mass'}
            logger.debug(f"[find_composition_in_spectrum] Monoisotopic mass = {theoretical_mass}")

            # Use the same formula as in composition analysis:
            # mass = neutral_mass + adduct_mass - (qcl + z + adduct_charge) * proton_mass
            # m/z = mass / z
            # The neutral formula loses (qcl + z + adduct_charge) protons
            mass = theoretical_mass + adduct_mass - (qcl + charge + adduct_charge) * self.m_p
            theoretical_mz = mass / charge
            protons_removed = qcl + charge + adduct_charge
            logger.debug(f"[find_composition_in_spectrum] Theoretical m/z = {theoretical_mz} (neutral mass + {adduct_mass:.4f} adduct - {protons_removed} protons) / {charge}")

            # Get bar isotope pattern from cached result
            barip = (pattern_result.get('mz', []), pattern_result.get('intensity', []))
            logger.debug(f"[find_composition_in_spectrum] Bar isotope pattern has {len(barip[0]) if barip else 0} peaks")

            if not barip[0] or len(barip[0]) == 0:
                return {'found': False, 'error': 'Could not generate isotope pattern'}

            # Use Gaussian pattern from cached result
            theoretical_pattern = (pattern_result.get('gaussian_mz', []), pattern_result.get('gaussian_intensity', []))
            logger.debug(f"[find_composition_in_spectrum] Gaussian pattern has {len(theoretical_pattern[0]) if theoretical_pattern else 0} peaks")

            if theoretical_pattern is None or len(theoretical_pattern[0]) == 0:
                return {'found': False, 'error': 'Could not generate Gaussian isotope pattern'}

            # Search through detected peaks for best match
            logger.debug(f"[find_composition_in_spectrum] Searching through {len(peaks_data)} peaks...")
            best_match = None
            best_x0_error = float('inf')
            mz_tolerance = 5.0  # m/z units

            # Track nearest peak even if outside tolerance
            nearest_peak_mz = None
            nearest_peak_error_mz = float('inf')
            nearest_peak_error_ppm = None
            nearest_peak_intensity = None

            for peak_idx, peak in enumerate(peaks_data):
                if peak_idx == 0:
                    logger.debug(f"[find_composition_in_spectrum] First peak keys: {peak.keys()}")
                    logger.debug(f"[find_composition_in_spectrum] First peak data: {peak}")

                # Use 'peak_mz' if 'mz' doesn't exist (check what key name is used)
                peak_mz = peak.get('peak_mz') or peak.get('mz')
                if peak_mz is None:
                    logger.error(f"[find_composition_in_spectrum] Peak {peak_idx} has no mz/peak_mz key! Keys: {peak.keys()}")
                    continue
                mass_error_ppm = abs((peak_mz - theoretical_mz) / theoretical_mz * 1e6)
                mass_error_mz = abs(peak_mz - theoretical_mz)

                # Track nearest peak for error reporting
                if mass_error_mz < nearest_peak_error_mz:
                    nearest_peak_mz = float(peak_mz)
                    nearest_peak_error_mz = float(mass_error_mz)
                    nearest_peak_error_ppm = float(mass_error_ppm)
                    nearest_peak_intensity = float(peak.get('intensity', 0))

                # Only consider peaks within reasonable m/z tolerance (5 m/z units)
                # This is generous to allow X0 matching to determine best fit
                if mass_error_mz > mz_tolerance:
                    continue

                # Extract experimental data around this peak
                window = 3.0  # m/z window
                mask = (mz_values >= peak_mz - window) & (mz_values <= peak_mz + window)
                exp_mz = mz_values[mask]
                exp_int = intensity_values[mask]

                if len(exp_mz) < 5:
                    continue

                # Normalize experimental intensities
                if np.max(exp_int) > 0:
                    exp_int_norm = exp_int / np.max(exp_int) * 100
                else:
                    continue

                # Calculate X0 error (now returns error in m/z units)
                x0_error = self.match_isotope_pattern(
                    theoretical_pattern[0], theoretical_pattern[1],
                    exp_mz, exp_int_norm,
                    theoretical_mz
                )

                if x0_error < best_x0_error:
                    best_x0_error = x0_error
                    best_match = {
                        'found': True,
                        'peak_mz': float(peak_mz),
                        'theoretical_mz': float(theoretical_mz),
                        'mass_error_ppm': float(mass_error_ppm),
                        'x0_error': float(x0_error),
                        'charge': charge,
                        'peak_charge': peak.get('charge', 'N/A'),
                        'peak_intensity': float(peak['intensity'])
                    }

            if best_match is None:
                # No matching peak found within tolerance
                # Return detailed info about nearest peak for user feedback
                result = {
                    'found': False,
                    'theoretical_mz': float(theoretical_mz),
                    'charge': charge,
                    'threshold_mz': float(mz_tolerance),
                    'reason': 'no_peaks_in_range' if nearest_peak_mz is None else 'mass_error_exceeds_threshold'
                }

                if nearest_peak_mz is not None:
                    result['nearest_peak_mz'] = nearest_peak_mz
                    result['mass_error_mz'] = nearest_peak_error_mz
                    result['mass_error_ppm'] = nearest_peak_error_ppm
                    result['nearest_peak_intensity'] = nearest_peak_intensity
                    result['message'] = (
                        f'Theoretical m/z {theoretical_mz:.4f} is {nearest_peak_error_mz:.2f} m/z '
                        f'({nearest_peak_error_ppm:.0f} ppm) away from nearest peak ({nearest_peak_mz:.4f}). '
                        f'Exceeds {mz_tolerance:.1f} m/z tolerance.'
                    )
                    logger.info(f"[find_composition_in_spectrum] No match: nearest peak at {nearest_peak_mz:.4f} is {nearest_peak_error_mz:.2f} m/z away (threshold: {mz_tolerance})")
                else:
                    result['message'] = f'No peaks detected in spectrum to compare with theoretical m/z {theoretical_mz:.4f}'
                    logger.info(f"[find_composition_in_spectrum] No match: no peaks in spectrum")

                return result

            logger.debug(f"[find_composition_in_spectrum] Search complete. Best match found: {best_match is not None}")
            return best_match

        except Exception as e:
            logger.exception(f"[find_composition_in_spectrum] Error: {type(e).__name__}: {str(e)}")
            return {'found': False, 'error': str(e)}

