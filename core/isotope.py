from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'lib'))

from pythoms.molecule import IPMolecule

try:
    import IsoSpecPy as isospec

    ISOSPEC_AVAILABLE = True
except ImportError:
    ISOSPEC_AVAILABLE = False

ISOTOPE_LIBRARY = 'isospec' if ISOSPEC_AVAILABLE else 'pythoms'

_isotope_pattern_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
_ISOTOPE_CACHE_MAX_SIZE = 1000

logger = logging.getLogger(__name__)


class IsotopeMixin:
    """Mixin for isotope pattern generation (IsoSpecPy and PythoMS backends)."""

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
            logger.debug(f'[generate_isotope_pattern] CACHE HIT for {formula[:30]}... (z={charge})')
            return _isotope_pattern_cache[cache_key]

        logger.debug(f'[generate_isotope_pattern] CACHE MISS for {formula[:30]}... (z={charge}) - computing...')
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
                result.append(f'{elem}{count}' if count > 1 else elem)

        # Add remaining elements alphabetically
        for elem in sorted(element_counts.keys()):
            count = element_counts[elem]
            result.append(f'{elem}{count}' if count > 1 else elem)

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
            bins = np.arange(min_mz - bin_width / 2, max_mz + bin_width, bin_width)

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
            logger.warning(f'IsoSpecPy failed for {formula}: {e}, falling back to PythoMS')
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
                threshold=0.01,
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
