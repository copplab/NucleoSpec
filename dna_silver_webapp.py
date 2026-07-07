"""
DNA-stabilized Silver Nanocluster Mass Spectrometry Analysis Web Application

This Flask-based web application analyzes mass spectrometry data for DNA-silver nanoclusters,
providing isotope pattern overlay, charge state identification, and composition analysis.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Callable, TypeVar

import numpy as np
import numpy.typing as npt
from flask import Flask, Response, jsonify, render_template, request, send_file

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Type aliases (guarded for Python < 3.9 / older NumPy)
try:
    NDArrayFloat = npt.NDArray[np.floating[Any]]
    NDArrayInt = npt.NDArray[np.integer[Any]]
except TypeError:
    NDArrayFloat = np.ndarray  # type: ignore[misc]
    NDArrayInt = np.ndarray  # type: ignore[misc]
F = TypeVar('F', bound=Callable[..., Any])
# Flask response type (can be Response, tuple of Response/dict and status code, or str)
try:
    FlaskResponse = Response | tuple[Response, int] | str
except TypeError:
    FlaskResponse = Any  # type: ignore[misc]

# Add lib directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'lib'))
from core.analyzer import MAX_SILVER, MAX_STRANDS, DNASilverAnalyzer

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check IsoSpecPy availability for faster isotope pattern generation
import importlib.util

from pythoms.molecule import composition_from_formula
from pythoms.senko_charge_assignment import detect_all_peaks_with_charge

ISOSPEC_AVAILABLE = importlib.util.find_spec('IsoSpecPy') is not None
if ISOSPEC_AVAILABLE:
    logger.info('IsoSpecPy available - faster isotope pattern generation enabled')
else:
    logger.warning('IsoSpecPy not installed - using PythoMS (pip install IsoSpecPy for faster performance)')

# Toggle for isotope pattern library: 'isospec' (faster) or 'pythoms' (original)
# Set to 'isospec' if available, otherwise fall back to 'pythoms'
ISOTOPE_LIBRARY = 'isospec' if ISOSPEC_AVAILABLE else 'pythoms'

DEFAULT_RESOLUTION = 20000
PEAK_WINDOW = 3.0


def parse_adduct_items(adducts_input: list[dict], adduct_library: dict) -> tuple[float, int, str, dict[str, int]]:
    """Parse adduct input list and return total mass, charge, display string, and element composition."""
    total_adduct_mass = 0.0
    total_adduct_charge = 0
    adduct_formula_parts: list[str] = []
    adduct_elements: dict[str, int] = {}

    for adduct_entry in adducts_input:
        adduct_name = adduct_entry.get('name')
        adduct_count = int(adduct_entry.get('count', 1))

        inline_mass = adduct_entry.get('mass')
        inline_charge = adduct_entry.get('charge')
        if inline_mass is not None and inline_charge is not None:
            adduct_mass = float(inline_mass)
            adduct_charge = int(inline_charge)
        elif adduct_name in adduct_library:
            adduct_mass, adduct_charge = adduct_library[adduct_name]
        else:
            logger.warning(f"Adduct '{adduct_name}' not found in library, skipping")
            continue

        total_adduct_mass += adduct_mass * adduct_count
        total_adduct_charge += adduct_charge * adduct_count

        if adduct_count == 1:
            adduct_formula_parts.append(adduct_name)
        else:
            adduct_formula_parts.append(f'{adduct_count}{adduct_name}')

        inline_formula = adduct_entry.get('formula')
        try:
            if inline_formula:
                adduct_comp = composition_from_formula(inline_formula)
                total_multiplier = adduct_count
            else:
                base_match = re.match(r'^(\d+)?(.+)$', adduct_name)
                if base_match and base_match.group(1):
                    inherent_count = int(base_match.group(1))
                    base_adduct = base_match.group(2)
                else:
                    inherent_count = 1
                    base_adduct = adduct_name
                adduct_comp = composition_from_formula(base_adduct)
                total_multiplier = inherent_count * adduct_count
            for element, count in adduct_comp.items():
                adduct_elements[element] = adduct_elements.get(element, 0) + (count * total_multiplier)
        except Exception as e:
            logger.warning(f"Could not parse adduct '{adduct_name}': {e}")

        logger.debug(
            f'Adduct: {adduct_count}×{adduct_name}: mass={adduct_mass * adduct_count:.4f} Da, charge={adduct_charge * adduct_count:+d}'
        )

    adduct_string = '+'.join(adduct_formula_parts) if adduct_formula_parts else ''
    return total_adduct_mass, total_adduct_charge, adduct_string, adduct_elements


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    Handles nested dictionaries, lists, and arrays.
    Also converts Infinity and NaN to None for valid JSON.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        # Convert Infinity and NaN to None (null in JSON)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, (float, int)):
        # Handle native Python float/int that might be Infinity or NaN
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def to_subscript(n: int | str) -> str:
    """Convert number to subscript format. E.g., 1 → ₁, 28 → ₂₈"""
    subscript_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    return str(n).translate(subscript_map)


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# SECRET_KEY: Use environment variable in production
# In development, use a random key; in production, require explicit setting
_secret_key = os.environ.get('SECRET_KEY')
if _secret_key:
    app.config['SECRET_KEY'] = _secret_key
else:
    # Development only - generate random key (will change on restart)
    import secrets

    app.config['SECRET_KEY'] = secrets.token_hex(32)
    if os.environ.get('FLASK_ENV') == 'production':
        logger.warning('SECRET_KEY not set in production! Set SECRET_KEY environment variable.')

# Session cookie security settings
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'  # HTTPS only in production
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection

# Simple rate limiting for analysis endpoints (prevents abuse)
# Stores: {ip_address: [timestamp1, timestamp2, ...]}
_rate_limit_requests: dict[str, list[float]] = {}
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX_REQUESTS = 30  # max analysis requests per window per IP


def check_rate_limit(ip_address: str | None) -> bool:
    """Check if IP has exceeded rate limit. Returns True if allowed, False if blocked."""
    import time

    now = time.time()

    if ip_address is None:
        return True

    # Clean old entries
    if ip_address in _rate_limit_requests:
        _rate_limit_requests[ip_address] = [t for t in _rate_limit_requests[ip_address] if now - t < _RATE_LIMIT_WINDOW]
    else:
        _rate_limit_requests[ip_address] = []

    # Check limit
    if len(_rate_limit_requests[ip_address]) >= _RATE_LIMIT_MAX_REQUESTS:
        return False

    # Record this request
    _rate_limit_requests[ip_address].append(now)
    return True


# Input validation functions
import html
import re


def validate_dna_sequence(sequence: str | None) -> tuple[bool, str | None]:
    """Validate DNA sequence contains only valid bases (ATCG)"""
    if not sequence:
        return False, 'Sequence cannot be empty'
    if not re.match(r'^[ATCG]+$', sequence.upper()):
        return False, 'Sequence must contain only A, T, C, G bases'
    if len(sequence) > 1000:
        return False, 'Sequence too long (max 1000 bases)'
    return True, None


def validate_chemical_formula(formula: str | None) -> tuple[bool, str | None]:
    """Validate chemical formula format"""
    if not formula:
        return False, 'Formula cannot be empty'
    # Allow element symbols followed by optional numbers, with subscripts
    if not re.match(r'^[A-Za-z0-9₀₁₂₃₄₅₆₇₈₉]+$', formula):
        return False, 'Invalid formula format'
    if len(formula) > 500:
        return False, 'Formula too long'
    return True, None


def validate_element_symbol(symbol: str | None) -> tuple[bool, str | None]:
    """Validate element symbol (e.g., Ag, Na, K)"""
    if not symbol:
        return False, 'Element symbol cannot be empty'
    if not re.match(r'^[A-Z][a-z]?$', symbol):
        return False, 'Invalid element symbol format'
    return True, None


def validate_numeric_param(value: Any, min_val: float, max_val: float, name: str) -> tuple[bool, str | None]:
    """Validate numeric parameter is within range"""
    try:
        num = float(value)
        if num < min_val or num > max_val:
            return False, f'{name} must be between {min_val} and {max_val}'
        return True, None
    except (ValueError, TypeError):
        return False, f'{name} must be a number'


def sanitize_string(s: Any, max_length: int = 100) -> str:
    """Sanitize string input - escape HTML and limit length"""
    if s is None:
        return ''
    s = str(s)[:max_length]
    return html.escape(s)


# CSRF protection for JSON API endpoints
def check_same_origin(f: F) -> F:
    """Decorator to verify request comes from same origin (CSRF protection for APIs)"""
    from functools import wraps

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        # For JSON APIs, verify the request has proper content type
        # Browsers won't send application/json cross-origin without CORS preflight
        content_type = request.content_type or ''
        if request.method == 'POST' and 'application/json' not in content_type:
            return jsonify({'error': 'Invalid content type'}), 400
        return f(*args, **kwargs)

    return decorated_function  # type: ignore[return-value]


# Global cache for isotope patterns - major speed optimization
# Key: (formula, charge, resolution) -> Value: isotope pattern dict
_isotope_pattern_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
_ISOTOPE_CACHE_MAX_SIZE = 500  # Limit cache size to prevent memory issues


analyzer = DNASilverAnalyzer()


@app.route('/add_manual_composition_by_formula', methods=['POST'])
def add_manual_composition_by_formula() -> FlaskResponse:
    """Add a user-specified composition using ion formula directly"""
    try:
        data = request.get_json()
        analyzer = DNASilverAnalyzer()
        analyzer.custom_adducts = data.get('custom_adducts', []) or []
        peak_mz = float(data.get('peak_mz'))
        charge = int(data.get('charge'))
        intensity = float(data.get('intensity'))
        formula = data.get('formula', '').strip()
        dna_sequence = data.get('dna_sequence', '')
        resolution = int(data.get('resolution', DEFAULT_RESOLUTION))
        spectrum_data = data.get('spectrum')
        custom_xna = data.get('custom_xna', None)  # Get XNA settings

        logger.debug(f'Manual composition by formula - Formula: {formula}, XNA mode: {custom_xna is not None}')

        # For XNA mode, sequence is not required
        if not custom_xna and not dna_sequence:
            return jsonify({'error': 'DNA sequence is required'}), 400

        if not spectrum_data:
            return jsonify({'error': 'No spectrum data available'}), 400

        if not formula:
            return jsonify({'error': 'Ion formula is required'}), 400

        # User enters the ION formula directly (e.g., C194H245N86O112P18Ag12 or C194H237N70O120P18Ag16NH4)
        # This is what's actually observed in MS after ionization - use it as-is!
        import re

        ion_formula = formula.strip()

        # Simple element parsing just to get nAg and nP for display
        element_pattern = r'([A-Z][a-z]?)(\d*)'
        elements = re.findall(element_pattern, ion_formula)
        num_silver = 0
        num_phosphorus = 0
        for element, count in elements:
            if element == 'Ag':
                num_silver = int(count) if count else 1
            elif element == 'P':
                num_phosphorus = int(count) if count else 1

        # MANUAL FORMULA ENTRY: Don't calculate N0/Qcl or parse adducts
        # User provides complete ion formula - we just use it directly for isotope pattern
        num_strands = None
        qcl = None
        n0 = None

        # For XNA mode, calculate corrected mass for pattern shifting
        user_neutral_mass = None
        if (
            custom_xna
            and custom_xna.get('formula')
            and custom_xna.get('molecular_weight') is not None
            and num_phosphorus > 0
        ):
            # Estimate num_strands from phosphorus count
            # Each nucleotide has 1 P, so P count = sequence_length * num_strands
            # For a typical 12-base sequence: P=12 means 1 strand, P=24 means 2 strands
            from pythoms.molecule import composition_from_formula

            xna_composition = composition_from_formula(custom_xna['formula'])
            p_per_strand = xna_composition.get('P', 0)
            if p_per_strand > 0:
                # Estimate number of strands from phosphorus count
                estimated_strands_raw = num_phosphorus / p_per_strand
                estimated_strands = round(estimated_strands_raw)

                # Warn if not a clean integer (formula might not match XNA composition)
                if abs(estimated_strands_raw - estimated_strands) > 0.1:
                    logger.warning(
                        f"P count ({num_phosphorus}) doesn't divide evenly by P per strand ({p_per_strand}). Calculated: {estimated_strands_raw:.2f} strands → Rounded to: {estimated_strands}. The entered formula may not match the XNA composition!"
                    )

                # Calculate corrected mass using XNA molecular weight
                mXNA_one = custom_xna['molecular_weight']
                mDNA_total = mXNA_one * estimated_strands
                mAg_total = analyzer.mAg * num_silver

                # User neutral mass is XNA + Ag (before ionization)
                # This is compared with the theoretical mass from the formula to calculate shift
                user_neutral_mass = mDNA_total + mAg_total

                logger.info(
                    f'XNA mode - estimated {estimated_strands} strands from P count {num_phosphorus}, XNA formula: {custom_xna["formula"]} (P={p_per_strand}), Total neutral mass: {user_neutral_mass:.2f} Da'
                )
            else:
                logger.warning('XNA formula has no P atoms, cannot estimate num_strands')

        logger.debug(f'Manual formula entry - using formula as-is: {ion_formula}')

        # Calculate expected m/z using PythoMS formula parser
        # This handles any formula including adducts
        from pythoms.molecule import Molecule

        try:
            mol = Molecule(ion_formula)
            mass = mol.mass
            expected_mz = mass / charge
            mass_error_ppm = abs((expected_mz - peak_mz) / peak_mz * 1e6)
        except Exception:
            # Fallback if formula parsing fails
            expected_mz = peak_mz
            mass_error_ppm = 0.0

        # Generate isotope pattern using the ion formula (as-is from user)
        mz_values = np.array(spectrum_data['mz'])
        intensity_values = np.array(spectrum_data['intensity'])

        theo_pattern = analyzer.generate_isotope_pattern(ion_formula, charge, resolution)

        if 'error' not in theo_pattern:
            # Extract experimental data around the peak
            window = 3.0
            mask = (mz_values >= peak_mz - window) & (mz_values <= peak_mz + window)
            exp_mz_window = mz_values[mask]
            exp_int_window = intensity_values[mask]

            theo_mz = theo_pattern['gaussian_mz']
            theo_intensity = theo_pattern['gaussian_intensity']

            # Calculate theoretical X0 from smooth Gaussian pattern (same method as exp_x0)
            theo_mz_gaussian = np.array(theo_pattern['gaussian_mz'])
            theo_int_gaussian = np.array(theo_pattern['gaussian_intensity'])
            if len(theo_mz_gaussian) > 0 and np.sum(theo_int_gaussian) > 0:
                theo_fit_result = analyzer.gaussian_fit_centroid(theo_mz_gaussian, theo_int_gaussian)
                if theo_fit_result and theo_fit_result[0] is not None:
                    theo_x0 = theo_fit_result[0]
                    theo_sigma = theo_fit_result[2] if len(theo_fit_result) > 2 else None
                else:
                    # Fallback to weighted average if Gaussian fit fails
                    theo_x0 = np.sum(theo_mz_gaussian * theo_int_gaussian) / np.sum(theo_int_gaussian)
                    theo_sigma = None
            else:
                theo_x0, theo_sigma = None, None

            # Check if exp_x0 was provided from manual fit (frontend sends it)
            provided_exp_x0 = data.get('exp_x0')

            if provided_exp_x0 is not None:
                # Use the provided exp_x0 from manual fit instead of recalculating
                exp_x0 = float(provided_exp_x0)
                exp_sigma = None  # We don't recalculate sigma here
                logger.debug(
                    f'[add_manual_composition_by_formula] Using provided exp_x0 = {exp_x0:.4f} from manual fit'
                )

                # Calculate X0 error using the provided (manual fit) exp_x0
                if theo_x0 is not None:
                    x0_error = abs(theo_x0 - exp_x0)
                else:
                    x0_error = 999.0
            else:
                # Calculate experimental X0 from Gaussian envelope (automatic fit)
                if len(exp_mz_window) > 0:
                    # Generate Gaussian envelope for experimental data
                    exp_mz_gaussian, exp_int_gaussian = analyzer.generate_experimental_gaussian_envelope(
                        exp_mz_window, exp_int_window, resolution
                    )
                    if exp_mz_gaussian is not None and exp_int_gaussian is not None:
                        exp_x0, exp_sigma, _ = analyzer.gaussian_fit_centroid(exp_mz_gaussian, exp_int_gaussian)
                    else:
                        exp_x0, exp_sigma, _ = analyzer.gaussian_fit_centroid(exp_mz_window, exp_int_window)

                    # Calculate X0 error as: |theo_x0 - exp_x0|
                    if theo_x0 is not None and exp_x0 is not None:
                        x0_error = abs(theo_x0 - exp_x0)
                    else:
                        x0_error = 999.0
                else:
                    x0_error = 999.0
                    exp_x0, exp_sigma = None, None
        else:
            x0_error = 999.0
            theo_mz = []
            theo_intensity = []
            theo_x0, theo_sigma = None, None
            exp_x0, exp_sigma = None, None

        # For manual formula entry: use the ion formula as-is (user-provided)
        # Don't try to reconstruct neutral formula since we're not calculating N0/Qcl
        display_formula = ion_formula
        logger.debug(f'Using ion formula for display: {display_formula}')

        # Build composition object - simplified for manual entry
        # Since N0/Qcl are always null, just show basic info
        full_notation = f'{display_formula} (z={charge})'

        # Default type based on silver content
        comp_type = 'nanocluster' if num_silver >= 2 else 'dna_ag_ion'

        composition = {
            'type': comp_type,
            'num_strands': None,  # Not calculated for manual formulas
            'num_silver': num_silver,
            'qcl': None,  # Not calculated for manual formulas
            'n0': None,  # Not calculated for manual formulas
            'z': charge,
            'formula': display_formula,  # Display ion formula as-is
            'ion_formula': display_formula,  # Same - user provided ion formula
            'neutral_formula': None,  # Not calculated for manual formulas
            'adduct': '',  # No separate adduct notation
            'full_notation': full_notation,
            'expected_mz': expected_mz,
            'mass_error_ppm': mass_error_ppm,
            'x0_error': x0_error,
            'abs_x0_error': abs(x0_error) if x0_error is not None else 999.0,
            'theo_mz': theo_mz,
            'theo_intensity': theo_intensity,
            'theo_x0': float(theo_x0) if theo_x0 is not None else None,
            'theo_sigma': float(theo_sigma) if theo_sigma is not None else None,
            'exp_x0': float(exp_x0) if exp_x0 is not None else None,
            'exp_sigma': float(exp_sigma) if exp_sigma is not None else None,
            'nH': 0,
            'nC': 0,
            'nN': 0,
            'nO': 0,
            'nP': 0,  # Not parsed individually
            'manual': True,  # Flag to indicate this was manually added (skip X₀ threshold)
        }

        return jsonify(convert_numpy_types({'composition': composition}))

    except Exception as e:
        logger.error(f'ANALYSIS_CRASH in add_manual_composition_by_formula: {type(e).__name__}: {str(e)}')
        import traceback

        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/add_manual_composition', methods=['POST'])
def add_manual_composition() -> FlaskResponse:
    """Add a user-specified composition to a peak analysis"""
    try:
        data = request.get_json()
        analyzer = DNASilverAnalyzer()
        analyzer.custom_adducts = data.get('custom_adducts', []) or []
        peak_mz = float(data.get('peak_mz'))
        charge = int(data.get('charge'))
        intensity = float(data.get('intensity'))
        num_strands = int(data.get('num_strands'))
        num_silver = int(data.get('num_silver'))
        qcl = int(data.get('qcl'))
        dna_sequence = data.get('dna_sequence', '')
        resolution = int(data.get('resolution', DEFAULT_RESOLUTION))
        spectrum_data = data.get('spectrum')
        custom_xna = data.get('custom_xna', None)  # Get XNA settings

        # Get adduct information (array of {name, count})
        # Example: [{"name": "NH4", "count": 1}, {"name": "Na", "count": 2}]
        adducts_input = data.get('adducts', [])

        if adducts_input:
            total_adduct_mass, total_adduct_charge, adduct_string, adduct_elements = parse_adduct_items(
                adducts_input, analyzer.adducts
            )
            logger.info(
                f'Total adduct: {adduct_string} (mass={total_adduct_mass:.4f} Da, charge={total_adduct_charge:+d})'
            )
        else:
            total_adduct_mass = 0.0
            total_adduct_charge = 0
            adduct_string = ''
            adduct_elements = {}

        # For XNA mode, sequence is not required
        if not custom_xna and not dna_sequence:
            return jsonify({'error': 'DNA sequence is required'}), 400

        if not spectrum_data:
            return jsonify({'error': 'No spectrum data available'}), 400

        # Calculate N0
        # Formula: N₀ + Qcl = nAg (always, regardless of adducts)
        # Therefore: N₀ = nAg - Qcl
        n0 = num_silver - qcl

        if n0 < 0:
            return jsonify({'error': 'Invalid composition: N0 must be >= 0 (N0 = nAg - Qcl)'}), 400

        # Get strand_type for complex mode (strand1, strand2, or complex)
        strand_type = data.get('strand_type', None)

        # Calculate composition (DNA or XNA)
        user_neutral_mass = None
        if custom_xna and custom_xna.get('formula'):
            # XNA mode: Use custom formula and molecular weight
            from pythoms.molecule import composition_from_formula

            # For complex mode, select the appropriate formula based on strand_type
            is_complex_mode = custom_xna.get('is_complex', False)
            if is_complex_mode and strand_type in ['strand1', 'strand2']:
                # Use individual strand formula
                strand1_formula = custom_xna.get('strand1_formula', '') or custom_xna.get('formula', '')
                strand2_formula = (
                    custom_xna.get('strand2_formula', '') or strand1_formula
                )  # Fallback to strand1 if same strands

                if strand_type == 'strand1':
                    xna_formula = strand1_formula
                    logger.info(f'Complex XNA mode - using strand1 formula: {xna_formula}')
                else:  # strand_type == 'strand2'
                    xna_formula = strand2_formula
                    logger.info(f'Complex XNA mode - using strand2 formula: {xna_formula}')
            else:
                # Use combined formula (default for complex or regular XNA mode)
                xna_formula = custom_xna['formula']
                if is_complex_mode:
                    logger.info(f'Complex mode - using combined complex formula: {xna_formula}')

            xna_composition = composition_from_formula(xna_formula)
            nH = xna_composition.get('H', 0) * num_strands
            nC = xna_composition.get('C', 0) * num_strands
            nN = xna_composition.get('N', 0) * num_strands
            nO = xna_composition.get('O', 0) * num_strands
            nP = xna_composition.get('P', 0) * num_strands

            # Calculate masses from elements (for isotope pattern shape)
            mH_total = analyzer.m_p * nH
            mC_total = analyzer.mC * nC
            mN_total = analyzer.mN * nN
            mO_total = analyzer.mO * nO
            mP_total = analyzer.mP * nP
            mAg_total = analyzer.mAg * num_silver

            # Use user-provided molecular weight for mass calculation
            mXNA_one = custom_xna.get('molecular_weight')
            if mXNA_one is None:
                mXNA_one = analyzer.calculate_mass_from_formula(xna_formula)
            mDNA_total = mXNA_one * num_strands

            # Calculate user_neutral_mass for pattern shifting
            # This is the NEUTRAL mass (before ionization) used to position the isotope pattern
            # Include adduct mass in the neutral mass calculation
            user_neutral_mass = mDNA_total + mAg_total + total_adduct_mass

            logger.info(
                f'XNA mode - using custom molecular weight: {xna_formula}, Total neutral mass: {user_neutral_mass:.2f} Da'
            )

            # Calculate expected m/z using corrected XNA mass WITH adduct
            neutral_mass = mDNA_total + mAg_total + total_adduct_mass
            mass = neutral_mass - (qcl + charge + total_adduct_charge) * analyzer.m_p
        elif custom_xna and custom_xna.get('is_complex') and not custom_xna.get('formula'):
            # DNA-only Complex mode: use appropriate sequence based on strand_type
            seq1 = custom_xna.get('dna_sequence1', dna_sequence)
            seq2 = custom_xna.get('dna_sequence2', seq1)

            if strand_type == 'strand2':
                # Single strand 2 only - works like normal DNA mode
                seq_to_use = seq2
                logger.info(f'DNA Complex mode - strand2 only: {seq_to_use[:20]}...')
            elif strand_type == 'strand1':
                # Single strand 1 only - works like normal DNA mode
                seq_to_use = seq1
                logger.info(f'DNA Complex mode - strand1 only: {seq_to_use[:20]}...')
            else:
                # strand_type == 'complex': Full complex
                # User enters total strands (2 strands = 1 complex)
                seq_to_use = seq1
                logger.info(f'DNA Complex mode - complex (seq1 × {num_strands}): {seq_to_use[:20]}...')

            nH, nC, nN, nO, nP = analyzer.calculate_dna_composition(seq_to_use, num_strands)

            # Calculate masses
            mH_total = analyzer.m_p * nH
            mC_total = analyzer.mC * nC
            mN_total = analyzer.mN * nN
            mO_total = analyzer.mO * nO
            mP_total = analyzer.mP * nP
            mAg_total = analyzer.mAg * num_silver
            mDNA_total = mP_total + mH_total + mC_total + mN_total + mO_total

            # Calculate expected m/z using the standard formula WITH adduct
            neutral_mass = mDNA_total + mAg_total + total_adduct_mass
            mass = neutral_mass - (qcl + charge + total_adduct_charge) * analyzer.m_p
        else:
            # DNA mode: Calculate from sequence
            nH, nC, nN, nO, nP = analyzer.calculate_dna_composition(dna_sequence, num_strands)

            # Calculate masses
            mH_total = analyzer.m_p * nH
            mC_total = analyzer.mC * nC
            mN_total = analyzer.mN * nN
            mO_total = analyzer.mO * nO
            mP_total = analyzer.mP * nP
            mAg_total = analyzer.mAg * num_silver
            mDNA_total = mP_total + mH_total + mC_total + mN_total + mO_total

            # Calculate expected m/z using the standard formula WITH adduct
            # protons_removed = Qcl + z + adduct_charge
            neutral_mass = mDNA_total + mAg_total + total_adduct_mass
            mass = neutral_mass - (qcl + charge + total_adduct_charge) * analyzer.m_p

        expected_mz = mass / charge
        mass_error_ppm = abs((expected_mz - peak_mz) / peak_mz * 1e6)

        # Build formulas
        is_dna_only = num_silver == 0

        # Calculate protons removed (accounting for adduct charge)
        # protons_removed = Qcl + z + adduct_charge
        protons_removed = (qcl + charge + total_adduct_charge) if not is_dna_only else (charge + total_adduct_charge)

        # Add adduct elements to base composition for ion formula
        nH_total_with_adduct = nH + adduct_elements.get('H', 0)
        nC_total_with_adduct = nC + adduct_elements.get('C', 0)
        nN_total_with_adduct = nN + adduct_elements.get('N', 0)
        nO_total_with_adduct = nO + adduct_elements.get('O', 0)
        nP_total_with_adduct = nP + adduct_elements.get('P', 0)
        nCl_with_adduct = adduct_elements.get('Cl', 0)
        nNa_with_adduct = adduct_elements.get('Na', 0)
        nK_with_adduct = adduct_elements.get('K', 0)

        if custom_xna:
            # XNA formula display
            xna_name = custom_xna['name']
            if is_dna_only:
                neutral_formula = f'({xna_name}){to_subscript(num_strands)}'
            else:
                neutral_formula = f'({xna_name}){to_subscript(num_strands)}Ag{to_subscript(num_silver)}'

            # Add adduct to display formula
            if adduct_string:
                neutral_formula = f'{neutral_formula}+{adduct_string}'

            # Ion formula for isotope pattern (element-based + adducts)
            nH_ion = nH_total_with_adduct - protons_removed
            ion_formula = (
                f'C{nC_total_with_adduct}H{nH_ion}N{nN_total_with_adduct}O{nO_total_with_adduct}P{nP_total_with_adduct}'
            )
            if num_silver > 0:
                ion_formula += f'Ag{num_silver}'
            if nCl_with_adduct > 0:
                ion_formula += f'Cl{nCl_with_adduct}' if nCl_with_adduct > 1 else 'Cl'
            if nNa_with_adduct > 0:
                ion_formula += f'Na{nNa_with_adduct}' if nNa_with_adduct > 1 else 'Na'
            if nK_with_adduct > 0:
                ion_formula += f'K{nK_with_adduct}' if nK_with_adduct > 1 else 'K'
        else:
            # DNA formula display
            if is_dna_only:
                neutral_formula = f'C{nC}H{nH}N{nN}O{nO}P{nP}'
                nH_ion = nH_total_with_adduct - protons_removed
                ion_formula = f'C{nC_total_with_adduct}H{nH_ion}N{nN_total_with_adduct}O{nO_total_with_adduct}P{nP_total_with_adduct}'
            else:
                neutral_formula = f'C{nC}H{nH}N{nN}O{nO}P{nP}Ag{num_silver}'
                nH_ion = nH_total_with_adduct - protons_removed
                ion_formula = f'C{nC_total_with_adduct}H{nH_ion}N{nN_total_with_adduct}O{nO_total_with_adduct}P{nP_total_with_adduct}Ag{num_silver}'

            # Add adduct to display formula
            if adduct_string:
                neutral_formula = f'{neutral_formula}+{adduct_string}'

            # Add adduct elements to ion formula
            if nCl_with_adduct > 0:
                ion_formula += f'Cl{nCl_with_adduct}' if nCl_with_adduct > 1 else 'Cl'
            if nNa_with_adduct > 0:
                ion_formula += f'Na{nNa_with_adduct}' if nNa_with_adduct > 1 else 'Na'
            if nK_with_adduct > 0:
                ion_formula += f'K{nK_with_adduct}' if nK_with_adduct > 1 else 'K'

        # Generate isotope pattern and calculate X₀ error
        mz_values = np.array(spectrum_data['mz'])
        intensity_values = np.array(spectrum_data['intensity'])
        theo_pattern = analyzer.generate_isotope_pattern(ion_formula, charge, resolution)

        if 'error' not in theo_pattern:
            theo_mz = np.array(theo_pattern['gaussian_mz'])
            theo_intensity = np.array(theo_pattern['gaussian_intensity'])

            # Theoretical X₀ from Gaussian centroid fit
            theo_x0, theo_sigma = None, None
            if len(theo_mz) > 0 and np.sum(theo_intensity) > 0:
                theo_fit_result = analyzer.gaussian_fit_centroid(theo_mz, theo_intensity)
                if theo_fit_result and theo_fit_result[0] is not None:
                    theo_x0 = theo_fit_result[0]
                    theo_sigma = theo_fit_result[2] if len(theo_fit_result) > 2 else None
                else:
                    theo_x0 = np.sum(theo_mz * theo_intensity) / np.sum(theo_intensity)

            # Experimental Gaussian envelope
            window = 3.0
            mask = (mz_values >= peak_mz - window) & (mz_values <= peak_mz + window)
            exp_mz_window = mz_values[mask]
            exp_int_window = intensity_values[mask]
            exp_mz_gaussian, exp_int_gaussian = None, None

            if len(exp_mz_window) > 0:
                exp_mz_gaussian, exp_int_gaussian = analyzer.generate_experimental_gaussian_envelope(
                    exp_mz_window, exp_int_window, resolution
                )

            # Experimental X₀
            provided_exp_x0 = data.get('exp_x0')
            if provided_exp_x0 is not None:
                exp_x0 = float(provided_exp_x0)
                exp_sigma = None
            elif exp_mz_gaussian is not None and exp_int_gaussian is not None:
                exp_x0, exp_sigma, _ = analyzer.gaussian_fit_centroid(exp_mz_gaussian, exp_int_gaussian)
            elif len(exp_mz_window) > 0:
                exp_x0, exp_sigma, _ = analyzer.gaussian_fit_centroid(exp_mz_window, exp_int_window)
            else:
                exp_x0, exp_sigma = None, None

            # X₀ error
            if theo_x0 is not None and exp_x0 is not None:
                x0_error = abs(theo_x0 - exp_x0)
            else:
                x0_error = 999.0
        else:
            theo_mz = []
            theo_intensity = []
            theo_x0, theo_sigma = None, None
            exp_x0, exp_sigma = None, None
            x0_error = 999.0

        # Build composition object
        # For display: displayed_qcl = qcl + total_adduct_charge
        displayed_qcl = qcl + total_adduct_charge

        if is_dna_only:
            comp_type = 'XNA Only' if custom_xna else 'DNA Only'
            full_notation = f'{neutral_formula} (z={charge})'
        else:
            comp_type = 'nanocluster'
            full_notation = f'{neutral_formula}-{qcl + charge}H (z={charge}, Qcl={displayed_qcl}, N0={n0})'

        composition = {
            'type': comp_type,
            'num_strands': num_strands,
            'num_silver': num_silver,
            'qcl': qcl,  # Internal Qcl (N₀ + Qcl = nAg always)
            'displayed_qcl': displayed_qcl,  # For display: qcl + adduct_charge
            'n0': n0,
            'z': charge,
            'formula': neutral_formula,
            'ion_formula': ion_formula,
            'neutral_formula': neutral_formula,
            'adduct': adduct_string,  # Include adduct information
            'adduct_charge': total_adduct_charge,  # Include for N0+Qcl relation display
            'full_notation': full_notation,
            'expected_mz': expected_mz,
            'mass_error_ppm': mass_error_ppm,
            'x0_error': x0_error,
            'abs_x0_error': abs(x0_error) if x0_error is not None else 999.0,
            'theo_mz': theo_mz,
            'theo_intensity': theo_intensity,
            'theo_x0': float(theo_x0) if theo_x0 is not None else None,
            'theo_sigma': float(theo_sigma) if theo_sigma is not None else None,
            'exp_x0': float(exp_x0) if exp_x0 is not None else None,
            'exp_sigma': float(exp_sigma) if exp_sigma is not None else None,
            'nH': nH,
            'nC': nC,
            'nN': nN,
            'nO': nO,
            'nP': nP,
            'manual': True,  # Flag to indicate this was manually added (skip X₀ threshold)
        }

        return jsonify(convert_numpy_types({'composition': composition}))

    except Exception as e:
        logger.error(f'ANALYSIS_CRASH in add_manual_composition: {type(e).__name__}: {str(e)}')
        import traceback

        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/add_manual_composition_search', methods=['POST'])
def add_manual_composition_search() -> FlaskResponse:
    """Search for best N₀/Qcl composition with specified adduct"""
    try:
        data = request.get_json()
        analyzer = DNASilverAnalyzer()
        analyzer.custom_adducts = data.get('custom_adducts', []) or []
        peak_mz = float(data.get('peak_mz'))
        charge = int(data.get('charge'))
        intensity = float(data.get('intensity'))
        num_strands = int(data.get('num_strands'))
        num_silver = int(data.get('num_silver'))
        adducts_input = data.get('adducts', [])  # Array of {name, count}
        dna_sequence = data.get('dna_sequence', '')
        resolution = int(data.get('resolution', DEFAULT_RESOLUTION))
        spectrum_data = data.get('spectrum')
        custom_xna = data.get('custom_xna', None)
        strand_type = data.get('strand_type', None)  # 'strand1', 'strand2', 'complex', or None

        logger.info(
            f'SEARCH MODE: Finding best N₀ for specified adducts - nf={num_strands}, nAg={num_silver}, Peak: m/z={peak_mz:.4f}, z={charge}'
        )

        # For XNA mode, sequence is not required
        if not custom_xna and not dna_sequence:
            return jsonify({'error': 'DNA sequence is required'}), 400

        if not spectrum_data:
            return jsonify({'error': 'No spectrum data available'}), 400

        if adducts_input:
            total_adduct_mass, total_adduct_charge, adduct_string, adduct_elements = parse_adduct_items(
                adducts_input, analyzer.adducts
            )
            if adduct_string:
                logger.info(
                    f'Total adducts: {adduct_string} (mass={total_adduct_mass:.4f} Da, charge={total_adduct_charge:+d})'
                )
        else:
            total_adduct_mass = 0.0
            total_adduct_charge = 0
            adduct_string = ''
            adduct_elements = {}

        # Calculate base DNA/XNA composition
        if custom_xna and custom_xna.get('formula'):
            # XNA mode
            from pythoms.molecule import composition_from_formula

            # For complex mode, select the appropriate formula based on strand_type
            is_complex_mode = custom_xna.get('is_complex', False)
            if is_complex_mode and strand_type in ['strand1', 'strand2']:
                # Use individual strand formula
                strand1_formula = custom_xna.get('strand1_formula', '') or custom_xna.get('formula', '')
                strand2_formula = (
                    custom_xna.get('strand2_formula', '') or strand1_formula
                )  # Fallback to strand1 if same strands

                if strand_type == 'strand1':
                    xna_formula = strand1_formula
                    logger.info(f'Complex XNA search mode - using strand1 formula: {xna_formula}')
                else:  # strand_type == 'strand2'
                    xna_formula = strand2_formula
                    logger.info(f'Complex XNA search mode - using strand2 formula: {xna_formula}')
            else:
                # Use combined formula (default for complex or regular XNA mode)
                xna_formula = custom_xna['formula']
                if is_complex_mode:
                    logger.info(f'Complex search mode - using combined complex formula: {xna_formula}')

            xna_composition = composition_from_formula(xna_formula)
            nH = xna_composition.get('H', 0) * num_strands
            nC = xna_composition.get('C', 0) * num_strands
            nN = xna_composition.get('N', 0) * num_strands
            nO = xna_composition.get('O', 0) * num_strands
            nP = xna_composition.get('P', 0) * num_strands

            # Get XNA molecular weight
            mXNA_one = custom_xna.get('molecular_weight')
            if mXNA_one is None:
                mXNA_one = analyzer.calculate_mass_from_formula(xna_formula)
            mDNA_total = mXNA_one * num_strands
        elif custom_xna and custom_xna.get('is_complex') and not custom_xna.get('formula'):
            # DNA-only Complex mode: use appropriate sequence based on strand_type
            seq1 = custom_xna.get('dna_sequence1', dna_sequence)
            seq2 = custom_xna.get('dna_sequence2', seq1)

            if strand_type == 'strand2':
                # Single strand 2 only - works like normal DNA mode
                seq_to_use = seq2
                logger.info(f'DNA Complex search mode - strand2 only: {seq_to_use[:20]}...')
            elif strand_type == 'strand1':
                # Single strand 1 only - works like normal DNA mode
                seq_to_use = seq1
                logger.info(f'DNA Complex search mode - strand1 only: {seq_to_use[:20]}...')
            else:
                # strand_type == 'complex': Full complex
                # User enters total strands (2 strands = 1 complex)
                seq_to_use = seq1
                logger.info(f'DNA Complex search mode - complex (seq1 × {num_strands}): {seq_to_use[:20]}...')

            nH, nC, nN, nO, nP = analyzer.calculate_dna_composition(seq_to_use, num_strands)
            mH_total = analyzer.m_p * nH
            mC_total = analyzer.mC * nC
            mN_total = analyzer.mN * nN
            mO_total = analyzer.mO * nO
            mP_total = analyzer.mP * nP
            mDNA_total = mP_total + mH_total + mC_total + mN_total + mO_total
        else:
            # DNA mode
            nH, nC, nN, nO, nP = analyzer.calculate_dna_composition(dna_sequence, num_strands)
            mH_total = analyzer.m_p * nH
            mC_total = analyzer.mC * nC
            mN_total = analyzer.mN * nN
            mO_total = analyzer.mO * nO
            mP_total = analyzer.mP * nP
            mDNA_total = mP_total + mH_total + mC_total + mN_total + mO_total

        mAg_total = analyzer.mAg * num_silver

        # Prepare spectrum data for pattern matching
        mz_values = np.array(spectrum_data['mz'])
        intensity_values = np.array(spectrum_data['intensity'])

        # Get manual fit range if provided
        manual_fit_range = data.get('manual_fit_range')
        provided_exp_x0 = data.get('exp_x0')

        # Search all N₀ values (qcl from 0 to nAg)
        all_compositions = []

        for qcl in range(num_silver + 1):
            # Formula: N₀ + Qcl = nAg (always, regardless of adducts)
            # Therefore: N₀ = nAg - Qcl
            n0 = num_silver - qcl

            # Calculate mass for this qcl
            # protons_removed = Qcl + z + adduct_charge
            protons_removed = qcl + charge + total_adduct_charge

            if custom_xna:
                user_neutral_mass = mDNA_total + mAg_total + total_adduct_mass
                # Ion mass = neutral mass - mass of removed protons
                mass_ion = user_neutral_mass - (protons_removed * analyzer.m_p)
            else:
                user_neutral_mass = None
                # Ion mass = DNA + Ag + adducts - mass of removed protons
                mass_ion = mDNA_total + mAg_total + total_adduct_mass - (protons_removed * analyzer.m_p)

            expected_mz = mass_ion / charge
            mass_error_ppm = abs((expected_mz - peak_mz) / peak_mz * 1e6)

            # Build formulas
            protons_removed = qcl + charge + total_adduct_charge

            # Add adduct elements to base counts (for H, C, N, O, P which are in the base formula)
            nH_total_with_adduct = nH + adduct_elements.get('H', 0)
            nC_total_with_adduct = nC + adduct_elements.get('C', 0)
            nN_total_with_adduct = nN + adduct_elements.get('N', 0)
            nO_total_with_adduct = nO + adduct_elements.get('O', 0)
            nP_total_with_adduct = nP + adduct_elements.get('P', 0)

            # Build formulas
            if custom_xna:
                xna_name = custom_xna['name']
                neutral_formula = f'({xna_name}){to_subscript(num_strands)}Ag{to_subscript(num_silver)}'
                if adduct_string:
                    neutral_formula = f'{neutral_formula}+{adduct_string}'

                nH_ion = nH_total_with_adduct - protons_removed
                ion_formula = f'C{nC_total_with_adduct}H{nH_ion}N{nN_total_with_adduct}O{nO_total_with_adduct}P{nP_total_with_adduct}Ag{num_silver}'
            else:
                neutral_formula = f'C{nC}H{nH}N{nN}O{nO}P{nP}Ag{num_silver}'
                if adduct_string:
                    neutral_formula = f'{neutral_formula}+{adduct_string}'

                nH_ion = nH_total_with_adduct - protons_removed
                ion_formula = f'C{nC_total_with_adduct}H{nH_ion}N{nN_total_with_adduct}O{nO_total_with_adduct}P{nP_total_with_adduct}Ag{num_silver}'

            # Add ALL adduct elements to ion formula (handles any element: Cl, Na, K, Br, I, etc.)
            # Skip elements already in base formula (C, H, N, O, P, Ag)
            base_elements = {'C', 'H', 'N', 'O', 'P', 'Ag'}
            for element, count in adduct_elements.items():
                if element not in base_elements and count > 0:
                    ion_formula += f'{element}{count}' if count > 1 else element

            # Generate isotope pattern
            theo_pattern = analyzer.generate_isotope_pattern(ion_formula, charge, resolution)

            if 'error' not in theo_pattern:
                theo_mz = np.array(theo_pattern['gaussian_mz'])
                theo_intensity = np.array(theo_pattern['gaussian_intensity'])

                # Theoretical X₀ from Gaussian centroid fit
                theo_x0, theo_sigma = None, None
                if len(theo_mz) > 0 and np.sum(theo_intensity) > 0:
                    theo_fit_result = analyzer.gaussian_fit_centroid(theo_mz, theo_intensity)
                    if theo_fit_result and theo_fit_result[0] is not None:
                        theo_x0 = theo_fit_result[0]
                        theo_sigma = theo_fit_result[2] if len(theo_fit_result) > 2 else None
                    else:
                        theo_x0 = np.sum(theo_mz * theo_intensity) / np.sum(theo_intensity)

                # Experimental Gaussian envelope
                window = 3.0
                mask = (mz_values >= peak_mz - window) & (mz_values <= peak_mz + window)
                exp_mz_window = mz_values[mask]
                exp_int_window = intensity_values[mask]
                exp_mz_gaussian, exp_int_gaussian = None, None

                if len(exp_mz_window) > 0:
                    exp_mz_gaussian, exp_int_gaussian = analyzer.generate_experimental_gaussian_envelope(
                        exp_mz_window, exp_int_window, resolution
                    )

                # Experimental X₀
                if provided_exp_x0 is not None:
                    exp_x0 = float(provided_exp_x0)
                    exp_sigma = None
                elif exp_mz_gaussian is not None and exp_int_gaussian is not None:
                    exp_x0, exp_sigma, _ = analyzer.gaussian_fit_centroid(exp_mz_gaussian, exp_int_gaussian)
                elif len(exp_mz_window) > 0:
                    exp_x0, exp_sigma, _ = analyzer.gaussian_fit_centroid(exp_mz_window, exp_int_window)
                else:
                    exp_x0, exp_sigma = None, None

                # X₀ error
                if theo_x0 is not None and exp_x0 is not None:
                    x0_error = abs(theo_x0 - exp_x0)
                else:
                    x0_error = 999.0

                # Pattern similarity (stick-vs-apex comparison)
                pattern_score = 0.0
                if len(exp_mz_window) > 0:
                    theo_stick_mz = np.array(theo_pattern['mz'])
                    theo_stick_int = np.array(theo_pattern['intensity'])
                    pattern_score = analyzer.calculate_pattern_similarity(
                        theo_stick_mz, theo_stick_int, exp_mz_window, exp_int_window
                    )
            else:
                theo_mz = []
                theo_intensity = []
                theo_x0, theo_sigma = None, None
                exp_x0, exp_sigma = None, None
                x0_error = 999.0
                pattern_score = 0.0

            # Build composition object
            # For display: displayed_qcl = qcl + total_adduct_charge
            displayed_qcl = qcl + total_adduct_charge
            full_notation = f'{neutral_formula}-{qcl + charge}H (z={charge}, Qcl={displayed_qcl}, N0={n0})'

            composition = {
                'type': 'nanocluster',
                'num_strands': num_strands,
                'num_silver': num_silver,
                'qcl': qcl,  # Internal Qcl (N₀ + Qcl = nAg always)
                'displayed_qcl': displayed_qcl,  # For display: qcl + adduct_charge
                'n0': n0,
                'z': charge,
                'formula': neutral_formula,
                'ion_formula': ion_formula,
                'neutral_formula': neutral_formula,
                'adduct': adduct_string,
                'adduct_charge': total_adduct_charge,  # Include for N0+Qcl relation display
                'full_notation': full_notation,
                'expected_mz': expected_mz,
                'mass_error_ppm': mass_error_ppm,
                'x0_error': x0_error,
                'abs_x0_error': abs(x0_error) if x0_error is not None else 999.0,
                'pattern_score': pattern_score,
                'theo_mz': theo_mz,
                'theo_intensity': theo_intensity,
                'theo_x0': float(theo_x0) if theo_x0 is not None else None,
                'theo_sigma': float(theo_sigma) if theo_sigma is not None else None,
                'exp_x0': float(exp_x0) if exp_x0 is not None else None,
                'exp_sigma': float(exp_sigma) if exp_sigma is not None else None,
                'nH': nH,
                'nC': nC,
                'nN': nN,
                'nO': nO,
                'nP': nP,
                'custom_xna': custom_xna,
                'manual': True,  # Flag to indicate this was manually added (skip X₀ threshold)
            }

            all_compositions.append(composition)
            logger.debug(f'N₀={n0} (Qcl={qcl}): X₀_error={x0_error:.4f} m/z, pattern_score={pattern_score:.3f}')

        # Sort ALL compositions by X₀ error (lowest first) - this is the primary metric
        all_compositions.sort(key=lambda c: c['abs_x0_error'])

        # Show all N₀ values searched (for debugging)
        logger.debug(f'All {len(all_compositions)} compositions sorted by X₀ error')

        # The true best is now the first one (lowest X₀ error)
        true_best = all_compositions[0]
        logger.info(
            f'Best composition (by X₀ error): N₀={true_best["n0"]}, Qcl={true_best["qcl"]}, X₀ error: {true_best["x0_error"]:.4f} m/z, Pattern score: {true_best["pattern_score"]:.3f}'
        )

        # COMPLEX MODE: Only return N₀=0 composition (Qcl = nAg)
        # All complex strand types (strand1, strand2, complex, nd=X) have N₀=0 constraint
        is_complex = strand_type and (strand_type in ['strand1', 'strand2', 'complex'] or strand_type.startswith('nd='))
        if is_complex:
            # For complex mode, N₀ = 0 always, so Qcl = nAg
            complex_comp = next((c for c in all_compositions if c['n0'] == 0), None)
            if complex_comp:
                logger.info(
                    f'COMPLEX MODE: Returning only N₀=0 composition - nAg={complex_comp["num_silver"]}, Qcl={complex_comp["qcl"]}, X₀_err={complex_comp["x0_error"]:.4f}'
                )
                return jsonify(convert_numpy_types({'compositions': [complex_comp]}))
            else:
                logger.info('COMPLEX MODE: No N₀=0 composition found')
                return jsonify({'compositions': []})

        # NON-COMPLEX MODE: Return exactly 3 compositions (best Qcl, Qcl+1, Qcl-1)
        best_qcl = true_best['qcl']

        # Build list with best composition first, then Qcl±1
        result_compositions = [true_best]  # Best composition (lowest X₀ error)

        # Find Qcl-1 composition
        qcl_minus_1 = next((c for c in all_compositions if c['qcl'] == best_qcl - 1), None)
        if qcl_minus_1:
            result_compositions.append(qcl_minus_1)

        # Find Qcl+1 composition
        qcl_plus_1 = next((c for c in all_compositions if c['qcl'] == best_qcl + 1), None)
        if qcl_plus_1:
            result_compositions.append(qcl_plus_1)

        # Sort by X₀ error (best first) to maintain consistent ranking
        result_compositions.sort(key=lambda c: c['abs_x0_error'])

        logger.info(f'Returning {len(result_compositions)} compositions (best Qcl={best_qcl} ± 1)')

        return jsonify(convert_numpy_types({'compositions': result_compositions}))

    except Exception as e:
        logger.error(f'ANALYSIS_CRASH in add_manual_composition_search: {type(e).__name__}: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/reanalyze_peak', methods=['POST'])
@check_same_origin
def reanalyze_peak() -> FlaskResponse:
    """Re-analyze a single peak with user-specified charge state"""
    # Rate limiting check
    if not check_rate_limit(request.remote_addr):
        return jsonify({'error': 'Rate limit exceeded. Please wait before making more requests.'}), 429
    try:
        data = request.get_json()
        analyzer = DNASilverAnalyzer()
        analyzer.custom_adducts = data.get('custom_adducts', []) or []
        peak_mz = float(data.get('peak_mz'))
        charge = int(data.get('charge'))
        intensity = float(data.get('intensity'))
        dna_sequence = data.get('dna_sequence', '')
        resolution = int(data.get('resolution', DEFAULT_RESOLUTION))
        spectrum_data = data.get('spectrum')
        custom_xna = data.get('custom_xna', None)  # Get XNA settings if provided
        complex_mode = data.get('complex_mode', None)  # Get complex mode settings if provided

        # Handle complex mode - pass complex settings to analyzer
        if complex_mode and complex_mode.get('enabled'):
            logger.debug('[reanalyze_peak] Complex mode enabled')
            # For complex mode, use the complex XNA settings
            if complex_mode.get('xna'):
                custom_xna = complex_mode['xna']
                logger.debug(f'[reanalyze_peak] Using complex XNA: {custom_xna}')
            else:
                # DNA-only Complex mode: no XNA formula, but still flag as complex for N0=0
                custom_xna = {'name': 'Complex', 'is_complex': True, 'same_strands': False}
                logger.debug(f'[reanalyze_peak] DNA-only Complex mode: {custom_xna}')

        if not spectrum_data:
            return jsonify({'error': 'No spectrum data available'}), 400

        # Extract spectrum arrays
        mz_values = np.array(spectrum_data['mz'])
        intensity_values = np.array(spectrum_data['intensity'])

        # Generate experimental Gaussian curve for X0 calculation
        # Extract experimental data around the peak
        window = 3.0
        mask = (mz_values >= peak_mz - window) & (mz_values <= peak_mz + window)
        exp_mz_window = mz_values[mask]
        exp_int_window = intensity_values[mask]

        # Generate smooth Gaussian envelope (SAME AS AUTO ANALYSIS)
        exp_mz_gaussian, exp_int_gaussian = analyzer.generate_experimental_gaussian_envelope(
            exp_mz_window, exp_int_window, resolution
        )

        # Fit Gaussian using gaussian_fit_centroid (same method as custom search)
        if exp_mz_gaussian is not None and exp_int_gaussian is not None:
            fit_result = analyzer.gaussian_fit_centroid(exp_mz_gaussian, exp_int_gaussian)
            if fit_result and fit_result[0] is not None:
                exp_x0 = fit_result[0]
                exp_sigma = fit_result[1]
            else:
                exp_x0, exp_sigma = None, None
        else:
            exp_x0, exp_sigma = None, None

        if exp_x0 is None:
            logger.debug('[reanalyze_peak] Envelope generation failed, using weighted average')
            exp_x0, exp_sigma = analyzer.weighted_average_centroid(exp_mz_window, exp_int_window)
            exp_mz_gaussian = exp_mz_window
            exp_int_gaussian = exp_int_window

        # Use SMART composition finding with adduct search and X0 error threshold detection
        logger.info(f'[reanalyze_peak] Finding compositions with smart adduct search for exp_x0={exp_x0:.4f}')
        compositions = analyzer.analyze_peak_with_smart_adduct_search(
            peak_mz,
            charge,
            dna_sequence,
            exp_x0,
            resolution=resolution,
            mz_values=mz_values,
            intensity_values=intensity_values,
            custom_xna=custom_xna,
        )

        # Refine with isotope matching
        has_other_strands = False
        all_compositions = []
        has_odd_n0_warning = False

        if len(compositions) > 0:
            # Use underscores for returned Gaussian values - we keep the fitted ones calculated above
            (
                refined_compositions,
                _,
                _,
                has_other_strands,
                all_compositions,
                has_odd_n0_warning,
                _,
                _,
                has_unrealistic_n0_warning,
            ) = analyzer.refine_compositions_with_isotope_matching(
                compositions=compositions,
                experimental_mz=mz_values,
                experimental_int=intensity_values,
                peak_mz=peak_mz,
                resolution=resolution,
                detected_centroid=exp_x0,  # Use calculated X₀
            )
            logger.info(f'[reanalyze_peak] Refined {len(refined_compositions)} compositions')
        else:
            refined_compositions = []
            logger.info('[reanalyze_peak] No compositions found')

        # Keep the fitted Gaussian curve for display (exp_mz_gaussian, exp_int_gaussian already set above)

        # Calculate peak symmetry
        symmetry_info = analyzer.calculate_peak_symmetry(
            mz_values=mz_values, intensity_values=intensity_values, center_mz=peak_mz, window=2.0
        )

        # Prepare Gaussian curve for display (already calculated above)
        exp_gaussian_mz: list[float] = []
        exp_gaussian_intensity: list[float] = []
        if exp_mz_gaussian is not None and exp_int_gaussian is not None:
            exp_gaussian_mz = exp_mz_gaussian.tolist()
            exp_gaussian_intensity = exp_int_gaussian.tolist()

        # Determine if this is complex mode
        is_complex = complex_mode and complex_mode.get('enabled', False)

        # Get max intensity around the peak for display
        max_int = float(np.max(exp_int_window)) if len(exp_int_window) > 0 else 0.0

        # Convert all NumPy types to native Python types for JSON serialization
        result = {
            'peak_mz': float(peak_mz),
            'intensity': max_int,  # Required for frontend display
            'compositions': refined_compositions,
            'charge': charge,
            'charge_method': 'user_specified',
            'charge_confidence': 1.0,  # User specified = full confidence
            'exp_x0': float(exp_x0) if exp_x0 is not None else None,
            'exp_sigma': float(exp_sigma) if exp_sigma is not None else None,
            'symmetry': symmetry_info,
            'has_other_strands': has_other_strands,
            'all_compositions': all_compositions,
            'exp_gaussian_mz': exp_gaussian_mz,
            'exp_gaussian_intensity': exp_gaussian_intensity,
            'is_complex': is_complex,
        }
        return jsonify(convert_numpy_types(result))

    except Exception as e:
        logger.error(f'ANALYSIS_CRASH in reanalyze_peak: {type(e).__name__}: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/recalculate_peak_with_manual_fit', methods=['POST'])
def recalculate_peak_with_manual_fit() -> FlaskResponse:
    """Recalculate peak analysis with user-specified m/z range for Gaussian fitting"""
    try:
        data = request.get_json()
        analyzer = DNASilverAnalyzer()
        analyzer.custom_adducts = data.get('custom_adducts', []) or []
        peak_mz = float(data.get('peak_mz'))
        charge = int(data.get('charge'))
        intensity = float(data.get('intensity'))
        start_mz = float(data.get('start_mz'))
        end_mz = float(data.get('end_mz'))
        dna_sequence = data.get('dna_sequence', '')
        resolution = int(data.get('resolution', DEFAULT_RESOLUTION))
        spectrum_data = data.get('spectrum')
        custom_xna = data.get('custom_xna', None)  # Get XNA settings
        complex_mode = data.get('complex_mode', None)  # Get complex mode settings if provided

        # Handle complex mode - pass complex settings to analyzer
        if complex_mode and complex_mode.get('enabled'):
            logger.debug('[recalculate_peak_with_manual_fit] Complex mode enabled')
            # For complex mode, use the complex XNA settings
            if complex_mode.get('xna'):
                custom_xna = complex_mode['xna']
                logger.debug(f'[recalculate_peak_with_manual_fit] Using complex XNA: {custom_xna}')
            else:
                # DNA-only Complex mode: no XNA formula, but still flag as complex for N0=0
                custom_xna = {'name': 'Complex', 'is_complex': True, 'same_strands': False}
                logger.debug(f'[recalculate_peak_with_manual_fit] DNA-only Complex mode: {custom_xna}')

        if not spectrum_data:
            return jsonify({'error': 'No spectrum data available'}), 400

        # Validate m/z range
        if start_mz >= end_mz:
            return jsonify({'error': 'Start m/z must be less than end m/z'}), 400

        # Extract spectrum arrays
        mz_values = np.array(spectrum_data['mz'])
        intensity_values = np.array(spectrum_data['intensity'])

        # Extract experimental data within the user-specified range
        mask = (mz_values >= start_mz) & (mz_values <= end_mz)
        exp_mz_window = mz_values[mask]
        exp_int_window = intensity_values[mask]

        if len(exp_mz_window) < 3:
            return jsonify({'error': 'Not enough data points in specified range. Please widen the range.'}), 400

        logger.info(
            f'USER_ACTION: Manual Gaussian fit - Peak m/z: {peak_mz:.4f}, z={charge}, range: [{start_mz:.4f}, {end_mz:.4f}], {len(exp_mz_window)} points'
        )

        # Generate smooth Gaussian envelope from user-specified range
        logger.debug('Generating smooth envelope from user-specified range')
        exp_mz_gaussian, exp_int_gaussian = analyzer.generate_experimental_gaussian_envelope(
            exp_mz_window, exp_int_window, resolution
        )

        # Fit Gaussian using gaussian_fit_centroid (same method as custom search)
        if exp_mz_gaussian is not None and exp_int_gaussian is not None:
            fit_result = analyzer.gaussian_fit_centroid(exp_mz_gaussian, exp_int_gaussian)
            if fit_result and fit_result[0] is not None:
                exp_x0 = fit_result[0]
                exp_sigma = fit_result[1]
            else:
                exp_x0, exp_sigma = None, None
        else:
            exp_x0, exp_sigma = None, None

        if exp_x0 is None:
            logger.debug('Envelope generation failed, using weighted average')
            exp_x0, exp_sigma = analyzer.weighted_average_centroid(exp_mz_window, exp_int_window)
            exp_mz_gaussian = exp_mz_window
            exp_int_gaussian = exp_int_window
            logger.debug(f'exp_x0={exp_x0:.4f} (fallback)')

        # Calculate symmetry (for info only)
        symmetry_info = analyzer.calculate_peak_symmetry(
            mz_values=mz_values, intensity_values=intensity_values, center_mz=peak_mz, window=2.0
        )
        symmetry_percent = symmetry_info.get('symmetry_score', 0.0) * 100
        logger.debug(f'Peak symmetry: {symmetry_percent:.1f}%')

        # Use SAME composition finding as automatic analysis (with smart adduct search)
        logger.info(f'Finding compositions using exp_x0={exp_x0:.4f}')
        compositions = analyzer.analyze_peak_with_smart_adduct_search(
            peak_mz,
            charge,
            dna_sequence,
            exp_x0,
            resolution=resolution,
            mz_values=mz_values,
            intensity_values=intensity_values,
            custom_xna=custom_xna,  # Pass XNA settings for pattern matching!
        )

        # Refine with isotope matching - SAME AS AUTO ANALYSIS
        has_other_strands = False
        all_compositions = []
        has_odd_n0_warning = False
        has_unrealistic_n0_warning = False

        if len(compositions) > 0:
            (
                compositions,
                exp_x0_refined,
                exp_sigma_refined,
                has_other_strands,
                all_compositions,
                has_odd_n0_warning,
                _,
                _,
                has_unrealistic_n0_warning,
            ) = analyzer.refine_compositions_with_isotope_matching(
                compositions,
                mz_values,
                intensity_values,
                peak_mz,
                resolution=resolution,
                detected_centroid=exp_x0,  # Use manual fit X₀
            )
            # Keep the manual fit X₀ (don't overwrite)
            # exp_x0 stays as is
            logger.info(f'Refined {len(compositions)} compositions using exp_x0={exp_x0:.4f}')

        # Sort compositions by combined score (pattern similarity + X0 error)
        # Higher score is better
        def combined_score_manual(comp):
            pattern_sim = comp.get('pattern_similarity', 0.0)
            x0_err = abs(comp.get('x0_error', 999.0)) if comp.get('x0_error') not in [None, 999.0] else 999.0
            return pattern_sim - (x0_err * 0.1)

        compositions.sort(key=combined_score_manual, reverse=True)  # reverse=True for descending (highest score first)

        logger.debug(
            f'Top {min(10, len(compositions))} compositions after manual fit recalculation (sorted by combined score)'
        )

        # FILTER BY Qcl ± 1: Find best composition by SMALLEST X0 ERROR, then keep only Qcl ± 1
        valid_comps = [c for c in compositions if c.get('x0_error', 999.0) != 999.0]

        if valid_comps:
            # Sort by SMALLEST X0 ERROR (for determining Qcl range)
            sorted_comps = sorted(valid_comps, key=lambda c: abs(c.get('x0_error', 999.0)))
            best_comp = sorted_comps[0]

            best_qcl = best_comp.get('qcl') if best_comp.get('type') == 'nanocluster' else None
            best_pattern_sim = best_comp.get('pattern_similarity', 0.0)
            best_x0_err = best_comp.get('x0_error', 999.0)

            logger.debug(
                f'Best X0 match for Qcl filtering: Type={best_comp.get("type")}, nAg={best_comp.get("num_silver")}, N0={best_comp.get("n0")}, Qcl={best_qcl}, X0_error={best_x0_err:.4f}'
            )

            # Filter to keep only Qcl ± 1 of best composition (for nanoclusters) + all non-nanoclusters
            if best_qcl is not None:
                filtered_compositions = [
                    comp
                    for comp in compositions
                    if (comp.get('type') != 'nanocluster')  # Keep all non-cluster
                    or (comp.get('qcl') is not None and abs(comp['qcl'] - best_qcl) <= 1)  # Qcl±1 of best
                ]

                nanocluster_count = len([c for c in filtered_compositions if c.get('type') == 'nanocluster'])
                non_cluster_count = len([c for c in filtered_compositions if c.get('type') != 'nanocluster'])
                logger.info(
                    f'FILTERED: Kept {nanocluster_count} nanoclusters (Qcl±1 of {best_qcl}) + {non_cluster_count} non-cluster'
                )

                compositions = filtered_compositions if len(filtered_compositions) > 0 else compositions
            else:
                logger.debug('Best composition is non-nanocluster - no Qcl filtering applied')
        else:
            logger.warning('No valid compositions with X0 error - returning all')

        # Re-sort filtered compositions by PATTERN SIMILARITY (highest first)
        # Now that we've filtered to correct Qcl range, rank by how well patterns match
        compositions.sort(key=combined_score_manual, reverse=True)  # Highest score first

        # Take top matches after filtering and re-sorting
        refined_compositions = compositions[:10] if len(compositions) > 10 else compositions

        logger.info(f'Final compositions: {len(refined_compositions)} after Qcl±1 filter')

        # Convert to JSON-serializable format
        result = {
            'compositions': refined_compositions,
            'charge': charge,
            'exp_x0': float(exp_x0) if exp_x0 is not None else None,
            'exp_sigma': float(exp_sigma) if exp_sigma is not None else None,
            'symmetry': symmetry_info,
            'manual_fit_range': [float(start_mz), float(end_mz)],
            'exp_gaussian_mz': exp_mz_gaussian.tolist() if exp_mz_gaussian is not None else [],
            'exp_gaussian_intensity': exp_int_gaussian.tolist() if exp_int_gaussian is not None else [],
            'has_unrealistic_n0_warning': has_unrealistic_n0_warning,
        }

        return jsonify(convert_numpy_types(result))

    except Exception as e:
        logger.error(f'ANALYSIS_CRASH in recalculate_peak_with_manual_fit: {type(e).__name__}: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/try_higher_strands', methods=['POST'])
def try_higher_strands() -> FlaskResponse:
    """Try higher strand numbers (4-6) for a given peak"""
    try:
        data = request.get_json()
        peak_mz = float(data['peak_mz'])
        _custom_adducts_payload = data.get('custom_adducts', []) or []
        dna_sequence = data.get('dna_sequence', '')
        charge = int(data.get('charge', 1))

        # Get spectrum data from request
        mz_list = data.get('mz_values', [])
        intensity_list = data.get('intensity_values', [])

        if not dna_sequence:
            return jsonify({'error': 'DNA sequence required'}), 400

        if not mz_list or not intensity_list:
            return jsonify({'error': 'No spectrum data provided'}), 400

        analyzer = DNASilverAnalyzer()
        analyzer.custom_adducts = _custom_adducts_payload

        mz_values = np.array(mz_list)
        intensity_values = np.array(intensity_list)

        # Generate initial compositions for strands 1-3 to find max strand number
        logger.info(f'Finding current compositions for peak at m/z {peak_mz:.4f}')
        compositions = []

        # First pass: generate compositions for strands 1-3
        for num_strands in range(1, MAX_STRANDS + 1):
            for num_ag in range(2, MAX_SILVER + 1):
                nH, nC, nN, nO, nP = analyzer.calculate_dna_composition(dna_sequence, num_strands)

                mH_total = analyzer.m_p * nH
                mC_total = analyzer.mC * nC
                mN_total = analyzer.mN * nN
                mO_total = analyzer.mO * nO
                mP_total = analyzer.mP * nP
                mAg_total = analyzer.mAg * num_ag

                z_values = [charge] if charge else [1, 2, 3, 4, 5, 6, 7, 8]

                for z_test in z_values:
                    if z_test is None or z_test <= 0:
                        continue

                    for qcl in range(0, num_ag + 1):
                        n0_valence = num_ag - qcl
                        if n0_valence < 0:
                            continue

                        mass = (
                            mP_total
                            + mH_total
                            + mC_total
                            + mN_total
                            + mO_total
                            + mAg_total
                            - (qcl + z_test) * analyzer.m_p
                        )
                        expected_mz = mass / z_test
                        mass_error_ppm = abs((expected_mz - peak_mz) / peak_mz * 1e6)

                        if mass_error_ppm < 200:
                            compositions.append(
                                {
                                    'num_strands': num_strands,
                                    'num_silver': num_ag,
                                    'qcl': qcl,
                                    'n0': n0_valence,
                                    'mass_error_ppm': mass_error_ppm,
                                }
                            )

        # Find the maximum strand number in current compositions
        max_strand = 0
        for comp in compositions:
            if comp.get('num_strands', 0) > max_strand:
                max_strand = comp['num_strands']

        # Search for next strand number (max + 1)
        next_strand = max_strand + 1
        logger.info(f'Current max strand number: {max_strand}, searching for strand number: {next_strand}')

        # Now generate compositions for the next strand number
        # Start fresh with only the next strand number
        compositions = []

        # Start from num_ag=0 to include DNA-only compositions
        for num_ag in range(0, MAX_SILVER + 1):
            num_strands = next_strand
            nH, nC, nN, nO, nP = analyzer.calculate_dna_composition(dna_sequence, num_strands)

            mH_total = analyzer.m_p * nH
            mC_total = analyzer.mC * nC
            mN_total = analyzer.mN * nN
            mO_total = analyzer.mO * nO
            mP_total = analyzer.mP * nP
            mAg_total = analyzer.mAg * num_ag

            z_values = [charge] if charge else [1, 2, 3, 4, 5, 6, 7, 8]

            for z_test in z_values:
                if z_test is None or z_test <= 0:
                    continue

                for qcl in range(0, num_ag + 1):
                    n0_valence = num_ag - qcl
                    if n0_valence < 0:
                        continue

                    mass = (
                        mP_total + mH_total + mC_total + mN_total + mO_total + mAg_total - (qcl + z_test) * analyzer.m_p
                    )
                    expected_mz = mass / z_test
                    mass_error_ppm = abs((expected_mz - peak_mz) / peak_mz * 1e6)

                    if mass_error_ppm < 200:
                        neutral_formula = f'C{nC}H{nH}N{nN}O{nO}P{nP}Ag{num_ag}'
                        nH_ion = nH - (qcl + z_test)
                        ion_formula = f'C{nC}H{nH_ion}N{nN}O{nO}P{nP}Ag{num_ag}'

                        compositions.append(
                            {
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
                                'full_notation': f'{neutral_formula}-{qcl + z_test}H (z={z_test}, Qcl={qcl}, N0={n0_valence})',
                                'expected_mz': expected_mz,
                                'mass_error_ppm': mass_error_ppm,
                                'x0_error': 999.0,
                                'nH': nH,
                                'nC': nC,
                                'nN': nN,
                                'nO': nO,
                                'nP': nP,
                            }
                        )

        logger.info(f'Found {len(compositions)} total compositions (strands 1-{next_strand})')

        # Refine with isotope matching
        (
            refined_compositions,
            exp_x0,
            exp_sigma,
            has_other_strands,
            all_compositions,
            has_odd_n0_warning,
            _,
            _,
            has_unrealistic_n0_warning,
        ) = analyzer.refine_compositions_with_isotope_matching(
            compositions, mz_values, intensity_values, peak_mz, resolution=20000
        )

        # Calculate symmetry
        symmetry_info = analyzer.calculate_peak_symmetry(
            mz_values=mz_values, intensity_values=intensity_values, center_mz=peak_mz, window=2.0
        )

        # Count compositions by strand number
        strand_counts: dict[int, int] = {}
        for comp in refined_compositions:
            num_strands = comp.get('num_strands', 0)
            strand_counts[num_strands] = strand_counts.get(num_strands, 0) + 1

        return jsonify(
            {
                'compositions': refined_compositions,
                'all_compositions': all_compositions,
                'charge': charge,
                'exp_x0': float(exp_x0) if exp_x0 is not None else None,
                'exp_sigma': float(exp_sigma) if exp_sigma is not None else None,
                'symmetry': symmetry_info,
                'has_other_strands': has_other_strands,
                'strand_range': f'1-{next_strand}',
                'strand_counts': strand_counts,
                'next_strand': next_strand,
                'has_unrealistic_n0_warning': has_unrealistic_n0_warning,
            }
        )

    except Exception as e:
        logger.error(f'ANALYSIS_CRASH in try_higher_strands: {type(e).__name__}: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index() -> str:
    """Render main page"""
    return render_template('index.html')


@app.route('/calculate_dna_mass', methods=['POST'])
def calculate_dna_mass() -> FlaskResponse:
    """Calculate the mass of single-stranded DNA from sequence"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        dna_sequence = data.get('dna_sequence', '').upper().strip()

        if not dna_sequence:
            return jsonify({'error': 'No DNA sequence provided'}), 400

        # Validate sequence (only ATCG allowed)
        valid_bases = set('ATCG')
        if not all(base in valid_bases for base in dna_sequence):
            return jsonify({'error': 'Invalid DNA sequence. Only A, T, C, G allowed.'}), 400

        # Count bases
        base_count = {
            'A': dna_sequence.count('A'),
            'T': dna_sequence.count('T'),
            'C': dna_sequence.count('C'),
            'G': dna_sequence.count('G'),
        }

        # Calculate composition for single strand (nf=1)
        nH, nC, nN, nO, nP = analyzer.calculate_dna_composition(dna_sequence, strands=1)

        # Calculate total mass
        mass = analyzer.m_p * nH + analyzer.mC * nC + analyzer.mN * nN + analyzer.mO * nO + analyzer.mP * nP

        return jsonify(
            {
                'mass': float(mass),
                'length': len(dna_sequence),
                'composition': {'H': nH, 'C': nC, 'N': nN, 'O': nO, 'P': nP},
                'base_count': base_count,
                'formula': f'C{nC}H{nH}N{nN}O{nO}P{nP}',
            }
        )

    except Exception as e:
        logger.error(f'ANALYSIS_CRASH in calculate_dna_mass: {type(e).__name__}: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_spectrum() -> FlaskResponse:
    """Handle spectrum file upload with automatic peak detection and charge assignment"""
    # Rate limiting check
    if not check_rate_limit(request.remote_addr):
        return jsonify({'error': 'Rate limit exceeded. Please wait before uploading.'}), 429
    try:
        # Clear caches when new spectrum is uploaded
        global _peak_analysis_cache, _isotope_pattern_cache
        peak_cache_size = len(_peak_analysis_cache)
        isotope_cache_size = len(_isotope_pattern_cache)
        _peak_analysis_cache.clear()
        _isotope_pattern_cache.clear()
        if peak_cache_size > 0 or isotope_cache_size > 0:
            logger.debug(f'Cleared caches (peak: {peak_cache_size}, isotope: {isotope_cache_size})')

        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read file content
        content = file.read().decode('utf-8')

        # Parse spectrum
        mz_values, intensity_values = analyzer.parse_txt_spectrum(content)

        if len(mz_values) == 0:
            return jsonify({'error': 'No valid data found in file'}), 400

        # Auto-detect resolution from spectrum
        estimated_resolution = analyzer.estimate_resolution(mz_values, intensity_values)
        logger.info(f'Auto-detected resolution: {estimated_resolution}')

        # AUTO-DETECT ALL PEAKS AND ASSIGN CHARGES (Senko et al. 1995 method)
        logger.info('Detecting peak regions (isotope envelopes) and assigning charge states...')
        detected_peaks = detect_all_peaks_with_charge(
            mz_values, intensity_values, prominence=0.01, charge_range=(1, 10), method='combination', merge_gap=1.5
        )
        logger.info(f'Found {len(detected_peaks)} isotope envelopes with charge states')

        # Cross-check each peak's charge with direct spacing measurement
        for peak in detected_peaks:
            if peak['charge'] is not None:
                spacing_check = analyzer.detect_charge_state(mz_values, intensity_values, peak['mz'], window=3.0)
                if (
                    spacing_check['charge'] is not None
                    and spacing_check['num_peaks'] >= 3
                    and spacing_check['charge'] != peak['charge']
                ):
                    logger.info(
                        f'Spacing override at m/z {peak["mz"]:.2f}: z={peak["charge"]} -> z={spacing_check["charge"]}'
                    )
                    peak['charge'] = spacing_check['charge']
                    peak['confidence'] = max(0.5, spacing_check['confidence'])
                    peak['method'] = 'spacing_override'

        # Convert to JSON-serializable format
        peaks_with_charge = []
        for peak in detected_peaks:
            peaks_with_charge.append(
                {
                    'mz': float(peak['mz']),
                    'intensity': float(peak['intensity']),
                    'charge': int(peak['charge']) if peak['charge'] is not None else None,
                    'confidence': float(peak['confidence']),
                    'method': peak['method'],
                }
            )

        # Return spectrum data with auto-detected peaks
        return jsonify(
            {
                'spectrum': {'mz': mz_values.tolist(), 'intensity': intensity_values.tolist()},
                'num_points': len(mz_values),
                'mz_range': [float(np.min(mz_values)), float(np.max(mz_values))],
                'resolution': int(estimated_resolution),
                'auto_detected_peaks': peaks_with_charge,  # NEW: Send peak list with z values
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/load_sample', methods=['GET'])
def load_sample() -> FlaskResponse:
    """Load sample spectrum file for demo/testing"""
    try:
        # Clear caches when new spectrum is loaded
        global _peak_analysis_cache, _isotope_pattern_cache
        _peak_analysis_cache.clear()
        _isotope_pattern_cache.clear()

        # Sample file path (in sample_data directory)
        sample_file = os.path.join(os.path.dirname(__file__), 'sample_data', 'GG208.txt')

        if not os.path.exists(sample_file):
            return jsonify({'error': 'Sample file not found in sample_data directory.'}), 404

        # Read file content
        with open(sample_file, 'r') as f:
            content = f.read()

        # Parse spectrum
        mz_values, intensity_values = analyzer.parse_txt_spectrum(content)

        if len(mz_values) == 0:
            return jsonify({'error': 'No valid data found in sample file'}), 400

        # Auto-detect resolution from spectrum
        estimated_resolution = analyzer.estimate_resolution(mz_values, intensity_values)
        logger.info(f'Auto-detected resolution: {estimated_resolution}')

        # AUTO-DETECT ALL PEAKS AND ASSIGN CHARGES
        logger.info('Detecting peak regions (isotope envelopes) and assigning charge states...')
        detected_peaks = detect_all_peaks_with_charge(
            mz_values, intensity_values, prominence=0.01, charge_range=(1, 10), method='combination', merge_gap=1.5
        )
        logger.info(f'Found {len(detected_peaks)} isotope envelopes with charge states')

        # Cross-check each peak's charge with direct spacing measurement
        for peak in detected_peaks:
            if peak['charge'] is not None:
                spacing_check = analyzer.detect_charge_state(mz_values, intensity_values, peak['mz'], window=3.0)
                if (
                    spacing_check['charge'] is not None
                    and spacing_check['num_peaks'] >= 3
                    and spacing_check['charge'] != peak['charge']
                ):
                    logger.info(
                        f'Spacing override at m/z {peak["mz"]:.2f}: z={peak["charge"]} -> z={spacing_check["charge"]}'
                    )
                    peak['charge'] = spacing_check['charge']
                    peak['confidence'] = max(0.5, spacing_check['confidence'])
                    peak['method'] = 'spacing_override'

        # Convert to JSON-serializable format
        peaks_with_charge = []
        for peak in detected_peaks:
            peaks_with_charge.append(
                {
                    'mz': float(peak['mz']),
                    'intensity': float(peak['intensity']),
                    'charge': int(peak['charge']) if peak['charge'] is not None else None,
                    'confidence': float(peak['confidence']),
                    'method': peak['method'],
                }
            )

        # Return spectrum data with auto-detected peaks
        return jsonify(
            {
                'spectrum': {'mz': mz_values.tolist(), 'intensity': intensity_values.tolist()},
                'num_points': len(mz_values),
                'mz_range': [float(np.min(mz_values)), float(np.max(mz_values))],
                'resolution': int(estimated_resolution),
                'auto_detected_peaks': peaks_with_charge,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download_sample', methods=['GET'])
def download_sample() -> FlaskResponse:
    """Download sample spectrum file so users can see the data format"""
    try:
        sample_file = os.path.join(os.path.dirname(__file__), 'sample_data', 'GG208.txt')

        if not os.path.exists(sample_file):
            return jsonify({'error': 'Sample file not found'}), 404

        return send_file(sample_file, as_attachment=True, download_name='sample_spectrum.txt', mimetype='text/plain')

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/isotope_pattern', methods=['POST'])
def get_isotope_pattern() -> FlaskResponse:
    """Generate isotope pattern for a given formula"""
    try:
        data = request.get_json()
        formula = data.get('formula')
        charge = int(data.get('charge', 1))
        resolution = int(data.get('resolution', DEFAULT_RESOLUTION))

        if not formula:
            return jsonify({'error': 'No formula provided'}), 400

        pattern = analyzer.generate_isotope_pattern(formula, charge, resolution)

        return jsonify(pattern)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/find_composition', methods=['POST'])
def find_composition() -> FlaskResponse:
    """Search for a composition in the experimental spectrum"""
    try:
        data = request.get_json()
        analyzer = DNASilverAnalyzer()
        analyzer.custom_adducts = data.get('custom_adducts', []) or []
        formula = data.get('formula')
        charge = int(data.get('charge', 1))
        qcl = int(data.get('qcl', 0))
        resolution = int(data.get('resolution', DEFAULT_RESOLUTION))
        spectrum_data = data.get('spectrum')
        peaks_data = data.get('peaks')
        adducts_input = data.get('adducts', [])

        if not formula:
            return jsonify({'error': 'No formula provided'}), 400

        # Parse adducts and calculate total mass and charge
        total_adduct_mass = 0.0
        total_adduct_charge = 0
        if adducts_input:
            for adduct_item in adducts_input:
                adduct_name = adduct_item.get('name', '')
                adduct_count = int(adduct_item.get('count', 1))

                inline_mass = adduct_item.get('mass')
                inline_charge = adduct_item.get('charge')
                if inline_mass is not None and inline_charge is not None:
                    adduct_mass = float(inline_mass)
                    adduct_charge = int(inline_charge)
                elif adduct_name in analyzer.adducts:
                    adduct_mass, adduct_charge = analyzer.adducts[adduct_name]
                else:
                    logger.warning(f"Adduct '{adduct_name}' not found in library, skipping")
                    continue

                total_adduct_mass += adduct_mass * adduct_count
                total_adduct_charge += adduct_charge * adduct_count
                logger.debug(
                    f'Adduct: {adduct_count}×{adduct_name}: mass={adduct_mass * adduct_count:.4f} Da, charge={adduct_charge * adduct_count:+d}'
                )

            logger.info(f'Total adducts: mass={total_adduct_mass:.4f} Da, charge={total_adduct_charge:+d}')

        # Sanitize formula - remove any "mz" suffix that might have been accidentally appended
        original_formula = formula
        formula = formula.strip()
        if formula.endswith('mz'):
            logger.warning(
                f"Removing 'mz' suffix from formula. Original: '{original_formula}', Cleaned: '{formula[:-2]}'"
            )
            formula = formula[:-2]

        logger.debug(
            f"Received formula for composition search: '{formula}', charge: {charge}, qcl: {qcl}, adduct_mass: {total_adduct_mass}, adduct_charge: {total_adduct_charge}"
        )

        if not spectrum_data or not peaks_data:
            return jsonify({'error': 'No spectrum data available. Please upload a spectrum first.'}), 400

        # Extract spectrum arrays
        mz_values = np.array(spectrum_data['mz'])
        intensity_values = np.array(spectrum_data['intensity'])

        # Search for composition
        result = analyzer.find_composition_in_spectrum(
            formula,
            charge,
            qcl,
            mz_values,
            intensity_values,
            peaks_data,
            resolution,
            adduct_mass=total_adduct_mass,
            adduct_charge=total_adduct_charge,
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Peak analysis cache - stores results by (peak_apex_mz, dna_sequence, resolution)
_peak_analysis_cache: dict[tuple[float, str, int], dict[str, Any]] = {}


@app.route('/analyze_region', methods=['POST'])
@check_same_origin
def analyze_region() -> FlaskResponse:
    """Analyze a clicked region - auto-detect charge state and find compositions"""
    # Rate limiting check
    if not check_rate_limit(request.remote_addr):
        return jsonify({'error': 'Rate limit exceeded. Please wait before making more requests.'}), 429
    logger.info('ANALYZE_REGION v2.1 - Processing request')
    sys.stdout.flush()  # Force immediate output
    try:
        data = request.get_json()
        analyzer = DNASilverAnalyzer()
        analyzer.custom_adducts = data.get('custom_adducts', []) or []
        clicked_mz = float(data.get('clicked_mz'))
        detected_centroid = data.get('detected_centroid')  # Frontend-calculated centroid
        spectrum_data = data.get('spectrum')
        dna_sequence = data.get('dna_sequence', '')
        custom_xna = data.get('custom_xna', None)  # Custom XNA nucleotide data

        # For XNA mode, sequence is not required (we use total mass directly)
        if not custom_xna and not dna_sequence:
            return jsonify({'error': 'DNA sequence is required'}), 400

        if not spectrum_data:
            return jsonify({'error': 'No spectrum data'}), 400

        # Extract spectrum arrays
        mz_values = np.array(spectrum_data['mz'])
        intensity_values = np.array(spectrum_data['intensity'])

        # Define window around clicked point (±2 m/z)
        window = 2.0
        mask = (mz_values >= clicked_mz - window) & (mz_values <= clicked_mz + window)
        region_mz = mz_values[mask]
        region_int = intensity_values[mask]

        if len(region_mz) < 5:
            return jsonify({'error': 'Insufficient data in clicked region'}), 400

        # Find the actual peak maximum in this region
        max_idx = np.argmax(region_int)
        peak_mz = region_mz[max_idx]
        max_int = region_int[max_idx]

        # Check if charge was already detected from upload (frontend passes it)
        detected_charge = data.get('detected_charge', None)
        detected_charge_confidence = data.get('charge_confidence', None)
        detected_charge_method = data.get('charge_method', None)

        if detected_charge is not None:
            # Verify auto-detected charge with direct spacing measurement
            charge = int(detected_charge)
            charge_confidence = float(detected_charge_confidence) if detected_charge_confidence is not None else 0.8
            charge_method = detected_charge_method if detected_charge_method is not None else 'auto_detected'

            # Cross-check: spacing method is more reliable for isotope-rich species (e.g. DNA-AgN)
            spacing_check = analyzer.detect_charge_state(mz_values, intensity_values, peak_mz, window=3.0)
            if spacing_check['charge'] is not None and spacing_check['num_peaks'] >= 3:
                spacing_charge = spacing_check['charge']
                spacing_conf = float(spacing_check.get('confidence', 0.0))
                if spacing_charge != charge:
                    # Guard against domain-wrong spacing overrides:
                    #  (a) Ag-doublet halving artifact: spacing detector halves z because
                    #      Ag isotope doublets look like z/2 spacing — exactly the signal
                    #      we are trying to assign, so this halving is actively wrong here.
                    #  (b) Senko upload-time auto-detect with high confidence already
                    #      considered the full envelope; the single-peak spacing heuristic
                    #      is less reliable at high z (spacing < FWHM).
                    halving_artifact = spacing_charge * 2 == charge
                    trusted_supplied = charge_method == 'auto_detected' and charge_confidence >= 0.85
                    spacing_stronger = spacing_check['num_peaks'] >= 4 and spacing_conf > charge_confidence
                    if halving_artifact or trusted_supplied or not spacing_stronger:
                        logger.info(
                            f'Spacing override SUPPRESSED at m/z {peak_mz:.4f}: spacing z={spacing_charge} '
                            f'(conf={spacing_conf:.2f}, peaks={spacing_check["num_peaks"]}) '
                            f'vs supplied z={charge} (conf={charge_confidence:.2f}, method={charge_method}); '
                            f'halving={halving_artifact}, trusted_supplied={trusted_supplied}, '
                            f'spacing_stronger={spacing_stronger}'
                        )
                    else:
                        logger.info(
                            f'Spacing method disagrees: z={spacing_charge} (from {spacing_check["num_peaks"]} peaks, spacing={spacing_check["spacing"]:.4f}) vs auto-detected z={charge}'
                        )
                        charge = spacing_charge
                        charge_method = 'spacing_override'
                        charge_confidence = max(0.5, spacing_conf)

            logger.info(
                f'Using charge for peak at m/z {peak_mz:.4f}: z={charge} (method: {charge_method}, confidence: {charge_confidence * 100:.1f}%)'
            )
        else:
            # No auto-detected charge available - detect charge now
            # Use improved charge detection:
            # 1. Direct isotope spacing measurement (primary)
            # 2. Senko method fallback
            # 3. Prompt user for manual input if all fail
            logger.info(f'Detecting charge state for clicked peak at m/z {peak_mz:.4f}...')

            charge_result = analyzer.detect_charge_for_clicked_peak(
                mz_values, intensity_values, target_mz=peak_mz, charge_range=(1, 10)
            )

            charge = charge_result['charge']
            charge_confidence = charge_result['confidence']
            charge_method = charge_result['method']

            # If charge detection failed completely, return special response to prompt user
            if charge is None:
                logger.warning('Charge detection failed - prompting user for manual input')
                return jsonify(
                    {
                        'charge_required': True,
                        'peak_mz': peak_mz,
                        'message': 'Could not automatically detect charge state. Please enter the charge (z) manually.',
                    }
                )

            logger.info(
                f'Final result: z={charge} (method: {charge_method}, confidence: {charge_confidence * 100:.1f}%)'
            )

            # If confidence is very low (< 30%), warn user
            if charge_confidence < 0.3:
                logger.warning(
                    "Low confidence charge detection! Peak may be weak, noisy, or overlapping. Consider using 'Re-analyze with new z' if results seem incorrect"
                )

        # Get resolution before calculating compositions
        resolution = int(data.get('resolution', DEFAULT_RESOLUTION))

        # STEP 0.5: Snap to peak apex to ensure consistent analysis
        # Find the peak maximum within ±1 m/z of clicked position
        logger.debug(f'STEP 0.5: Snap to peak apex (clicked at {peak_mz:.4f})')
        snap_window = 1.0
        snap_mask = (mz_values >= peak_mz - snap_window) & (mz_values <= peak_mz + snap_window)
        snap_mz = mz_values[snap_mask]
        snap_int = intensity_values[snap_mask]

        if len(snap_int) > 0:
            apex_idx = np.argmax(snap_int)
            peak_apex = float(snap_mz[apex_idx])
            logger.debug(f'Peak apex found: {peak_apex:.4f} m/z (shift: {abs(peak_apex - peak_mz):.4f} m/z)')
            # Use apex as center for all subsequent analysis
            peak_mz = peak_apex
        else:
            logger.debug('No data in snap window, using clicked position')

        # Check cache - if we've analyzed this exact peak before, return cached result
        # Include custom_xna is_complex flag to avoid returning non-complex cached results in complex mode
        is_complex_for_cache = custom_xna.get('is_complex', False) if custom_xna else False
        cache_key = (round(peak_mz, 3), dna_sequence, resolution, is_complex_for_cache)  # Round to 0.001 m/z precision
        if cache_key in _peak_analysis_cache:
            logger.debug(f'CACHE HIT! Returning cached result for peak {peak_mz:.4f} m/z')
            cached_data = _peak_analysis_cache[cache_key]
            try:
                result_json = jsonify(cached_data)
                return result_json
            except Exception as e:
                logger.error(f'Error serializing cached data: {e}', exc_info=True)
                # Clear bad cache entry and fall through to recompute
                del _peak_analysis_cache[cache_key]
                logger.debug('Cleared bad cache entry, will recompute')

        # STEP 1: Generate DISPLAY smooth envelope FIRST (this is what user sees)
        logger.debug(f'STEP 1: Generate DISPLAY smooth envelope (centered at {peak_mz:.4f})')
        window = 3.0  # ±3 m/z window for display (same as backup version)
        mask = (mz_values >= peak_mz - window) & (mz_values <= peak_mz + window)
        exp_mz_window = mz_values[mask]
        exp_int_window = intensity_values[mask]

        logger.debug(
            f'Using DISPLAY window: [{peak_mz - window:.4f}, {peak_mz + window:.4f}] m/z ({len(exp_mz_window)} points)'
        )

        exp_gaussian_mz = []
        exp_gaussian_intensity = []
        exp_x0 = None
        exp_sigma = None

        if len(exp_mz_window) > 0:
            # Generate smooth Gaussian envelope
            exp_mz_gauss, exp_int_gauss = analyzer.generate_experimental_gaussian_envelope(
                exp_mz_window, exp_int_window, resolution
            )
            if exp_mz_gauss is not None and exp_int_gauss is not None and len(exp_mz_gauss) > 3:
                # Check if envelope is truly flat (no isotope structure at all)
                mean_int_window = float(np.mean(exp_int_window))
                mean_int_gauss = float(np.mean(exp_int_gauss))
                raw_cv = float(np.std(exp_int_window)) / mean_int_window if mean_int_window > 0 else 0.0
                envelope_cv = float(np.std(exp_int_gauss)) / mean_int_gauss if mean_int_gauss > 0 else 0.0

                # Only flag as flat if the envelope itself has almost no variation (< 5%)
                is_envelope_flat = envelope_cv < 0.05

                if is_envelope_flat:
                    logger.debug(
                        f'Envelope too flat (raw_cv={raw_cv * 100:.1f}%, envelope_cv={envelope_cv * 100:.1f}%) - not displaying envelope or X0'
                    )
                    # Keep exp_gaussian_mz, exp_gaussian_intensity as empty lists
                    # Keep exp_x0, exp_sigma as None
                else:
                    # Fit Gaussian using gaussian_fit_centroid (same method as custom search)
                    fit_result = analyzer.gaussian_fit_centroid(exp_mz_gauss, exp_int_gauss)
                    if fit_result and fit_result[0] is not None:
                        exp_x0 = fit_result[0]
                        exp_sigma = fit_result[1]
                    else:
                        exp_x0, exp_sigma = None, None
                    # Keep original envelope for display - it has the correct shape
                    exp_gaussian_mz = exp_mz_gauss.tolist()
                    exp_gaussian_intensity = exp_int_gauss.tolist()

        # Calculate peak symmetry
        symmetry_info = analyzer.calculate_peak_symmetry(
            mz_values=mz_values, intensity_values=intensity_values, center_mz=peak_mz, window=2.0
        )

        # STEP 3: Calculate compositions and filter using the CORRECT X₀ from narrow envelope
        # If envelope is flat, fallback to peak_mz
        if exp_x0 is None:
            exp_x0 = peak_mz
            logger.debug(f'STEP 3: Calculate compositions using X₀={exp_x0:.4f} (fallback to clicked peak)')
        else:
            logger.debug(f'STEP 3: Calculate compositions using X₀={exp_x0:.4f}')
        if custom_xna and custom_xna.get('formula'):
            # Use user-provided molecular weight if available, otherwise calculate from formula
            xna_mass = custom_xna.get('molecular_weight')
            if xna_mass is None:
                xna_mass = analyzer.calculate_mass_from_formula(custom_xna['formula'])
                logger.info(
                    f'Using custom XNA: {custom_xna["name"]} (Formula: {custom_xna["formula"]}, Calculated Mass: {xna_mass:.2f} Da)'
                )
            else:
                logger.info(
                    f'Using custom XNA: {custom_xna["name"]} (Formula: {custom_xna["formula"]}, User-Provided Mass: {xna_mass:.2f} Da)'
                )
        elif custom_xna and custom_xna.get('is_complex'):
            # DNA-only Complex mode - no XNA formula, will use DNA sequence for mass calculation
            logger.info(
                f'Complex DNA mode (no XNA formula) - is_complex={custom_xna.get("is_complex")}, using DNA sequence for mass calculation'
            )
        # Use smart adduct search (mass-based filtering, triggered by X₀ error > 0.5)
        compositions = analyzer.analyze_peak_with_smart_adduct_search(
            peak_mz,
            charge,
            dna_sequence,
            exp_x0,
            resolution=resolution,
            mz_values=mz_values,
            intensity_values=intensity_values,
            custom_xna=custom_xna,
        )

        # Refine with isotope matching - this will filter based on CORRECT X₀
        has_other_strands = False
        all_compositions = []
        if len(compositions) > 0:
            (
                compositions,
                exp_x0_refined,
                exp_sigma_refined,
                has_other_strands,
                all_compositions,
                has_odd_n0_warning,
                _,
                _,
                has_unrealistic_n0_warning,
            ) = analyzer.refine_compositions_with_isotope_matching(
                compositions,
                mz_values,
                intensity_values,
                peak_mz,
                resolution=resolution,
                detected_centroid=exp_x0,  # Use DISPLAY envelope X₀
            )
            # Keep the display envelope X₀ (don't overwrite with narrow envelope)
            # exp_x0 stays as the display envelope value
        else:
            has_odd_n0_warning = False
            has_unrealistic_n0_warning = False

        # Calculate composition estimates (for when no auto compositions found)
        composition_estimates = []
        is_complex = custom_xna.get('is_complex', False) if custom_xna else False
        if len(compositions) == 0 and charge is not None:
            composition_estimates = analyzer.calculate_composition_estimates(
                peak_mz, charge, dna_sequence, custom_xna=custom_xna, max_strands=MAX_STRANDS * 2, is_complex=is_complex
            )
            logger.debug(
                f'Calculated {len(composition_estimates)} composition estimates for user guidance (complex={is_complex})'
            )

        # Convert all NumPy types to native Python types for JSON serialization
        result = {
            'clicked_mz': clicked_mz,
            'peak_mz': float(peak_mz),
            'intensity': float(max_int),
            'charge': charge,
            'charge_method': charge_method,
            'charge_confidence': float(charge_confidence) if charge_confidence is not None else None,
            'compositions': compositions,
            'composition_estimates': composition_estimates,
            'region_mz_range': [float(region_mz[0]), float(region_mz[-1])],
            'exp_x0': float(exp_x0) if exp_x0 is not None else None,
            'exp_sigma': float(exp_sigma) if exp_sigma is not None else None,
            'symmetry': symmetry_info,
            'has_other_strands': has_other_strands,
            'all_compositions': all_compositions,
            'has_odd_n0_warning': has_odd_n0_warning,
            'has_unrealistic_n0_warning': has_unrealistic_n0_warning,
            'exp_gaussian_mz': exp_gaussian_mz,
            'exp_gaussian_intensity': exp_gaussian_intensity,
            'is_complex': is_complex,
        }

        logger.debug(f'Sending response: exp_gaussian_mz={len(exp_gaussian_mz)} points')

        # Convert to JSON-serializable format and cache the result
        json_result = convert_numpy_types(result)
        _peak_analysis_cache[cache_key] = json_result
        logger.debug(f'Cached result for peak {peak_mz:.4f} m/z (cache size: {len(_peak_analysis_cache)})')

        return jsonify(json_result)

    except Exception as e:
        logger.error(
            f'ANALYSIS_CRASH in analyze_region: Input={data.get("clicked_mz", "unknown")}, Error={type(e).__name__}: {str(e)}',
            exc_info=True,
        )
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_peak', methods=['POST'])
@check_same_origin
def analyze_peak() -> FlaskResponse:
    """Analyze a specific peak for composition (legacy endpoint)"""
    # Rate limiting check
    if not check_rate_limit(request.remote_addr):
        return jsonify({'error': 'Rate limit exceeded. Please wait before making more requests.'}), 429
    try:
        data = request.get_json()
        analyzer = DNASilverAnalyzer()
        analyzer.custom_adducts = data.get('custom_adducts', []) or []
        mz = float(data.get('mz'))
        charge = int(data.get('charge', 1))
        dna_sequence = data.get('dna_sequence', None)
        resolution = int(data.get('resolution', DEFAULT_RESOLUTION))

        # For now, without neighboring peaks, assume charge from input
        compositions = analyzer.calculate_dna_silver_composition(mz, charge, dna_sequence, resolution=resolution)

        return jsonify({'mz': mz, 'charge': charge, 'compositions': compositions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_all_adducts', methods=['GET'])
def get_all_adducts() -> FlaskResponse:
    """Get list of ALL adducts (built-in + custom) from the library"""
    try:
        # Build list of all adducts with their properties
        all_adducts = []
        for name, (mass, charge) in analyzer.adducts.items():
            all_adducts.append({'name': name, 'mass': mass, 'charge': charge})

        # Sort by name for easier selection
        all_adducts.sort(key=lambda x: x['name'])

        return jsonify({'adducts': all_adducts, 'total_count': len(all_adducts)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/parse_adduct_formula', methods=['POST'])
def parse_adduct_formula() -> FlaskResponse:
    """Stateless: validate name + formula/mass + charge and return {mass, charge}.

    Custom adducts live in the client's localStorage, not on the server, so
    different HF Spaces users never share or overwrite each other's lists.
    """
    try:
        data = request.get_json()
        name = (data.get('name') or '').strip()
        formula = (data.get('formula') or '').strip()
        mass = data.get('mass')
        charge = data.get('charge')

        if not name:
            return jsonify({'success': False, 'error': 'Adduct name is required'}), 400
        if charge is None:
            return jsonify({'success': False, 'error': 'Charge is required'}), 400
        try:
            charge = int(charge)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Charge must be an integer'}), 400

        if not formula and mass in (None, ''):
            return jsonify({'success': False, 'error': 'Formula or mass is required'}), 400

        computed_mass = None
        resolved_formula = None
        if formula:
            ok, calc_mass, err = analyzer.calculate_mass_from_formula_with_validation(formula)
            if ok:
                computed_mass = calc_mass
                resolved_formula = formula
            else:
                try:
                    computed_mass = float(formula)
                except ValueError:
                    return jsonify({'success': False, 'error': err or f'Invalid formula: {formula}'}), 400
        else:
            try:
                computed_mass = float(mass)
            except (TypeError, ValueError):
                return jsonify({'success': False, 'error': 'Invalid mass value'}), 400

        if computed_mass is None or computed_mass < 0.1 or computed_mass > 10000:
            return jsonify({'success': False, 'error': 'Mass must be between 0.1 and 10000 Da'}), 400

        if charge < -5 or charge > 5:
            return jsonify({'success': False, 'error': 'Charge must be between -5 and +5'}), 400

        return jsonify(
            {
                'success': True,
                'name': name,
                'formula': resolved_formula,
                'mass': computed_mass,
                'charge': charge,
            }
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/smiles_to_formula', methods=['POST'])
def smiles_to_formula() -> FlaskResponse:
    """Convert SMILES string to molecular formula using RDKit"""
    try:
        data = request.get_json()
        smiles = data.get('smiles', '').strip()

        if not smiles:
            return jsonify({'error': 'No SMILES string provided'}), 400

        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors

            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return jsonify({'error': f'Invalid SMILES string: {smiles}'}), 400

            # Add hydrogens to get complete formula
            mol = Chem.AddHs(mol)

            # Get molecular formula
            formula = rdMolDescriptors.CalcMolFormula(mol)

            # Get molecular weight for reference
            mol_weight = rdMolDescriptors.CalcExactMolWt(mol)

            return jsonify({'formula': formula, 'smiles': smiles, 'molecular_weight': mol_weight})

        except ImportError:
            # RDKit not installed - provide fallback message
            return jsonify({'error': 'RDKit is not installed. Please install it with: pip install rdkit'}), 500

    except Exception as e:
        logger.error(f'Error in smiles_to_formula: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/isotope_library', methods=['GET', 'POST'])
def isotope_library_settings() -> FlaskResponse:
    """
    GET: Return current isotope library settings
    POST: Switch isotope library (isospec or pythoms)
    """
    global ISOTOPE_LIBRARY, _isotope_pattern_cache

    if request.method == 'GET':
        return jsonify(
            {
                'current_library': ISOTOPE_LIBRARY,
                'isospec_available': ISOSPEC_AVAILABLE,
                'cache_size': len(_isotope_pattern_cache),
            }
        )

    # POST - switch library
    try:
        data = request.get_json()
        new_library = data.get('library', '').lower()

        if new_library not in ['isospec', 'pythoms']:
            return jsonify({'error': 'Invalid library. Use "isospec" or "pythoms"'}), 400

        if new_library == 'isospec' and not ISOSPEC_AVAILABLE:
            return jsonify({'error': 'IsoSpecPy is not installed. Install with: pip install IsoSpecPy'}), 400

        old_library = ISOTOPE_LIBRARY
        ISOTOPE_LIBRARY = new_library

        # Clear cache when switching libraries to ensure consistency
        cache_size = len(_isotope_pattern_cache)
        _isotope_pattern_cache.clear()

        logger.info(f'Switched isotope library: {old_library} → {new_library} (cleared {cache_size} cached patterns)')

        return jsonify(
            {'success': True, 'old_library': old_library, 'new_library': new_library, 'cache_cleared': cache_size}
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/benchmark_isotope', methods=['POST'])
def benchmark_isotope() -> FlaskResponse:
    """
    Benchmark isotope pattern generation with both libraries.
    Returns timing comparison.
    """
    import time

    try:
        data = request.get_json()
        formula = data.get('formula', 'C200H280N80O120P20Ag16')
        charge = int(data.get('charge', 4))
        resolution = int(data.get('resolution', DEFAULT_RESOLUTION))
        iterations = int(data.get('iterations', 5))

        benchmark_analyzer = DNASilverAnalyzer()
        results: dict[str, Any] = {}

        # Clear cache before benchmarking
        global _isotope_pattern_cache
        _isotope_pattern_cache.clear()

        # Benchmark PythoMS
        start = time.time()
        for _ in range(iterations):
            _isotope_pattern_cache.clear()  # Clear cache each iteration
            pattern_pythoms = benchmark_analyzer._generate_isotope_pattern_pythoms(formula, charge, resolution)
        pythoms_time = (time.time() - start) / iterations
        results['pythoms'] = {'time_per_call': pythoms_time, 'success': 'error' not in pattern_pythoms}

        # Benchmark IsoSpecPy if available
        if ISOSPEC_AVAILABLE:
            start = time.time()
            for _ in range(iterations):
                _isotope_pattern_cache.clear()
                pattern_isospec = benchmark_analyzer._generate_isotope_pattern_isospec(formula, charge, resolution)
            isospec_time = (time.time() - start) / iterations
            results['isospec'] = {'time_per_call': isospec_time, 'success': 'error' not in pattern_isospec}

            if results['pythoms']['success'] and results['isospec']['success']:
                results['speedup'] = pythoms_time / isospec_time

                # Compare X0 values
                pythoms_mz = np.array(pattern_pythoms['mz'])
                pythoms_int = np.array(pattern_pythoms['intensity'])
                isospec_mz = np.array(pattern_isospec['mz'])
                isospec_int = np.array(pattern_isospec['intensity'])

                pythoms_x0 = np.average(pythoms_mz, weights=pythoms_int)
                isospec_x0 = np.average(isospec_mz, weights=isospec_int)

                results['x0_comparison'] = {
                    'pythoms_x0': pythoms_x0,
                    'isospec_x0': isospec_x0,
                    'difference': abs(pythoms_x0 - isospec_x0),
                }
        else:
            results['isospec'] = {'error': 'IsoSpecPy not installed'}

        results['formula'] = formula
        results['charge'] = charge
        results['iterations'] = iterations

        return jsonify(convert_numpy_types(results))

    except Exception as e:
        logger.error(f'Error in benchmark_isotope: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Check if port 8080 is available
    import socket

    def is_port_in_use(port: int) -> bool:
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                return result == 0  # True if port is in use
            except Exception:
                return False

    port = int(os.environ.get('PORT', 8080))

    logger.info('Checking for existing server instances...')
    if is_port_in_use(port):
        logger.warning(
            f'Port {port} is already in use! Another server instance may be running. Please stop it first or use a different port.'
        )
        import sys

        sys.exit(1)
    else:
        logger.info('No existing instances found')

    # Debug mode: OFF by default for security. Enable for development only:
    #   export FLASK_DEBUG=1  (on Mac/Linux)
    #   set FLASK_DEBUG=1     (on Windows)
    debug_mode = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')

    logger.info('Starting DNA-stabilized Silver Nanocluster Mass Spec Analysis Web Server...')
    logger.info(f'Open your browser to http://localhost:{port}')
    if debug_mode:
        logger.warning('Debug mode: ON (for development only, not secure for production)')
    else:
        logger.info('Debug mode: OFF (production-safe)')
    logger.info('Press CTRL+C to stop the server')

    # Enable threaded mode and use_reloader=False to avoid port conflicts
    app.run(debug=debug_mode, host='0.0.0.0', port=port, threaded=True, use_reloader=False)
