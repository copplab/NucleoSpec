"""
Automated Assignment of Charge States from Resolved Isotopic Peaks

Implementation of methods from:
Senko, M.W., Beu, S.C., and McLafferty, F.W. (1995)
"Automated Assignment of Charge States from Resolved Isotopic Peaks for Multiply Charged Ions"
J. Am. Soc. Mass Spectrom., 6, 52-56

This module provides three complementary algorithms for charge state determination:
1. Patterson Function - Best for low charge states (z < 5) with high S/N
2. Fourier Transform - Best for high charge states (z > 5) with low resolving power
3. Combination Method - Multiplies Patterson × Fourier (recommended for all cases)

The methods achieved >95% accuracy in the original paper and work even when
isotope clusters overlap.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def patterson_function(mz_array, intensity_array, charge_range=(1, 10), step_size=1/3):
    """
    Patterson function for charge state determination.

    Best for low charge states (z < 5) with high S/N and resolving power.

    From the paper:
    P(ΔM) = Σ f(Mi - ΔM/2) * f(Mi + ΔM/2)

    where ΔM is the inverse of the charge being evaluated.

    Parameters:
    -----------
    mz_array : np.ndarray
        m/z values of the isotope envelope
    intensity_array : np.ndarray
        Intensity values
    charge_range : tuple
        (min_charge, max_charge) to test
    step_size : float
        Step size for charge evaluation (default 1/3 for smooth maps)

    Returns:
    --------
    charges : np.ndarray
        Array of charge values tested
    patterson_map : np.ndarray
        Patterson function values for each charge
    """
    min_z, max_z = charge_range

    # Create interpolation function for intensity
    # Use linear interpolation between data points
    interp_func = interp1d(mz_array, intensity_array, kind='linear',
                           bounds_error=False, fill_value=0.0)

    # Generate charge values to test (with fractional steps for smooth map)
    charges = np.arange(min_z - 1/3, max_z + 1, step_size)
    patterson_map = np.zeros(len(charges))

    for idx, z in enumerate(charges):
        if z < 1:
            continue

        delta_m = 1.0 / z  # Spacing for this charge state

        # Calculate Patterson function
        # Sum over all m/z points
        patterson_sum = 0.0
        for mz in mz_array:
            # Get intensities at mz - delta_m/2 and mz + delta_m/2
            I_minus = interp_func(mz - delta_m / 2)
            I_plus = interp_func(mz + delta_m / 2)
            patterson_sum += I_minus * I_plus

        patterson_map[idx] = patterson_sum

    return charges, patterson_map


def fourier_function(mz_array, intensity_array, charge_range=(1, 10)):
    """
    Fourier transform method for charge state determination.

    Best for high charge states (z > 5) with low resolving power.
    Produces sharper peaks than Patterson method.

    The FFT considers isotopic peaks in terms of their frequency of occurrence,
    not their spacing. The repetitive spacing produces a maximum in the frequency domain.

    Parameters:
    -----------
    mz_array : np.ndarray
        m/z values of the isotope envelope
    intensity_array : np.ndarray
        Intensity values
    charge_range : tuple
        (min_charge, max_charge) to test

    Returns:
    --------
    charges : np.ndarray
        Array of charge values
    fourier_map : np.ndarray
        Fourier transform magnitude for each charge
    """
    min_z, max_z = charge_range

    # Baseline correction - subtract minimum
    baseline = np.min(intensity_array)
    corrected_intensity = intensity_array - baseline

    # Pad data to next power of 2 for efficient FFT
    n_points = len(corrected_intensity)
    n_padded = 2 ** int(np.ceil(np.log2(n_points)))
    padded_intensity = np.zeros(n_padded)
    padded_intensity[:n_points] = corrected_intensity

    # Perform FFT
    fft_result = np.fft.fft(padded_intensity)
    fft_magnitude = np.abs(fft_result)

    # Get frequency axis
    # The m/z spacing
    mz_spacing = np.mean(np.diff(mz_array))
    frequencies = np.fft.fftfreq(n_padded, d=mz_spacing)

    # Convert frequencies to charge states
    # Isotope spacing = 1.003 / z (approximately 1/z)
    # Frequency = 1 / spacing = z / 1.003
    # So: z ≈ frequency * 1.003

    # Map FFT results to charge states
    charges = np.arange(min_z, max_z + 1)
    fourier_map = np.zeros(len(charges))

    for idx, z in enumerate(charges):
        # Expected frequency for this charge
        expected_freq = z / 1.003

        # Find closest frequency in FFT
        freq_idx = np.argmin(np.abs(frequencies - expected_freq))
        fourier_map[idx] = fft_magnitude[freq_idx]

    return charges, fourier_map


def combination_function(mz_array, intensity_array, charge_range=(1, 10)):
    """
    Combination method: Patterson × Fourier.

    RECOMMENDED for all cases. Achieves >95% accuracy.

    From the paper:
    C(z) = F(z) * P(z)

    Only the true maximum should be present in both maps, and thus should be
    most abundant in the combination map. This reduces false maxima from both methods.

    Parameters:
    -----------
    mz_array : np.ndarray
        m/z values of the isotope envelope
    intensity_array : np.ndarray
        Intensity values
    charge_range : tuple
        (min_charge, max_charge) to test

    Returns:
    --------
    charges : np.ndarray
        Array of charge values
    combination_map : np.ndarray
        Combined Patterson × Fourier values
    patterson_map : np.ndarray
        Patterson function values
    fourier_map : np.ndarray
        Fourier transform values
    """
    # Get Patterson map
    charges_p, patterson_map = patterson_function(mz_array, intensity_array, charge_range)

    # Get Fourier map (interpolate to match Patterson charges)
    charges_f, fourier_map_raw = fourier_function(mz_array, intensity_array, charge_range)

    # Interpolate Fourier to match Patterson charge grid
    fourier_interp = interp1d(charges_f, fourier_map_raw, kind='linear',
                              bounds_error=False, fill_value=0.0)
    fourier_map = fourier_interp(charges_p)

    # Normalize both maps to [0, 1]
    if np.max(patterson_map) > 0:
        patterson_norm = patterson_map / np.max(patterson_map)
    else:
        patterson_norm = patterson_map

    if np.max(fourier_map) > 0:
        fourier_norm = fourier_map / np.max(fourier_map)
    else:
        fourier_norm = fourier_map

    # Multiply the two maps
    combination_map = patterson_norm * fourier_norm

    return charges_p, combination_map, patterson_norm, fourier_norm


def find_envelope_boundaries(mz_array, intensity_array, valley_threshold=0.02):
    """
    Find isotope envelope boundaries by locating global apex and global valleys.

    Algorithm:
    1. Find the global apex (highest point)
    2. Smooth the signal to get envelope shape (ignore local isotope oscillations)
    3. Go left/right from apex until smoothed intensity drops below threshold

    This finds the true envelope boundaries, not local valleys between isotope peaks.

    Parameters:
    -----------
    mz_array : np.ndarray
        m/z values
    intensity_array : np.ndarray
        Intensity values
    valley_threshold : float
        Valley is found when intensity drops below this fraction of max (default 0.02 = 2%)

    Returns:
    --------
    dict with:
        - 'global_apex_idx': int, index of global apex
        - 'left_valley_idx': int, index of left boundary
        - 'right_valley_idx': int, index of right boundary
        - 'envelope_mz': np.ndarray, m/z values within envelope
        - 'envelope_intensity': np.ndarray, intensity values within envelope
    """
    if len(mz_array) < 3:
        return {
            'global_apex_idx': 0,
            'left_valley_idx': 0,
            'right_valley_idx': len(mz_array) - 1,
            'envelope_mz': mz_array,
            'envelope_intensity': intensity_array
        }

    # Find global apex
    global_apex_idx = np.argmax(intensity_array)
    max_intensity = intensity_array[global_apex_idx]
    threshold = max_intensity * valley_threshold

    # Smooth the signal to find envelope shape
    # Use a wider window to smooth over isotope peak oscillations
    mz_span = mz_array[-1] - mz_array[0]
    points_per_mz = len(mz_array) / mz_span if mz_span > 0 else 10
    window_size = max(5, int(points_per_mz * 0.5))  # ~0.5 m/z window
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, len(intensity_array) // 3)  # Don't make window too big

    # Pad and smooth using convolution
    half_win = window_size // 2
    padded = np.pad(intensity_array, half_win, mode='edge')
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode='valid')

    # Ensure smoothed is same length as input
    if len(smoothed) > len(intensity_array):
        smoothed = smoothed[:len(intensity_array)]
    elif len(smoothed) < len(intensity_array):
        smoothed = np.pad(smoothed, (0, len(intensity_array) - len(smoothed)), mode='edge')

    # Go left to find left valley (using smoothed signal)
    left_valley_idx = 0
    for i in range(global_apex_idx - 1, -1, -1):
        if smoothed[i] < threshold:
            left_valley_idx = i
            break

    # Go right to find right valley (using smoothed signal)
    right_valley_idx = len(intensity_array) - 1
    for i in range(global_apex_idx + 1, len(intensity_array)):
        if smoothed[i] < threshold:
            right_valley_idx = i
            break

    return {
        'global_apex_idx': global_apex_idx,
        'left_valley_idx': left_valley_idx,
        'right_valley_idx': right_valley_idx,
        'envelope_mz': mz_array[left_valley_idx:right_valley_idx+1],
        'envelope_intensity': intensity_array[left_valley_idx:right_valley_idx+1]
    }


def extract_apexes(mz_array, intensity_array, min_prominence_ratio=0.05):
    """
    Extract local maxima (apexes) from a spectrum region.

    These apexes represent the individual isotope peaks within an envelope.
    Using apexes instead of raw data improves charge detection for complex
    spectra like Duplex DNA where broad envelopes can confuse the algorithms.

    Parameters:
    -----------
    mz_array : np.ndarray
        m/z values of the region
    intensity_array : np.ndarray
        Intensity values of the region
    min_prominence_ratio : float
        Minimum prominence as fraction of max intensity (default 0.05 = 5%)

    Returns:
    --------
    apex_mz : np.ndarray
        m/z values of the apexes
    apex_intensity : np.ndarray
        Intensity values of the apexes
    apex_indices : np.ndarray
        Indices of apexes in the original arrays
    """
    if len(mz_array) < 3:
        return mz_array, intensity_array, np.arange(len(mz_array))

    max_intensity = np.max(intensity_array)
    min_prominence = max_intensity * min_prominence_ratio

    # Find local maxima with sufficient prominence
    apex_indices, properties = find_peaks(
        intensity_array,
        prominence=min_prominence,
        distance=2
    )

    # If no apexes found, fall back to using the maximum point
    if len(apex_indices) == 0:
        max_idx = np.argmax(intensity_array)
        apex_indices = np.array([max_idx])

    apex_mz = mz_array[apex_indices]
    apex_intensity = intensity_array[apex_indices]

    return apex_mz, apex_intensity, apex_indices


def assign_charge_senko(mz_array, intensity_array, charge_range=(1, 10),
                        method='combination', return_all_maps=False):
    """
    Assign charge state using Senko et al. 1995 methods.

    This is the main function to call for charge state assignment.

    Parameters:
    -----------
    mz_array : np.ndarray
        m/z values of the isotope envelope
    intensity_array : np.ndarray
        Intensity values
    charge_range : tuple
        (min_charge, max_charge) to test
    method : str
        'patterson', 'fourier', or 'combination' (recommended)
    return_all_maps : bool
        If True, return all charge maps for visualization

    Returns:
    --------
    dict with keys:
        - 'charge': int, assigned charge state
        - 'confidence': float, normalized score for assigned charge
        - 'method': str, method used
        - 'charge_map': dict with charges and scores (if return_all_maps=True)
    """
    if len(mz_array) < 2:
        return {
            'charge': None,
            'confidence': 0.0,
            'method': method,
            'error': 'Insufficient data points'
        }

    # Choose method
    if method == 'patterson':
        charges, charge_map = patterson_function(mz_array, intensity_array, charge_range)
    elif method == 'fourier':
        charges, charge_map = fourier_function(mz_array, intensity_array, charge_range)
    elif method == 'combination':
        charges, charge_map, patterson_map, fourier_map = combination_function(
            mz_array, intensity_array, charge_range
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find charge with maximum score
    max_idx = np.argmax(charge_map)
    assigned_charge = charges[max_idx]

    # Round to nearest integer
    assigned_charge = int(round(assigned_charge))

    # Calculate confidence (normalized score)
    if np.max(charge_map) > 0:
        confidence = charge_map[max_idx] / np.max(charge_map)
    else:
        confidence = 0.0

    result = {
        'charge': assigned_charge,
        'confidence': float(confidence),
        'method': method
    }

    if return_all_maps:
        result['charge_map'] = {
            'charges': charges.tolist(),
            'scores': charge_map.tolist()
        }
        if method == 'combination':
            result['patterson_map'] = patterson_map.tolist()
            result['fourier_map'] = fourier_map.tolist()

    return result


def extract_isotope_envelope(mz_array, intensity_array, peak_mz, window=2.0):
    """
    Extract an isotope envelope around a peak for charge state analysis.

    Parameters:
    -----------
    mz_array : np.ndarray
        Full m/z array
    intensity_array : np.ndarray
        Full intensity array
    peak_mz : float
        Center m/z of the peak
    window : float
        Window size in m/z units (±window from peak_mz)

    Returns:
    --------
    envelope_mz : np.ndarray
        m/z values in the envelope
    envelope_intensity : np.ndarray
        Intensity values in the envelope
    """
    # Find region around peak
    mask = (mz_array >= peak_mz - window) & (mz_array <= peak_mz + window)
    envelope_mz = mz_array[mask]
    envelope_intensity = intensity_array[mask]

    return envelope_mz, envelope_intensity


def find_peak_regions(mz_values, intensity_values, threshold=0.05, merge_gap=1.5):
    """
    Find isotope envelope regions using LOCAL MAXIMA detection.

    Parameters:
    -----------
    mz_values : np.ndarray
        m/z values
    intensity_values : np.ndarray
        Intensity values
    threshold : float
        Relative intensity threshold (0-1) - peaks below this are ignored
    merge_gap : float
        Merge regions separated by less than this m/z (same isotope envelope)

    Returns:
    --------
    list of tuples (start_idx, end_idx) for each region
    """
    if len(mz_values) < 5:
        return []

    max_intensity = np.max(intensity_values)
    mz_spacing = np.median(np.diff(mz_values))

    # Estimate noise floor from the spectrum median (baseline)
    noise_floor = np.median(intensity_values)
    noise_threshold = noise_floor * 3  # 3× median as noise cutoff

    print(f"[find_peak_regions] Max intensity: {max_intensity:.0f}, noise floor: {noise_floor:.0f}, noise threshold: {noise_threshold:.0f}, mz_spacing: {mz_spacing:.4f}")

    min_height = max(max_intensity * threshold, noise_threshold)
    min_prominence = max(min_height * 0.5, noise_threshold)
    min_distance = max(10, int(10.0 / mz_spacing))

    # Find peaks
    peak_indices, properties = find_peaks(
        intensity_values,
        height=min_height,
        prominence=min_prominence,
        distance=min_distance
    )

    print(f"[find_peak_regions] height_threshold={min_height:.0f}, distance={min_distance} indices")
    print(f"[find_peak_regions] Found {len(peak_indices)} peaks above threshold")
    if len(peak_indices) > 0:
        # Show top 5 peaks by intensity
        peak_ints = intensity_values[peak_indices]
        top_5_idx = np.argsort(peak_ints)[-5:][::-1]  # Get indices of top 5
        print(f"[find_peak_regions] Top peaks: ", end="")
        for i in top_5_idx:
            if i < len(peak_indices):
                mz = mz_values[peak_indices[i]]
                inten = intensity_values[peak_indices[i]]
                print(f"m/z={mz:.1f}(I={inten:.0f}), ", end="")
        print()

    if len(peak_indices) == 0:
        # Fallback: try with lower requirements
        peak_indices, properties = find_peaks(
            intensity_values,
            height=min_height * 0.5,
            prominence=min_prominence * 0.5,
            distance=min_distance // 2
        )

    if len(peak_indices) == 0:
        return []

    # For each detected peak, create a region around it (±5 m/z window)
    # This captures the isotope envelope while avoiding merging nearby envelopes
    envelope_half_width = 5.0  # m/z
    envelope_half_idx = int(envelope_half_width / mz_spacing)

    regions = []
    for peak_idx in peak_indices:
        left_idx = max(0, peak_idx - envelope_half_idx)
        right_idx = min(len(mz_values) - 1, peak_idx + envelope_half_idx)
        regions.append((left_idx, right_idx))

    if len(regions) <= 1:
        return regions

    # Merge overlapping or close regions (same isotope envelope)
    regions.sort(key=lambda x: x[0])

    merged_regions = []
    current_start, current_end = regions[0]

    for i in range(1, len(regions)):
        next_start, next_end = regions[i]

        # Check for overlap or small gap
        gap = mz_values[next_start] - mz_values[current_end] if next_start > current_end else 0

        if next_start <= current_end or gap < merge_gap:
            # Merge: extend current region
            current_end = max(current_end, next_end)
        else:
            # Save current region and start new one
            merged_regions.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged_regions.append((current_start, current_end))

    # print(f"[find_peak_regions] After merging: {len(merged_regions)} regions")
    # for i, (s, e) in enumerate(merged_regions[:5]):  # Print first 5
    #     print(f"  Region {i+1}: m/z {mz_values[s]:.1f} - {mz_values[e]:.1f}")

    return merged_regions


def weighted_centroid(mz_values, intensity_values, start_idx, end_idx):
    """
    Calculate peak centroid (m/z at maximum intensity).

    Returns:
    --------
    centroid_mz : float
        m/z at maximum intensity
    max_intensity : float
        Maximum intensity in the region
    """
    region_mz = mz_values[start_idx:end_idx+1]
    region_int = intensity_values[start_idx:end_idx+1]

    if len(region_mz) == 0 or np.sum(region_int) == 0:
        return None, None

    # Find the m/z at maximum intensity (peak apex)
    max_idx = np.argmax(region_int)
    centroid_mz = region_mz[max_idx]
    max_intensity = region_int[max_idx]

    return centroid_mz, max_intensity


def measure_direct_spacing(mz_array, intensity_array):
    """
    Determine charge by counting apexes in a 1 m/z window.

    Simple and robust approach: since isotope spacing = 1.003/z,
    the number of isotope peaks in a 1 m/z window equals the charge state.

    Filters out noise spikes (peaks too close together) before counting.

    Returns:
        dict with 'spacing', 'charge', 'num_peaks', 'has_alternating_pattern'
    """
    if len(mz_array) < 5:
        return {'spacing': None, 'charge': None, 'num_peaks': 0, 'has_alternating_pattern': False}

    # Extract apexes (local maxima) - use lower prominence for isotope peaks
    peak_mzs, peak_ints, peaks = extract_apexes(mz_array, intensity_array, min_prominence_ratio=0.02)

    if len(peaks) < 3:
        return {'spacing': None, 'charge': None, 'num_peaks': len(peaks), 'has_alternating_pattern': False}

    # Filter out noise spikes: peaks too close together (< 0.08 m/z) are likely noise
    # For z=10, spacing would be ~0.1 m/z, so 0.08 is a safe minimum
    MIN_SPACING = 0.08
    filtered_mzs = [peak_mzs[0]]
    filtered_ints = [peak_ints[0]]

    for i in range(1, len(peak_mzs)):
        spacing = peak_mzs[i] - filtered_mzs[-1]
        if spacing >= MIN_SPACING:
            # Normal spacing - keep this peak
            filtered_mzs.append(peak_mzs[i])
            filtered_ints.append(peak_ints[i])
        else:
            # Too close - keep the more intense one
            if peak_ints[i] > filtered_ints[-1]:
                filtered_mzs[-1] = peak_mzs[i]
                filtered_ints[-1] = peak_ints[i]

    filtered_mzs = np.array(filtered_mzs)
    filtered_ints = np.array(filtered_ints)

    if len(filtered_mzs) < 3:
        return {'spacing': None, 'charge': None, 'num_peaks': len(filtered_mzs), 'has_alternating_pattern': False}

    # COUNT APEXES IN 1 m/z WINDOW to determine charge
    # Use multiple 1 m/z windows and take the most common count
    mz_min = filtered_mzs[0]
    mz_max = filtered_mzs[-1]
    mz_span = mz_max - mz_min

    if mz_span < 1.0:
        # Envelope too small - count all peaks as the charge estimate
        charge = len(filtered_mzs)
        return {
            'spacing': 1.003 / charge if charge > 0 else None,
            'charge': charge,
            'num_peaks': len(filtered_mzs),
            'has_alternating_pattern': False
        }

    # Sample multiple 1 m/z windows centered at different positions
    window_counts = []
    step = 0.2  # Step through the envelope

    for start_mz in np.arange(mz_min, mz_max - 1.0 + step, step):
        end_mz = start_mz + 1.0
        # Count peaks in this 1 m/z window
        count = np.sum((filtered_mzs >= start_mz) & (filtered_mzs <= end_mz))
        if count >= 1:
            window_counts.append(count)

    if len(window_counts) == 0:
        return {'spacing': None, 'charge': None, 'num_peaks': len(filtered_mzs), 'has_alternating_pattern': False}

    # Use the median count (robust to outliers at edges)
    charge = int(round(np.median(window_counts)))

    # Sanity check: charge should be between 1 and 10
    charge = max(1, min(10, charge))

    print(f"  [measure_direct_spacing] Counted apexes in 1 m/z windows: {window_counts[:10]}... -> z={charge}")

    has_overlap = False
    if charge >= 4 and len(filtered_mzs) >= 6:
        # Check step-2 spacings: if peak[i+2] - peak[i] gives charge/2, two species overlap
        step2_spacings = filtered_mzs[2:] - filtered_mzs[:-2]
        step2_median = np.median(step2_spacings)
        half_charge = charge / 2.0

        if step2_median > 0:
            step2_z = 1.003 / step2_median
            spacing_ok = abs(step2_z - half_charge) < 1.0

            # Intensity balance: true overlap has comparable even/odd intensities
            even_avg = np.mean(filtered_ints[0::2])
            odd_avg = np.mean(filtered_ints[1::2])
            intensity_balance = min(even_avg, odd_avg) / max(even_avg, odd_avg) if max(even_avg, odd_avg) > 0 else 0

            # Envelope roughness: overlap creates jagged envelope, single species is smooth
            mid_ints = filtered_ints[1:-1]
            neighbor_avg = (filtered_ints[:-2] + filtered_ints[2:]) / 2
            roughness = np.mean(np.abs(mid_ints - neighbor_avg)) / np.mean(filtered_ints) if np.mean(filtered_ints) > 0 else 0

            print(f"  [overlap check] step2_z={step2_z:.1f}, half_charge={half_charge:.1f}, "
                  f"spacing_ok={spacing_ok}, intensity_balance={intensity_balance:.2f}, roughness={roughness:.2f}")

            if spacing_ok and intensity_balance > 0.3 and roughness > 0.15:
                corrected_charge = int(round(step2_z))
                corrected_charge = max(1, min(10, corrected_charge))
                print(f"  [overlap detection] Two overlapping species detected! "
                      f"step2_z={step2_z:.1f}, balance={intensity_balance:.2f}, roughness={roughness:.2f} -> corrected z={corrected_charge}")
                charge = corrected_charge
                has_overlap = True

    return {
        'spacing': 1.003 / charge,
        'charge': charge,
        'num_peaks': len(filtered_mzs),
        'has_alternating_pattern': has_overlap
    }


def detect_all_peaks_with_charge(mz_array, intensity_array,
                                  prominence=0.05, charge_range=(1, 10),
                                  method='combination', merge_gap=1.5):
    """
    Detect all isotope envelopes (peak regions) in a spectrum and assign charge states.

    Each isotope envelope (M, M+1, M+2, ...) is detected as ONE peak region
    and assigned ONE charge state.

    Parameters:
    -----------
    mz_array : np.ndarray
        Full spectrum m/z values
    intensity_array : np.ndarray
        Full spectrum intensity values
    prominence : float
        Relative intensity threshold for region detection (0-1)
    charge_range : tuple
        (min_charge, max_charge) to test
    method : str
        'patterson', 'fourier', or 'combination'
    merge_gap : float
        Merge regions separated by less than this m/z (default 1.5)

    Returns:
    --------
    list of dicts, each containing:
        - 'mz': float, peak centroid m/z
        - 'intensity': float, peak maximum intensity
        - 'charge': int, assigned charge
        - 'confidence': float, confidence score
        - 'method': str, method used
    """
    if len(mz_array) == 0:
        return []

    regions = find_peak_regions(mz_array, intensity_array, prominence, merge_gap)

    print(f"[detect_all_peaks] Found {len(regions)} initial peak regions")

    if len(regions) == 0:
        return []

    # For each region (isotope envelope), assign ONE charge
    results = []

    for start_idx, end_idx in regions:
        # Get initial region data
        region_mz = mz_array[start_idx:end_idx+1]
        region_int = intensity_array[start_idx:end_idx+1]

        # STEP 1: Find envelope boundaries using global apex → global valleys
        # This refines the region to the actual isotope envelope (removes noise)
        envelope = find_envelope_boundaries(region_mz, region_int)
        envelope_mz = envelope['envelope_mz']
        envelope_int = envelope['envelope_intensity']

        # Use envelope data for analysis (refined boundaries)
        if len(envelope_mz) >= 3:
            analysis_mz = envelope_mz
            analysis_int = envelope_int
        else:
            # Fall back to original region if envelope is too small
            analysis_mz = region_mz
            analysis_int = region_int

        # Calculate centroid from the refined envelope
        global_apex_idx = envelope['global_apex_idx']
        centroid_mz = region_mz[global_apex_idx] if global_apex_idx < len(region_mz) else None
        max_intensity = np.max(analysis_int) if len(analysis_int) > 0 else 0

        if centroid_mz is None:
            continue

        # Skip if region is too small for reliable charge assignment
        if len(analysis_mz) < 3:
            # Skip this peak - not enough data points
            print(f"  Skipping peak at m/z {centroid_mz:.2f}: only {len(analysis_mz)} data points in envelope")
            continue

        # STEP 2: Assign charge using Senko method on the refined envelope
        try:
            charge_result = assign_charge_senko(
                analysis_mz, analysis_int, charge_range, method
            )
            charge = charge_result['charge']
            confidence = charge_result['confidence']

            # STEP 3: VALIDATION using apex counting in 1 m/z window
            spacing_result = measure_direct_spacing(analysis_mz, analysis_int)
            if spacing_result['charge'] is not None and spacing_result['num_peaks'] >= 4:
                spacing_charge = spacing_result['charge']

                # If Senko gives low charge (z<=3) but apex counting gives high charge (z>=5),
                # trust the apex counting - Senko often fails on complex Ag spectra
                if charge <= 3 and spacing_charge >= 5:
                    print(f"  Apex counting correction at m/z {centroid_mz:.2f}: z={charge} -> z={spacing_charge} (Senko gave implausibly low charge)")
                    charge = spacing_charge
                    confidence = 0.85

                # If overlap detected (two interleaved species), use corrected charge
                if spacing_result.get('has_alternating_pattern') and spacing_charge != charge:
                    print(f"  Overlap correction at m/z {centroid_mz:.2f}: z={charge} -> z={spacing_charge} (two interleaved species)")
                    charge = spacing_charge
                    confidence = 0.90

            # Add ALL peaks with valid charge assignments (no confidence threshold)
            # Display confidence so users can judge reliability themselves
            if charge is not None:
                results.append({
                    'mz': float(centroid_mz),
                    'intensity': float(max_intensity),
                    'charge': charge,
                    'confidence': float(confidence),
                    'method': method
                })
                if confidence < 0.5:
                    print(f"Low confidence charge at m/z {centroid_mz:.2f}: z={charge}, confidence={confidence:.2f}")
                else:
                    print(f"Detected charge at m/z {centroid_mz:.2f}: z={charge}, confidence={confidence:.2f}")
            else:
                # Skip only if charge assignment completely failed (returned None)
                print(f"Skipping peak at m/z {centroid_mz:.2f} - charge assignment failed")

        except Exception as e:
            # Skip this peak - Senko algorithm failed with exception
            print(f"Skipping peak at m/z {centroid_mz:.2f} - Error: {e}")

    return results


# Example usage
if __name__ == '__main__':
    print("Senko Charge Assignment Module")
    print("=" * 60)
    print("Based on: Senko et al., J. Am. Soc. Mass Spectrom. 1995, 6, 52-56")
    print()

    # Simulate an isotope envelope for z=3
    # Isotope spacing = 1.003/3 ≈ 0.334 Da
    mz_sim = np.array([1000.0, 1000.334, 1000.668, 1001.002, 1001.336])
    # Gaussian-like envelope
    intensity_sim = np.array([10, 45, 100, 75, 30])

    print("Simulated isotope envelope (z=3):")
    print(f"  m/z spacing: ~{np.mean(np.diff(mz_sim)):.3f}")
    print(f"  Expected for z=3: {1.003/3:.3f}")
    print()

    # Test all three methods
    for method in ['patterson', 'fourier', 'combination']:
        result = assign_charge_senko(mz_sim, intensity_sim, charge_range=(1, 10), method=method)
        print(f"{method.capitalize()} Method:")
        print(f"  Assigned charge: {result['charge']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print()

    print("=" * 60)
    print("Module ready for integration!")
