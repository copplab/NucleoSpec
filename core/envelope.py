from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class EnvelopeMixin:
    """Mixin for Gaussian envelope generation, fitting, and peak symmetry analysis."""

    def smooth_gaussian_pattern(self, barip: list[list], fwhm: float, num_points_per_fwhm: int = 100) -> list[list]:
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

    def calculate_peak_symmetry(
        self,
        mz_values: npt.NDArray[np.float64],
        intensity_values: npt.NDArray[np.float64],
        center_mz: float,
        window: float = 2.0,
    ) -> dict:
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
            return {'symmetry_score': 0.0, 'skewness': 0.0, 'is_symmetric': False, 'note': 'Insufficient data points'}

        # Find peak apex
        max_idx = np.argmax(region_int)
        apex_mz = region_mz[max_idx]
        max_intensity = region_int[max_idx]

        # Divide into left and right sides from apex
        left_mz = region_mz[: max_idx + 1]
        left_int = region_int[: max_idx + 1]
        right_mz = region_mz[max_idx:]
        right_int = region_int[max_idx:]

        if len(left_mz) < 2 or len(right_mz) < 2:
            return {'symmetry_score': 0.0, 'skewness': 0.0, 'is_symmetric': False, 'note': 'Peak too narrow'}

        # Calculate statistical skewness
        mean_mz = np.average(region_mz, weights=region_int)
        variance = np.average((region_mz - mean_mz) ** 2, weights=region_int)
        std_dev = np.sqrt(variance)

        if std_dev > 0:
            skewness = np.average(((region_mz - mean_mz) / std_dev) ** 3, weights=region_int)
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
            'apex_mz': float(apex_mz),
        }

    def generate_experimental_gaussian_envelope(
        self, exp_mz: npt.NDArray[np.float64], exp_int: npt.NDArray[np.float64], resolution: int
    ) -> tuple[Optional[npt.NDArray], Optional[npt.NDArray]]:
        """
        Generate smooth Gaussian envelope for experimental data.
        Uses Gaussian smoothing with kernel based on instrument resolution.
        This will show the natural asymmetry of the experimental data.
        """
        try:
            logger.debug('GENERATE_EXPERIMENTAL_GAUSSIAN_ENVELOPE CALLED')
            logger.debug(f'Input: {len(exp_mz)} m/z points, resolution={resolution}')

            if len(exp_mz) == 0 or len(exp_int) == 0:
                logger.warning('FAILED: Empty input data')
                return None, None

            # Convert to numpy arrays
            exp_mz = np.array(exp_mz)
            exp_int = np.array(exp_int)

            # Calculate FWHM and sigma from resolution
            peak_center = np.average(exp_mz, weights=exp_int)
            fwhm = peak_center / resolution
            sigma = fwhm / 2.355  # Convert FWHM to sigma

            logger.debug(f'Peak center: {peak_center:.4f}, FWHM: {fwhm:.6f}, sigma: {sigma:.6f}')

            # SMART APPROACH: Find apex (local maximum) of each isotope peak
            # Then use the SAME smooth_gaussian_pattern function as theoretical data
            # This ensures consistent smooth curves!

            from scipy.signal import find_peaks

            # Find local maxima (apex of each isotope peak)
            # Use a small distance to separate isotope peaks (~0.2 Da for typical spacing)
            min_distance = int(0.2 / np.median(np.diff(exp_mz))) if len(exp_mz) > 1 else 2
            peaks_idx, properties = find_peaks(
                exp_int, distance=max(2, min_distance), prominence=np.max(exp_int) * 0.05
            )

            if len(peaks_idx) < 3:
                # Not enough peaks found - use all data points
                logger.debug(f'Found only {len(peaks_idx)} apex points, using all data')
                apex_mz = exp_mz
                apex_int = exp_int
            else:
                # Extract apex points
                all_apex_mz = exp_mz[peaks_idx]
                all_apex_int = exp_int[peaks_idx]
                logger.debug(f'Found {len(all_apex_mz)} apex points (local maxima)')

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

                apex_mz = all_apex_mz[start : end + 1]
                apex_int = all_apex_int[start : end + 1]

                logger.debug(
                    f'Kept {len(apex_mz)} contiguous apex points '
                    f'[{apex_mz[0]:.4f}, {apex_mz[-1]:.4f}] around max at '
                    f'{all_apex_mz[max_apex_idx]:.4f} (gap_threshold={gap_threshold:.3f})'
                )

                if len(apex_mz) < 3:
                    logger.debug(f'Too few contiguous points, using all {len(all_apex_mz)} apex points')
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
                        logger.info(
                            f'Alternating pattern detected (ratio={alternation_ratio:.2f}): '
                            f'interpolated {len(minor_idx)} minor peaks from {len(major_idx)} major peaks'
                        )

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
                logger.debug(f'STEP 1: Cubic spline through {len(apex_mz)} apex -> {len(mz_grid)} points')
            else:
                intensity_interp = np.interp(mz_grid, apex_mz, apex_int)
                logger.debug(f'STEP 1: Linear interpolation through {len(apex_mz)} apex -> {len(mz_grid)} points')

            # STEP 2: Apply STRONGER Gaussian smoothing for better curve fitting
            mz_step = (mz_max - mz_min) / num_points if num_points > 1 else fwhm / 100
            sigma_pixels = (sigma / mz_step) * 15.0  # 15x stronger smoothing for better Gaussian fit
            intensity_grid = gaussian_filter1d(intensity_interp, sigma=sigma_pixels, mode='nearest')
            logger.debug(f'STEP 2: STRONG Gaussian smoothing (sigma={sigma:.6f} m/z x 15 = {sigma_pixels:.2f} pixels)')

            # Clip negative values (artifacts from edge smoothing)
            intensity_grid = np.maximum(intensity_grid, 0.0)

            # Normalize to 100
            if np.max(intensity_grid) > 0:
                intensity_grid = (intensity_grid / np.max(intensity_grid)) * 100.0

            logger.info(f'SUCCESS: Smooth envelope from {len(apex_mz)} apex points')
            logger.debug(f'Envelope: {len(mz_grid)} points, m/z [{np.min(mz_grid):.4f}, {np.max(mz_grid):.4f}]')

            return mz_grid, intensity_grid

        except Exception as e:
            logger.exception(f'[generate_experimental_gaussian_envelope] Exception: {str(e)}')
            return None, None

    def fit_gaussian_to_smooth_envelope(
        self,
        mz_array: Optional[npt.NDArray[np.float64]],
        int_array: Optional[npt.NDArray[np.float64]],
        resolution: int,
        context: str = '',
    ) -> tuple[Optional[float], Optional[float], bool]:
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
        from scipy.signal import find_peaks

        # Igor Pro-style 4-parameter Gaussian: f(x) = y0 + A × exp(-((x - x₀) / w)²)
        def gaussian(x, y0, A, x0, width):
            return y0 + A * np.exp(-(((x - x0) / width) ** 2))

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
                    if i > 0 and int_apex[i] < int_apex[i - 1]:
                        if int_apex[i] < int_apex[i + 1] * 0.9:
                            left_bound_idx = i
                            break

                # Scan right to find valley
                right_bound_idx = len(int_apex) - 1
                for i in range(center_idx + 1, len(int_apex)):
                    if i < len(int_apex) - 1 and int_apex[i] < int_apex[i + 1]:
                        if int_apex[i] < int_apex[i - 1] * 0.9:
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
                    logger.debug(
                        f'[{context}] Valley boundaries: [{left_mz:.4f}, {right_mz:.4f}], fitting {len(mz_fit)} points'
                    )
            else:
                # Fallback: use all data
                mz_fit = mz_array
                int_fit = int_array
                if context:
                    logger.debug(f'[{context}] Too few apexes ({len(peaks_idx)}), using all {len(mz_fit)} points')

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
                bounds=(
                    [-A_init * 0.1, 0, mz_fit[0], 0.001],
                    [A_init * 0.5, A_init * 2, mz_fit[-1], fwhm_estimate * 2 * np.sqrt(2)],
                ),
                maxfev=5000,
            )

            x0 = float(popt[2])
            sigma = float(abs(popt[3]) / np.sqrt(2))

            if context:
                logger.debug(f'[{context}] Gaussian fit: X0={x0:.4f} m/z, sigma={sigma:.6f} m/z')

            return x0, sigma, True

        except Exception as e:
            if context:
                logger.warning(f'[{context}] Gaussian fit failed ({str(e)}), using apex fallback')
            # Fallback: use apex of smooth envelope
            max_idx = np.argmax(int_array)
            x0 = float(mz_array[max_idx])
            fwhm = x0 / resolution
            sigma = float(fwhm / 2.355)

            return x0, sigma, False

    def detect_peak_asymmetry_visual(
        self, mz_array: npt.NDArray[np.float64], int_array: npt.NDArray[np.float64], threshold_ratio: float = 0.3
    ) -> tuple[bool, int, str]:
        """
        Detect peak asymmetry using visual characteristics:
        - Count local maxima (multiple bumps = asymmetric)
        - Check for shoulders (secondary peaks)
        - Measure envelope smoothness

        Returns: (is_asymmetric, num_maxima, details)
        """
        try:
            if len(mz_array) < 5:
                return False, 1, 'Too few points'

            mz_array = np.array(mz_array)
            int_array = np.array(int_array)

            # Normalize intensity
            max_int = np.max(int_array)
            if max_int == 0:
                return False, 1, 'Zero intensity'

            int_norm = int_array / max_int

            # Find local maxima (peaks)
            from scipy.signal import find_peaks

            # Detect peaks with minimum height (to avoid noise)
            # Prominence helps identify significant peaks vs noise
            peaks, properties = find_peaks(
                int_norm,
                height=threshold_ratio,  # At least 30% of max height
                prominence=0.1,  # Must be prominent enough
                distance=3,  # Separated by at least 3 points
            )

            num_maxima = len(peaks)

            # Determine if asymmetric based on number of significant maxima
            is_asymmetric = num_maxima > 1

            details = f'{num_maxima} local maxima detected'
            if num_maxima > 1:
                peak_positions = [f'{mz_array[p]:.2f}' for p in peaks]
                details += f' at m/z: {", ".join(peak_positions)}'

            logger.debug(f'[Visual asymmetry detection] {details} -> {"ASYMMETRIC" if is_asymmetric else "SYMMETRIC"}')

            return is_asymmetric, num_maxima, details

        except Exception as e:
            logger.error(f'[detect_peak_asymmetry_visual] Error: {str(e)}')
            return False, 1, f'Error: {str(e)}'

    def calculate_peak_skewness(
        self, mz_array: npt.NDArray[np.float64], int_array: npt.NDArray[np.float64]
    ) -> Optional[float]:
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
            variance = np.sum(int_array * (mz_array - mean) ** 2) / np.sum(int_array)
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
            logger.error(f'[calculate_peak_skewness] Exception: {str(e)}')
            return None

    def weighted_average_centroid(
        self, mz_array: npt.NDArray[np.float64], int_array: npt.NDArray[np.float64]
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate centroid using weighted average method.
        This is used for general centroid calculations.
        Returns x₀ = Σ(m/z × intensity) / Σ(intensity) and σ (weighted std dev)
        """
        try:
            if len(mz_array) == 0 or len(int_array) == 0:
                logger.warning('[weighted_average_centroid] Empty arrays')
                return None, None

            # Convert to numpy arrays
            mz_array = np.array(mz_array, dtype=float)
            int_array = np.array(int_array, dtype=float)

            total_intensity = np.sum(int_array)
            if total_intensity == 0 or np.isnan(total_intensity) or np.isinf(total_intensity):
                logger.warning(f'[weighted_average_centroid] Invalid total intensity: {total_intensity}')
                return None, None

            # Weighted average: x₀ = Σ(m/z × intensity) / Σ(intensity)
            x0 = np.sum(mz_array * int_array) / total_intensity

            # Weighted standard deviation: σ = sqrt(Σ(intensity × (m/z - x₀)²) / Σ(intensity))
            sigma = np.sqrt(np.sum(int_array * (mz_array - x0) ** 2) / total_intensity)

            if np.isnan(x0) or np.isinf(x0):
                return None, None

            logger.debug(f'[weighted_average_centroid] x0={x0:.4f}, sigma={sigma:.4f}')
            return float(x0), float(sigma)

        except Exception as e:
            logger.error(f'[weighted_average_centroid] Exception: {str(e)}')
            return None, None

    def gaussian_fit_centroid(
        self, mz_array: npt.NDArray[np.float64], int_array: npt.NDArray[np.float64], return_quality: bool = False
    ) -> tuple:
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
                logger.warning(
                    f'[gaussian_fit_centroid] Empty arrays: mz length={len(mz_array)}, int length={len(int_array)}'
                )
                if return_quality:
                    return None, None, None, None
                else:
                    return None, None, None

            # Convert to numpy arrays
            mz_array = np.array(mz_array, dtype=float)
            int_array = np.array(int_array, dtype=float)

            if len(mz_array) < 3:
                logger.warning('[gaussian_fit_centroid] Need at least 3 points for fitting')
                if return_quality:
                    return None, None, None, None
                else:
                    return None, None, None

            total_intensity = np.sum(int_array)
            if total_intensity == 0 or np.isnan(total_intensity) or np.isinf(total_intensity):
                logger.warning(f'[gaussian_fit_centroid] Invalid total intensity: {total_intensity}')
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
                logger.debug(f'[gaussian_fit_centroid] Found {len(peaks_idx)} isotope peak apexes')

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
                    if i > 0 and int_apex[i] < int_apex[i - 1]:
                        if int_apex[i] < int_apex[i + 1] * 0.9 and int_apex[i] < valley_threshold:
                            left_bound_idx = i
                            break

                # Scan right from center to find minimum
                right_bound_idx = len(int_apex) - 1
                for i in range(center_idx + 1, len(int_apex)):
                    if i < len(int_apex) - 1 and int_apex[i] < int_apex[i + 1]:
                        if int_apex[i] < int_apex[i - 1] * 0.9 and int_apex[i] < valley_threshold:
                            right_bound_idx = i
                            break

                # Step 4: Extract apex points BETWEEN envelope valleys
                mz_fit = mz_apex[left_bound_idx : right_bound_idx + 1]
                int_fit = int_apex[left_bound_idx : right_bound_idx + 1]

                logger.debug(
                    f'[gaussian_fit_centroid] Envelope valleys: left={left_bound_idx}, center={center_idx}, right={right_bound_idx}'
                )
                logger.debug(f'[gaussian_fit_centroid] Fitting to {len(mz_fit)} apex points between envelope valleys')
            else:
                # Fallback: if too few peaks, use all apex points or top 70%
                if len(peaks_idx) >= 3:
                    mz_fit = mz_array[peaks_idx]
                    int_fit = int_array[peaks_idx]
                    logger.warning(f'[gaussian_fit_centroid] Only {len(peaks_idx)} apexes, using all')
                else:
                    logger.warning('[gaussian_fit_centroid] Too few apexes, using top 70%')
                    max_intensity = np.max(int_array)
                    threshold = max_intensity * 0.70
                    high_intensity_mask = int_array >= threshold
                    mz_fit = mz_array[high_intensity_mask]
                    int_fit = int_array[high_intensity_mask]

            if len(mz_fit) < 3:
                logger.error('[gaussian_fit_centroid] Too few points, using all data')
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

            logger.debug(
                f'[gaussian_fit_centroid] Initial guesses: x0={x0_guess:.4f} (apex), sigma={sigma_guess:.4f}, A={A_guess:.2e}, y0={y0_guess:.2e}'
            )

            # Igor Pro-style 4-parameter Gaussian: f(x) = y0 + A × exp(-((x - x₀) / w)²)
            # where w = sqrt(2) × σ, so exponent = -(x - x₀)² / (2σ²)
            def gaussian(x, y0, A, x0, width):
                return y0 + A * np.exp(-(((x - x0) / width) ** 2))

            # Fit Gaussian curve to HIGH-INTENSITY data only
            try:
                from scipy.optimize import curve_fit

                width_guess = sigma_guess * np.sqrt(2)

                # Allow x0 to vary within the full valley boundaries
                bounds = (
                    [-max_int_fit * 0.1, 0, mz_min_all, 0.01],  # Lower bounds: [y0_min, A_min, x0_min, width_min]
                    [max_int_fit * 0.5, np.inf, mz_max_all, (mz_max_all - mz_min_all) * 2],  # Upper bounds
                )

                logger.debug(f'[gaussian_fit_centroid] x0 bounds: [{mz_min_all:.4f}, {mz_max_all:.4f}]')

                popt, pcov = curve_fit(
                    gaussian,
                    mz_fit,  # Fit to high-intensity points only
                    int_fit,
                    p0=[y0_guess, A_guess, x0_guess, width_guess],
                    bounds=bounds,
                    maxfev=10000,
                    ftol=1e-10,  # Function tolerance for convergence (more precise)
                    xtol=1e-10,  # Parameter tolerance for convergence (more precise)
                )

                y0_fit, A_fit, x0_fit, width_fit = popt
                sigma_fit = width_fit / np.sqrt(2)

                # Calculate standard errors from covariance matrix
                # pcov diagonal gives variance of parameters, sqrt gives standard error
                perr = np.sqrt(np.diag(pcov))
                y0_err, A_err, x0_err, width_err = perr

                # Validate fitted parameters
                if np.isnan(x0_fit) or np.isinf(x0_fit) or np.isnan(sigma_fit) or np.isinf(sigma_fit):
                    raise ValueError('Fit returned invalid parameters')

                # Calculate R² (coefficient of determination) if requested
                if return_quality:
                    # Predicted values from fitted Gaussian
                    y_pred = gaussian(mz_array, y0_fit, A_fit, x0_fit, width_fit)

                    # Calculate R²
                    ss_res = np.sum((int_array - y_pred) ** 2)  # Residual sum of squares
                    ss_tot = np.sum((int_array - np.mean(int_array)) ** 2)  # Total sum of squares
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                    logger.debug(
                        f'[gaussian_fit_centroid] Gaussian fit: x₀={x0_fit:.4f}±{x0_err:.4f}, σ={sigma_fit:.4f}, y0={y0_fit:.2e}, R²={r_squared:.4f}'
                    )
                    return float(x0_fit), float(sigma_fit), float(x0_err), float(r_squared)
                else:
                    logger.debug(
                        f'[gaussian_fit_centroid] Gaussian fit: x₀={x0_fit:.4f}±{x0_err:.4f}, σ={sigma_fit:.4f}, y0={y0_fit:.2e}'
                    )
                    return float(x0_fit), float(sigma_fit), float(x0_err)

            except Exception as fit_error:
                # If Gaussian fit fails, fall back to weighted average
                logger.warning(
                    f'[gaussian_fit_centroid] Gaussian fit failed ({fit_error}), using weighted average fallback'
                )
                x0_fallback = x0_guess
                sigma_fallback = sigma_guess

                if np.isnan(x0_fallback) or np.isinf(x0_fallback):
                    if return_quality:
                        return None, None, None, None
                    else:
                        return None, None, None

                if return_quality:
                    return (
                        float(x0_fallback),
                        float(sigma_fallback),
                        None,
                        0.0,
                    )  # No fitting error, R² = 0 indicates fit failed
                else:
                    return float(x0_fallback), float(sigma_fallback), None  # No fitting error available

        except Exception as e:
            logger.exception(f'[gaussian_fit_centroid] Exception: {str(e)}')
            if return_quality:
                return None, None, None, None
            else:
                return None, None, None
