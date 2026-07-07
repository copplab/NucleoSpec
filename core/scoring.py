from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class ScoringMixin:
    """Mixin for isotope pattern matching and similarity scoring."""

    def calculate_pattern_similarity(
        self,
        theo_mz: npt.NDArray[np.float64],
        theo_int: npt.NDArray[np.float64],
        exp_mz: npt.NDArray[np.float64],
        exp_int: npt.NDArray[np.float64],
        window: float = 3.0,
    ) -> float:
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
            peaks_idx, _ = find_peaks(exp_int, distance=max(2, min_distance), prominence=np.max(exp_int) * 0.02)

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
                np.linalg.norm(paired_theo) * np.linalg.norm(paired_exp) + 1e-10
            )
            cosine_sim = max(0.0, min(1.0, cosine_sim))

            # Pearson correlation
            if np.std(paired_theo) > 0 and np.std(paired_exp) > 0:
                correlation = np.corrcoef(paired_theo, paired_exp)[0, 1]
                correlation = max(0.0, min(1.0, correlation))
            else:
                correlation = 0.0

            return float((cosine_sim + correlation) / 2.0)

        except Exception as e:
            logger.exception(f'[calculate_pattern_similarity] Exception: {str(e)}')
            return 0.0

    def calculate_multi_parameter_fit_score(
        self,
        theo_mz: npt.NDArray[np.float64],
        theo_int: npt.NDArray[np.float64],
        exp_mz: npt.NDArray[np.float64],
        exp_int: npt.NDArray[np.float64],
        theo_x0: Optional[float],
        theo_sigma: Optional[float],
        exp_x0: Optional[float],
        exp_sigma: Optional[float],
    ) -> tuple[float, dict]:
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
                logger.error(f'[R-squared calculation failed]: {str(e)}')
                r_squared = 0.0

            # Composite score (weighted combination - lower is better)
            # Weight factors - adjust these based on importance
            w_x0 = 10.0  # X₀ error weight (m/z units)
            w_sigma = 5.0  # σ deviation weight
            w_r2 = 20.0  # R² weight (inverted since higher R² is better)

            composite_score = (
                w_x0 * x0_error  # Centroid position error
                + w_sigma * sigma_deviation  # Width mismatch
                + w_r2 * (1.0 - r_squared)  # Shape overlap quality (inverted)
            )

            metrics = {
                'x0_error': float(x0_error),
                'sigma_ratio': float(sigma_ratio) if sigma_ratio else None,
                'sigma_deviation': float(sigma_deviation),
                'r_squared': float(r_squared),
                'composite_score': float(composite_score),
            }

            logger.debug(
                f'[Fit Score] X0_err={x0_error:.4f}, sigma_ratio={sigma_ratio:.3f}, R_squared={r_squared:.4f}, Score={composite_score:.2f}'
            )

            return composite_score, metrics

        except Exception as e:
            logger.exception(f'[calculate_multi_parameter_fit_score] Exception: {str(e)}')
            return 999.0, {'x0_error': 999.0, 'sigma_ratio': None, 'r_squared': None}

    def match_isotope_pattern(
        self,
        experimental_mz: npt.NDArray[np.float64],
        experimental_int: npt.NDArray[np.float64],
        theoretical_pattern: dict,
        tolerance: float = 0.5,
    ) -> float:
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
