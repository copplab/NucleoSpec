from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np
import numpy.typing as npt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'lib'))

from pythoms.tome import autoresolution

logger = logging.getLogger(__name__)


class SpectrumMixin:
    """Mixin for spectrum parsing and peak detection methods."""

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

    def estimate_resolution(self, mz_values: npt.NDArray[np.float64], intensity_values: npt.NDArray[np.float64]) -> int:
        """
        Estimate the resolution of the spectrum using PythoMS methodology
        """
        try:
            res = autoresolution(list(mz_values), list(intensity_values), n=10, v=False)
            if res is None or not np.isfinite(res) or res <= 0:
                return 20000  # Default resolution
            return int(res)
        except Exception:
            return 20000  # Default resolution if estimation fails

    def calculate_fwhm(self, mz: float, resolution: int) -> float:
        """
        Calculate Full Width at Half Maximum for a given m/z and resolution
        FWHM = m/z / resolution
        """
        return mz / resolution

    def find_local_maximum(
        self,
        mz_values: npt.NDArray[np.float64],
        intensity_values: npt.NDArray[np.float64],
        center_mz: float,
        lookwithin: Optional[float] = None,
    ) -> tuple[Optional[float], Optional[float]]:
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

    def find_peak_regions(
        self,
        mz_values: npt.NDArray[np.float64],
        intensity_values: npt.NDArray[np.float64],
        threshold: float = 0.05,
        merge_gap: float = 1.5,
    ) -> list[tuple[int, int]]:
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

    def detect_peak_boundaries(
        self, mz_array: npt.NDArray[np.float64], int_array: npt.NDArray[np.float64], peak_mz: float
    ) -> tuple[float, float, float]:
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

        logger.debug(f'Detecting isotope envelope boundaries around clicked m/z={peak_mz:.4f}')
        logger.debug(f'Clicked at index={peak_idx}, mz={mz_array[peak_idx]:.4f}')
        logger.debug(f'Found APEX at index={apex_idx}, mz={apex_mz:.4f}, intensity={apex_intensity:.0f}')

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
                if int_array[i - 1] > current_intensity * 1.1 and int_array[i] > int_array[i + 1] * 1.1:
                    # Found a valley - intensity is rising on the left
                    logger.debug(
                        f'Left valley at index={left_idx}, mz={mz_array[left_idx]:.4f}, intensity={int_array[left_idx]:.0f}'
                    )
                    break

        # If we hit the edge without finding a valley, use the minimum we found
        if left_idx == apex_idx:
            logger.debug(f'Left boundary at edge: index={left_idx}, mz={mz_array[left_idx]:.4f}')

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
                if int_array[i + 1] > current_intensity * 1.1 and int_array[i] > int_array[i - 1] * 1.1:
                    # Found a valley - intensity is rising on the right
                    logger.debug(
                        f'Right valley at index={right_idx}, mz={mz_array[right_idx]:.4f}, intensity={int_array[right_idx]:.0f}'
                    )
                    break

        # If we hit the edge without finding a valley, use the minimum we found
        if right_idx == apex_idx:
            logger.debug(f'Right boundary at edge: index={right_idx}, mz={mz_array[right_idx]:.4f}')

        left_boundary_mz = mz_array[left_idx]
        right_boundary_mz = mz_array[right_idx]
        width = right_boundary_mz - left_boundary_mz
        num_points = right_idx - left_idx + 1

        logger.debug(
            f'Final envelope: [{left_boundary_mz:.4f}, {right_boundary_mz:.4f}] m/z (width={width:.4f}, {num_points} points)'
        )
        logger.debug(f'Apex at {apex_mz:.4f} (Gaussian will use MIDPOINT of boundaries as initial guess)')

        return left_boundary_mz, right_boundary_mz, apex_mz

    def weighted_centroid(
        self,
        mz_values: npt.NDArray[np.float64],
        intensity_values: npt.NDArray[np.float64],
        start_idx: int,
        end_idx: int,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate peak centroid (position of maximum intensity) matching PythoMS isotope overlay method

        This uses the m/z value at maximum intensity for peak position, which is consistent
        with how PythoMS's plot_mass_spectrum and localmax functions work.
        """
        region_mz = mz_values[start_idx : end_idx + 1]
        region_int = intensity_values[start_idx : end_idx + 1]

        if len(region_mz) == 0 or np.sum(region_int) == 0:
            return None, None

        # Find the m/z at maximum intensity (peak apex)
        # This matches PythoMS isotope overlay behavior
        max_idx = np.argmax(region_int)
        centroid_mz = region_mz[max_idx]
        max_intensity = region_int[max_idx]

        return centroid_mz, max_intensity
