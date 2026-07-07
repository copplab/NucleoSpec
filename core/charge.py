from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np
import numpy.typing as npt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'lib'))


logger = logging.getLogger(__name__)


class ChargeMixin:
    """Mixin for charge state detection from isotope spacing."""

    def group_isotope_envelope(
        self, peak_mz: npt.NDArray[np.float64], peak_intensity: npt.NDArray[np.float64], charge: Optional[int]
    ) -> Optional[int]:
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

    def detect_charge_state(
        self,
        mz_values: npt.NDArray[np.float64],
        intensity_values: npt.NDArray[np.float64],
        target_mz: float,
        window: float = 3.0,
    ) -> dict:
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

            left = sum(1 for k in range(1, n_iso + 1) if np.min(np.abs(peak_mzs - (target_mz - k * spacing))) <= tol)
            right = sum(1 for k in range(1, n_iso + 1) if np.min(np.abs(peak_mzs - (target_mz + k * spacing))) <= tol)

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
            if half in results and results[half]['viable'] and results[best_z]['alt'] < DOUBLET_ALT_THRESHOLD:
                logger.info(
                    f'[detect_charge_state] Ag-doublet halving at m/z {target_mz:.4f}: '
                    f'z={best_z} -> z={half} (alt={results[best_z]["alt"]:.2f})'
                )
                best_z = half
            else:
                break

        best = results[best_z]
        confidence = max(0.0, min(1.0, best['score']))
        num_matched = best['left'] + best['right'] + 1

        logger.debug(
            f'[detect_charge_state] target={target_mz:.4f} -> z={best_z} '
            f'(coverage={best["coverage"]:.2f}, gap={best["gap_frac"]:.2f}, '
            f'alt={best["alt"]:.2f}, score={best["score"]:+.3f})'
        )

        return {
            'spacing': float(best['spacing']),
            'charge': int(best_z),
            'confidence': float(confidence),
            'num_peaks': int(num_matched),
            'scores': results,
        }

    def detect_charge_for_clicked_peak(
        self,
        mz_values: npt.NDArray[np.float64],
        intensity_values: npt.NDArray[np.float64],
        target_mz: float,
        charge_range: tuple[int, int] = (1, 10),
    ) -> dict:
        """
        Determine charge state of a user-clicked peak.

        Primary: isotope-grid scoring via detect_charge_state.
        Fallback: Senko charge assignment on the surrounding envelope.
        Returns dict with 'charge', 'confidence', 'method' (and 'spacing',
        'num_peaks' when produced by the primary method). 'charge' is None
        only when both methods fail.
        """
        logger.info(f'[Charge Detection] Analyzing peak at m/z {target_mz:.4f}')

        result = self.detect_charge_state(mz_values, intensity_values, target_mz, window=3.0)
        charge = result.get('charge')
        if charge is not None and charge_range[0] <= charge <= charge_range[1]:
            num_peaks = int(result.get('num_peaks', 0))
            confidence = float(result.get('confidence', 0.0))
            if num_peaks < 3:
                confidence = min(0.6, confidence)
            logger.info(f'[Charge Detection] z={charge} via grid (conf={confidence * 100:.0f}%, {num_peaks} matched)')
            return {
                'charge': int(charge),
                'confidence': float(confidence),
                'method': 'spacing',
                'spacing': float(result['spacing']),
                'num_peaks': num_peaks,
            }

        logger.debug('[Charge Detection] Grid method inconclusive, falling back to Senko')
        try:
            from pythoms.senko_charge_assignment import detect_all_peaks_with_charge

            detected = detect_all_peaks_with_charge(
                mz_values,
                intensity_values,
                prominence=0.01,
                charge_range=charge_range,
                method='combination',
                merge_gap=1.5,
            )
            closest = min(
                (p for p in detected if abs(p['mz'] - target_mz) < 5.0 and p.get('charge') is not None),
                key=lambda p: abs(p['mz'] - target_mz),
                default=None,
            )
            if closest is not None:
                logger.info(f'[Charge Detection] z={closest["charge"]} via Senko fallback')
                return {
                    'charge': int(closest['charge']),
                    'confidence': float(closest['confidence']) * 0.8,
                    'method': 'senko_fallback',
                }
        except Exception as e:
            logger.error(f'[Charge Detection] Senko fallback error: {e}')

        logger.warning('[Charge Detection] All methods failed; user input required')
        return {'charge': None, 'confidence': 0.0, 'method': 'user_input_required'}
