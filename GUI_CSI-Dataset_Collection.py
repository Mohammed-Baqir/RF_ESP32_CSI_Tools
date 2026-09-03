#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2021-2025 Espressif Systems (Shanghai) CO LTD
# SPDX-FileCopyrightText: 2026 Mohammed Baqir (PhD Contributions & Enhancements)
# SPDX-License-Identifier: Apache-2.0
#
"""
===========================================================================
Unified Wi-Fi CSI Viewer & Spatial Diversity Analyzer (v4.0)
===========================================================================
Developed as part of PhD research in RF Sensing and Wi-Fi CSI Analysis.

Hardware Topology:
    TX : 1x Tplink-WiFi (3 Antennas, 2.4 GHz)
    RX : 1x ESP32 (1 Antenna,  2.4 GHz)
    Multi-RX mode supports up to 8 RX nodes simultaneously.

===========================================================================
FEATURES
===========================================================================

[1] CSI Dataset Acquisition
    - Unified Architecture: Single-RX (legacy) & Multi-RX (up to 8 nodes)
    - Time-Alignment Engine for Multi-RX synchronization
    - 1-Second Auto-Marking for uniform temporal data collection
    - Structured session export (metadata.json, events.csv, csi_data.npy,
      agc_data.npy, fft_data.npy, session_info.json, rf_summary.json)
    - Categorized output: 01_human_activity, 02_metal_object,
      03_environment_with_metal

[2] Signal Processing & Anomaly Detection
    - Hampel + Butterworth Causal Filtering (per-subcarrier, streaming)
    - Ensemble Anomaly Detection:
        • Isolation Forest
        • Local Outlier Factor (novelty mode)
        • Elliptic Envelope
    - Adaptive thresholding with rolling percentile of clean scores
    - Spatial Diversity Engine:
        • Maximum Ratio Combining (MRC)
        • Phase Difference (circular)
        • Coherence Index
        • Spatial PCA (top-3 principal components)
    - Continuous Wavelet Transform (CWT) Feature Extraction
    - Rich feature extraction (amplitude, phase, temporal, spectral,
      inter-subcarrier correlation)

[3] Real-Time GUI & Visualization
    - Live subcarrier amplitude over time
    - IQ (Real / Imaginary) component plots
    - Live spectrogram / waterfall display
    - Ensemble anomaly score plot with adaptive threshold line
    - Alert system (visual indicator + audio beep + severity levels)
    - RF parameter display (RSSI, MCS, bandwidth, channel, etc.)
    - Real-time statistics (FPS, processing time, anomaly rate, memory)

[4] Model Management
    - Upload trained models (.pkl, .joblib) for real-time evaluation
    - Pipeline state save/load (.joblib) for calibration persistence
    - Headless (console-only) mode for server deployment

===========================================================================
KNOWN LIMITATIONS
===========================================================================
- Running large classification models in real-time may cause system
  overload (observed on RTX 3070 GPU, 32 GB RAM). Optimization of
  the inference pipeline is ongoing.
- The Anomaly Detection module is currently an independent research
  track and has not yet been fully integrated with the labeled dataset
  for supervised training.

===========================================================================
"""
import sys
import csv
import json
import argparse
import numpy as np
import serial
import os
import time
import gc
import threading
import traceback
from datetime import datetime
from collections import deque
from io import StringIO

from PyQt5.Qt import *
from PyQt5 import QtCore
import pyqtgraph as pg

import joblib
from scipy import stats
from scipy.signal import welch
import pywt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# 4T AI INFERENCE ENGINE  (import with __main__ namespace fix)
# ─────────────────────────────────────────────────────────
_4T_ENGINE_AVAILABLE = False
try:
    SCRIPT_DIR_4T = os.path.dirname(os.path.abspath(__file__))
    OPT_DIR_4T = os.path.join(SCRIPT_DIR_4T, '4T_opt')
    if SCRIPT_DIR_4T not in sys.path:
        sys.path.insert(0, SCRIPT_DIR_4T)
    if OPT_DIR_4T not in sys.path:
        sys.path.insert(0, OPT_DIR_4T)

    # ── sklearn version compatibility shim ────────────────────────
    # Models pickled with sklearn 1.3 reference sklearn.ensemble._gb_losses
    # which was removed in sklearn 1.4+.  Create a shim module so
    # joblib.load() can unpickle them on newer sklearn versions.
    import types as _types
    try:
        import sklearn.ensemble._gb_losses          # noqa: exists in <=1.3
    except ImportError:
        _shim = _types.ModuleType('sklearn.ensemble._gb_losses')
        # sklearn 1.5 GBR calls loss.link.link(x) — identity for regression
        try:
            from sklearn._loss.link import IdentityLink as _IdLink
        except ImportError:
            class _IdLink:
                def link(self, y):   return y
                def inverse(self, y): return y

        class _LeastSquaresError:
            K = 1
            is_multi_class = False
            def __init__(self):
                self.link = _IdLink()
            def __setstate__(self, state):
                self.__dict__.update(state)
                if not hasattr(self, 'link'):
                    self.link = _IdLink()

        class _QuantileLossFunction:
            K = 1
            is_multi_class = False
            def __init__(self, n_classes=1, alpha=0.05):
                self.K = n_classes
                self.alpha = alpha
                self.percentile = alpha * 100
                self.link = _IdLink()
            def __setstate__(self, state):
                self.__dict__.update(state)
                if not hasattr(self, 'link'):
                    self.link = _IdLink()

        _shim.LeastSquaresError = _LeastSquaresError
        _shim.QuantileLossFunction = _QuantileLossFunction
        sys.modules['sklearn.ensemble._gb_losses'] = _shim
        import sklearn.ensemble
        sklearn.ensemble._gb_losses = _shim
        print("[4T] Applied sklearn._gb_losses compatibility shim")
    # ──────────────────────────────────────────────────────────────

    import __main__ as _main
    from approach1_environment_recognition import StackingEnsemble
    from approach2_object_classification import BayesianGBMMeta
    from approach3_distance_estimation import ConformalQuantileForest
    from approach4_person_activity import HierarchicalCascade, CalibratedClassifier
    _main.StackingEnsemble = StackingEnsemble
    _main.BayesianGBMMeta = BayesianGBMMeta
    _main.ConformalQuantileForest = ConformalQuantileForest
    _main.HierarchicalCascade = HierarchicalCascade
    _main.CalibratedClassifier = CalibratedClassifier

    # NOTE: realtime_inference_4t has been removed from the 4T pipeline.
    # The GUI's 4T AI panel is kept for layout consistency but is disabled.
    from signal_eliminator_4t import PersonActivityEliminator
    from shared_preprocessing import (
        preprocess_csi_segment, extract_all_features,
        csi_segment_to_tensor_7ch, csi_segment_to_tensor,
        _build_tensor_7ch_from_pp,
    )
    _4T_ENGINE_AVAILABLE = False  # realtime engine removed
    print("[4T] Real-time inference engine removed; 4T panel disabled.")
except ImportError as e:
    print(f"[4T] AI modules not available: {e}")
    print("[4T] 4T predictions will be disabled")

# =========================================================
# GLOBALS / CONSTANTS
# =========================================================
CSI_DATA_INDEX = 200
csi_data_complex = None
agc_gain_data = None
fft_gain_data = None

DATA_COLUMNS_NAMES_C5C6 = [
    "type", "id", "mac", "rssi", "rate", "noise_floor", "fft_gain", "agc_gain",
    "channel", "local_timestamp", "sig_len", "rx_state", "len", "first_word", "data"
]

DATA_COLUMNS_NAMES = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing",
    "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor",
    "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant",
    "sig_len", "rx_state", "len", "first_word", "data"
]

seen_lengths = set()


# =========================================================
# ENHANCED SIGNAL PROCESSING PIPELINE  (Ensemble Detector)
# =========================================================
class EnhancedSignalProcessingPipeline:
    """
    Signal processing pipeline with ensemble anomaly detection:
      • Isolation Forest  – tree-based, robust to high dimensions
      • Local Outlier Factor (novelty mode) – density-based, sensitive to local clusters
      • Elliptic Envelope – covariance-based, catches Gaussian outliers
    Scores are z-normalised then combined with weighted voting.
    An adaptive threshold is maintained from a rolling percentile of clean scores.
    """

    def __init__(self, window_size=30):
        self.window_size = window_size

        # Hampel filter parameters
        self.hampel_window = 5
        self.hampel_n_sigma = 3

        # ── Ensemble anomaly detectors ──────────────────────────────
        self.isolation_forest = None
        self.lof = None
        self.elliptic_env = None

        # Shared scaler for all detectors
        self.anomaly_scaler = StandardScaler()
        self.empty_room_features = None
        self.is_calibrated = False
        self.calibration_samples = 0
        self.calibration_target = 150           # more samples → better baseline

        # Ensemble weights (IF, LOF, EE) — tunable
        self.ensemble_weights = np.array([0.40, 0.35, 0.25])

        # Per-model normalisation statistics (fitted on training data)
        # Each entry: (mean, std) so inference uses the same scale as training
        self._score_stats = {
            'if':  (0.0, 1.0),
            'lof': (0.0, 1.0),
            'ee':  (0.0, 1.0),
        }

        # Adaptive threshold tracking
        self.clean_score_buffer = deque(maxlen=500)
        self.anomaly_threshold = -0.5
        self.adaptive_threshold_percentile = 5  # bottom 5% of clean scores

        # CWT parameters – kept for feature extraction fed to external classifier
        self.cwt_wavelet = 'morl'
        self.cwt_scales = np.arange(1, 32)
        self.cwt_enabled = True
        self.cwt_processing_count = 0
        self.cwt_skip_count = 0

        # Buffers
        self.amplitude_buffer = deque(maxlen=window_size)
        self.phase_buffer = deque(maxlen=window_size)
        self.rf_params_buffer = deque(maxlen=window_size)

        # Anomaly detection state
        self.anomaly_scores = deque(maxlen=200)
        self.in_anomaly_window = False
        self.anomaly_start_idx = None
        self.current_anomaly_segment = None
        self.last_anomaly_time = 0
        self.anomaly_cooldown = 1.0

        # Feature + CWT storage (anomaly segments only)
        self.basic_features_history = deque(maxlen=50)
        self.cwt_features_history = deque(maxlen=50)
        self.cwt_coefficients_history = deque(maxlen=10)

        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.recent_anomalies = deque(maxlen=10)

        # Threading for heavy CWT processing
        self.processing_thread = None
        self.processing_queue = deque(maxlen=10)
        self.processing_result = None
        self.processing_lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────
    # FEATURE EXTRACTION  (richer than before)
    # ─────────────────────────────────────────────────────────────────
    def extract_basic_features(self, amplitude_segment, phase_segment):
        """
        Extract a rich feature vector from a CSI window:
          • Amplitude:  mean, std, var, median, skew, kurt, range, energy,
                        RMS, crest factor, peak-to-peak, 10th/90th pct
          • Phase:      mean, std, var, circular mean, circular std
          • Temporal:   diff mean, diff std, diff energy, autocorr lag-1, slope
          • Frequency:  first 8 magnitudes of FFT of mean-amplitude signal
          • IQ domain:  mean/std of real & imaginary over the window
          • Inter-sub:  mean pairwise correlation (top-N subcarriers)
        """
        features = []

        # ── Amplitude statistics ──────────────────────────────────────
        amp_flat = amplitude_segment.flatten()
        amp_mean = np.mean(amp_flat)
        amp_std  = np.std(amp_flat)
        amp_var  = np.var(amp_flat)
        amp_energy = np.sum(amp_flat ** 2)
        amp_rms  = np.sqrt(amp_energy / max(len(amp_flat), 1))
        amp_range = np.max(amp_flat) - np.min(amp_flat)
        amp_crest = np.max(np.abs(amp_flat)) / (amp_rms + 1e-9)

        features.extend([
            amp_mean, amp_std, amp_var,
            np.median(amp_flat),
            stats.skew(amp_flat) if len(amp_flat) > 1 else 0,
            stats.kurtosis(amp_flat) if len(amp_flat) > 1 else 0,
            amp_range, amp_energy, amp_rms, amp_crest,
            np.percentile(amp_flat, 10),
            np.percentile(amp_flat, 90),
        ])

        # ── Phase statistics ──────────────────────────────────────────
        phase_flat = phase_segment.flatten()
        features.extend([
            np.mean(phase_flat),
            np.std(phase_flat),
            np.var(phase_flat),
            np.angle(np.mean(np.exp(1j * phase_flat))),   # circular mean
            np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * phase_flat))) + 1e-9)),  # circular std
        ])

        # ── Temporal features ─────────────────────────────────────────
        if amplitude_segment.shape[0] > 1:
            mean_over_time = np.mean(amplitude_segment, axis=1)
            temporal_diff  = np.diff(mean_over_time)
            features.extend([
                np.mean(temporal_diff),
                np.std(temporal_diff),
                np.sum(temporal_diff ** 2),
                # Autocorrelation at lag 1
                float(np.corrcoef(mean_over_time[:-1], mean_over_time[1:])[0, 1])
                if len(mean_over_time) > 2 else 0,
                # Linear trend slope
                float(np.polyfit(np.arange(len(mean_over_time)), mean_over_time, 1)[0]),
            ])
        else:
            features.extend([0, 0, 0, 0, 0])

        # ── Frequency-domain features (Welch PSD of mean-amplitude) ──
        if amplitude_segment.shape[0] >= 8:
            mean_sig = np.mean(amplitude_segment, axis=1)
            nperseg = min(len(mean_sig), 8)
            _, psd = welch(mean_sig, nperseg=nperseg)
            psd_norm = psd / (np.sum(psd) + 1e-9)
            features.extend(psd_norm[:8].tolist())
        else:
            features.extend([0] * 8)

        # ── Inter-subcarrier correlation ──────────────────────────────
        n_sub = min(amplitude_segment.shape[1], 10)
        if amplitude_segment.shape[0] > 2 and n_sub > 1:
            sub_matrix = amplitude_segment[:, :n_sub]
            corr_mat = np.corrcoef(sub_matrix.T)
            upper_tri = corr_mat[np.triu_indices(n_sub, k=1)]
            features.extend([
                np.mean(upper_tri),
                np.std(upper_tri),
            ])
        else:
            features.extend([0, 0])

        # ── Subcarrier-level statistics ───────────────────────────────
        sub_means = np.mean(amplitude_segment, axis=0)
        sub_stds  = np.std(amplitude_segment, axis=0)
        features.extend([
            np.mean(sub_means), np.std(sub_means),
            np.mean(sub_stds),  np.std(sub_stds),
        ])

        # ── Sanitise: replace NaN / Inf with 0 and clip to ±1e6 ─────
        arr = np.array(features, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1e6, 1e6)
        return arr

    # ─────────────────────────────────────────────────────────────────
    # HAMPEL FILTER
    # ─────────────────────────────────────────────────────────────────
    def hampel_filter(self, signal):
        """Remove hardware spikes via Hampel identifier."""
        if len(signal) < self.hampel_window * 2 + 1:
            return signal.copy()

        filtered = signal.copy()
        half_window = self.hampel_window // 2

        for i in range(len(signal)):
            start_idx = max(0, i - half_window)
            end_idx   = min(len(signal), i + half_window + 1)
            window    = signal[start_idx:end_idx]
            median    = np.median(window)
            mad       = np.median(np.abs(window - median))
            sigma     = 1.4826 * mad if mad > 0 else 0

            if sigma > 0 and np.abs(signal[i] - median) > self.hampel_n_sigma * sigma:
                filtered[i] = median

        return filtered

    # ─────────────────────────────────────────────────────────────────
    # CALIBRATION  (collect empty-room baseline)
    # ─────────────────────────────────────────────────────────────────
    def calibrate_empty_room(self, amplitude_segment, phase_segment, rf_params):
        features = self.extract_basic_features(amplitude_segment, phase_segment)

        if self.empty_room_features is None:
            self.empty_room_features = features.reshape(1, -1)
        else:
            self.empty_room_features = np.vstack([self.empty_room_features, features.reshape(1, -1)])

        self.calibration_samples += 1

        if self.calibration_samples >= self.calibration_target:
            return self.train_anomaly_detector()
        return False

    # ─────────────────────────────────────────────────────────────────
    # TRAIN ENSEMBLE ANOMALY DETECTOR
    # ─────────────────────────────────────────────────────────────────
    def train_anomaly_detector(self):
        """
        Fit a three-model ensemble on empty-room baseline:
          1. Isolation Forest   – global anomaly score
          2. LOF (novelty)      – local density score
          3. Elliptic Envelope  – Mahalanobis-distance score
        Adaptive threshold is set from the bottom percentile of clean scores.
        """
        if self.empty_room_features is None or len(self.empty_room_features) < 50:
            return False

        X = self.anomaly_scaler.fit_transform(self.empty_room_features)

        # 1. Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X)

        # 2. Local Outlier Factor (novelty=True for inference on new samples)
        self.lof = LocalOutlierFactor(
            n_neighbors=min(20, len(X) - 1),
            contamination=0.05,
            novelty=True,
            n_jobs=-1
        )
        self.lof.fit(X)

        # 3. Elliptic Envelope (needs enough samples for covariance)
        try:
            self.elliptic_env = EllipticEnvelope(
                contamination=0.05,
                support_fraction=0.9,
                random_state=42
            )
            self.elliptic_env.fit(X)
        except Exception as e:
            print(f"⚠️  EllipticEnvelope skipped: {e}")
            self.elliptic_env = None
            self.ensemble_weights = np.array([0.55, 0.45])

        # ── Store per-model score statistics from training data ───────
        # These are used at inference time so z-normalisation is stable
        # even when only a single sample is being scored.
        s_if  = self.isolation_forest.decision_function(X)
        s_lof = self.lof.decision_function(X)
        self._score_stats['if']  = (float(np.mean(s_if)),  float(np.std(s_if)  + 1e-9))
        self._score_stats['lof'] = (float(np.mean(s_lof)), float(np.std(s_lof) + 1e-9))
        if self.elliptic_env is not None:
            s_ee = self.elliptic_env.decision_function(X)
            self._score_stats['ee'] = (float(np.mean(s_ee)), float(np.std(s_ee) + 1e-9))

        print(f"   Score stats  IF : μ={self._score_stats['if'][0]:.4f}  σ={self._score_stats['if'][1]:.4f}")
        print(f"   Score stats LOF: μ={self._score_stats['lof'][0]:.4f}  σ={self._score_stats['lof'][1]:.4f}")

        # Compute combined scores on training data for adaptive threshold
        combined = self._combined_score(X)
        self.clean_score_buffer.extend(combined.tolist())
        self.anomaly_threshold = float(np.percentile(
            list(self.clean_score_buffer), self.adaptive_threshold_percentile
        ))

        self.is_calibrated = True
        print(f"✅ Ensemble anomaly detector trained on {len(X)} samples")
        print(f"   Threshold (P{self.adaptive_threshold_percentile}): {self.anomaly_threshold:.4f}")

        self.save_pipeline_state()
        return True

    # ─────────────────────────────────────────────────────────────────
    # ENSEMBLE SCORE COMBINATION
    # ─────────────────────────────────────────────────────────────────
    def _normalise_with_stats(self, score_arr, key):
        """
        Normalise using the mean / std recorded during training.
        Safe for single-sample arrays because it does NOT recompute
        std from the array itself — it uses the stored training statistics.
        """
        mu, sigma = self._score_stats[key]
        return (score_arr - mu) / sigma

    def _combined_score(self, X_scaled):
        """
        Returns a combined anomaly score per sample.
        Each sub-model score is normalised using *training-time* statistics
        so the result is stable even when X_scaled contains only one row.
        A more negative combined score → more anomalous.
        Final score is clipped to [-10, 10] to prevent blow-up from
        out-of-distribution features.
        """
        scores = []

        s_if  = self.isolation_forest.decision_function(X_scaled)
        s_lof = self.lof.decision_function(X_scaled)
        scores.append(self._normalise_with_stats(s_if,  'if'))
        scores.append(self._normalise_with_stats(s_lof, 'lof'))

        if self.elliptic_env is not None:
            s_ee = self.elliptic_env.decision_function(X_scaled)
            scores.append(self._normalise_with_stats(s_ee, 'ee'))
            w = self.ensemble_weights
        else:
            w = self.ensemble_weights[:2] / self.ensemble_weights[:2].sum()

        stacked  = np.column_stack(scores)
        combined = stacked @ w

        # Hard clip — prevents NaN/Inf propagation into EMA and alerts
        return np.clip(combined, -10.0, 10.0)

    # ─────────────────────────────────────────────────────────────────
    # ANOMALY DETECTION  (per-frame, online)
    # ─────────────────────────────────────────────────────────────────
    def detect_anomaly(self, amplitude_segment, phase_segment):
        """Detect anomalies using the trained ensemble."""
        if not self.is_calibrated or self.isolation_forest is None:
            return False, 0.0

        features = self.extract_basic_features(amplitude_segment, phase_segment)
        X_scaled = self.anomaly_scaler.transform(features.reshape(1, -1))
        raw_score = float(self._combined_score(X_scaled)[0])

        # Smooth with EMA (α = 0.3 → prefer recent frames)
        alpha = 0.3
        if self.anomaly_scores:
            smoothed = alpha * raw_score + (1 - alpha) * self.anomaly_scores[-1]
        else:
            smoothed = raw_score

        self.anomaly_scores.append(smoothed)

        # Adapt threshold periodically using scores that look clean
        if smoothed > self.anomaly_threshold and len(self.clean_score_buffer) % 20 == 0:
            self.clean_score_buffer.append(smoothed)
            if len(self.clean_score_buffer) >= 50:
                self.anomaly_threshold = float(np.percentile(
                    list(self.clean_score_buffer), self.adaptive_threshold_percentile
                ))

        is_anomaly = smoothed < self.anomaly_threshold
        return is_anomaly, smoothed

    # ─────────────────────────────────────────────────────────────────
    # ANOMALY VALIDATION (multi-criteria)
    # ─────────────────────────────────────────────────────────────────
    def validate_anomaly(self, anomaly_score, amplitude_segment, consecutive_threshold=3):
        validation_flags = []

        if anomaly_score < self.anomaly_threshold:
            validation_flags.append("score_threshold")

        self.recent_anomalies.append(anomaly_score < self.anomaly_threshold)
        if sum(self.recent_anomalies) >= consecutive_threshold:
            validation_flags.append("consecutive")

        if amplitude_segment.shape[0] > 5:
            recent_std   = np.std(amplitude_segment[-5:])
            baseline_std = np.std(amplitude_segment[:5]) if len(amplitude_segment) > 10 else 0
            if recent_std > baseline_std * 2.0:
                validation_flags.append("amplitude_variance")

        if self.rf_params_buffer and len(self.rf_params_buffer) > 0:
            recent_rssi = [p.get('rssi', -100) for p in list(self.rf_params_buffer)[-5:]]
            if len(recent_rssi) >= 3 and np.std(recent_rssi) < 5:
                validation_flags.append("stable_rf")

        return len(validation_flags) >= 2, validation_flags

    # ─────────────────────────────────────────────────────────────────
    # CWT FEATURE EXTRACTION  (selective – anomaly segments only)
    # ─────────────────────────────────────────────────────────────────
    def _cwt_processing_thread(self, amplitude_segment):
        try:
            avg_amplitude = np.mean(amplitude_segment, axis=1)
            clean_signal  = self.hampel_filter(avg_amplitude)

            coefficients, frequencies = pywt.cwt(
                clean_signal, self.cwt_scales, self.cwt_wavelet
            )

            cwt_features = []
            scale_energies = np.sum(coefficients ** 2, axis=1)
            cwt_features.extend([
                np.mean(scale_energies), np.std(scale_energies),
                np.max(scale_energies),  np.min(scale_energies),
                np.median(scale_energies), np.sum(scale_energies),
            ])

            coeff_flat = coefficients.flatten()
            if len(coeff_flat) > 0:
                cwt_features.extend([
                    np.mean(coeff_flat), np.std(coeff_flat),
                    stats.skew(coeff_flat)     if len(coeff_flat) > 2 else 0,
                    stats.kurtosis(coeff_flat) if len(coeff_flat) > 3 else 0,
                    np.percentile(coeff_flat, 90),
                    np.percentile(coeff_flat, 10),
                    np.median(np.abs(coeff_flat)),
                ])

            cwt_features.extend([
                np.mean(frequencies), np.std(frequencies),
                np.max(frequencies),  np.min(frequencies),
            ])

            with self.processing_lock:
                self.processing_result = (np.array(cwt_features), coefficients)

        except Exception as e:
            print(f"CWT thread error: {e}")
            with self.processing_lock:
                self.processing_result = (None, None)

    def extract_cwt_features_selective(self, amplitude_segment, anomaly_score):
        start_time = time.time()

        if not self.cwt_enabled:
            return None, None, 0

        current_time = time.time()
        if current_time - self.last_anomaly_time < self.anomaly_cooldown:
            self.cwt_skip_count += 1
            return None, None, 0

        anomaly_severity = abs(anomaly_score - self.anomaly_threshold) / (abs(self.anomaly_threshold) + 1e-9)
        if anomaly_severity < 0.5:
            return None, None, 0

        is_valid, _ = self.validate_anomaly(anomaly_score, amplitude_segment)
        if not is_valid:
            self.cwt_skip_count += 1
            return None, None, 0

        try:
            self.cwt_processing_count += 1
            self.last_anomaly_time = current_time

            if self.processing_thread is None or not self.processing_thread.is_alive():
                self.processing_thread = threading.Thread(
                    target=self._cwt_processing_thread,
                    args=(amplitude_segment,),
                    daemon=True
                )
                self.processing_thread.start()

            with self.processing_lock:
                if self.processing_result is not None:
                    cwt_features, coefficients = self.processing_result
                    self.processing_result = None
                else:
                    cwt_features, coefficients = None, None

            if coefficients is not None:
                self.cwt_coefficients_history.append(coefficients)
                self.cwt_features_history.append(cwt_features)

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            return cwt_features, coefficients, processing_time

        except Exception as e:
            print(f"CWT extraction error: {e}")
            return None, None, 0

    # ─────────────────────────────────────────────────────────────────
    # MAIN FRAME PROCESSING
    # ─────────────────────────────────────────────────────────────────
    def process_frame(self, amplitude_frame, phase_frame, rf_params):
        self.amplitude_buffer.append(amplitude_frame)
        self.phase_buffer.append(phase_frame)
        self.rf_params_buffer.append(rf_params)

        if len(self.amplitude_buffer) < self.window_size:
            return None, None, 0.0, False, None

        amplitude_segment = np.array(self.amplitude_buffer)
        phase_segment     = np.array(self.phase_buffer)

        is_anomaly, anomaly_score = self.detect_anomaly(amplitude_segment, phase_segment)

        # Save basic features for labelling
        if is_anomaly:
            basic_feats = self.extract_basic_features(amplitude_segment, phase_segment)
            self.basic_features_history.append(basic_feats)

        cwt_features = cwt_coefficients = None

        if is_anomaly and self.cwt_enabled:
            cwt_features, cwt_coefficients, _ = self.extract_cwt_features_selective(
                amplitude_segment, anomaly_score
            )
            if cwt_features is not None:
                self.current_anomaly_segment = amplitude_segment
                if not self.in_anomaly_window:
                    self.in_anomaly_window = True
                    self.anomaly_start_idx = len(self.amplitude_buffer) - 1
                    print(f"🚨 Anomaly detected! Score: {anomaly_score:.4f}")
        else:
            if self.in_anomaly_window:
                self.in_anomaly_window = False

        return cwt_features, cwt_coefficients, is_anomaly, anomaly_score, amplitude_segment

    # ─────────────────────────────────────────────────────────────────
    # STATE PERSISTENCE
    # ─────────────────────────────────────────────────────────────────
    def save_pipeline_state(self, filepath="pipeline_state.joblib"):
        state = {
            'isolation_forest':   self.isolation_forest,
            'lof':                self.lof,
            'elliptic_env':       self.elliptic_env,
            'anomaly_scaler':     self.anomaly_scaler,
            'empty_room_features': self.empty_room_features,
            'is_calibrated':      self.is_calibrated,
            'calibration_samples': self.calibration_samples,
            'anomaly_threshold':  self.anomaly_threshold,
            'ensemble_weights':   self.ensemble_weights,
            '_score_stats':       self._score_stats,
            'window_size':        self.window_size,
            'cwt_scales':         self.cwt_scales,
            'cwt_wavelet':        self.cwt_wavelet,
            'cwt_processing_count': self.cwt_processing_count,
            'cwt_skip_count':     self.cwt_skip_count,
        }
        joblib.dump(state, filepath)
        print(f"✅ Pipeline state saved to {filepath}")
        return True

    def load_pipeline_state(self, filepath="pipeline_state.joblib"):
        if os.path.exists(filepath):
            try:
                state = joblib.load(filepath)
                for key, value in state.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                print(f"✅ Pipeline state loaded from {filepath}")
                return True
            except Exception as e:
                print(f"❌ Error loading pipeline state: {e}")
        return False

    # ─────────────────────────────────────────────────────────────────
    # PERFORMANCE METRICS
    # ─────────────────────────────────────────────────────────────────
    def get_performance_metrics(self):
        total_anomalies = len([s for s in self.anomaly_scores if s < self.anomaly_threshold])
        total_scores    = max(len(self.anomaly_scores), 1)

        return {
            'total_frames':       len(self.amplitude_buffer),
            'anomaly_rate':       total_anomalies / total_scores,
            'avg_processing_time': np.mean(list(self.processing_times)) if self.processing_times else 0,
            'memory_usage_mb':    self.get_memory_usage(),
            'buffer_utilization': len(self.amplitude_buffer) / self.amplitude_buffer.maxlen,
        }

    def get_memory_usage(self):
        total = 0
        if self.amplitude_buffer:
            total += sys.getsizeof(list(self.amplitude_buffer)) / 1024 / 1024
        if self.cwt_coefficients_history:
            for coeff in self.cwt_coefficients_history:
                if hasattr(coeff, 'nbytes'):
                    total += coeff.nbytes / 1024 / 1024
        return round(total, 2)


# =========================================================
# UTILS
# =========================================================
def generate_subcarrier_colors(red_range, green_range, yellow_range, total_num, interval=1):
    colors = []
    for i in range(total_num):
        if red_range and red_range[0] <= i <= red_range[1]:
            intensity = int(255 * (i - red_range[0]) / (red_range[1] - red_range[0] or 1))
            colors.append((intensity, 0, 0))
        elif green_range and green_range[0] <= i <= green_range[1]:
            intensity = int(255 * (i - green_range[0]) / (green_range[1] - green_range[0] or 1))
            colors.append((0, intensity, 0))
        elif yellow_range and yellow_range[0] <= i <= yellow_range[1]:
            intensity = int(255 * (i - yellow_range[0]) / (yellow_range[1] - yellow_range[0] or 1))
            colors.append((0, intensity, intensity))
        else:
            colors.append((200, 200, 200))
    return colors


def ensure_global_buffers(n_complex_cols):
    global csi_data_complex, agc_gain_data, fft_gain_data
    if csi_data_complex is None or csi_data_complex.shape[1] != n_complex_cols:
        csi_data_complex = np.zeros((CSI_DATA_INDEX, n_complex_cols), dtype=np.complex64)
        agc_gain_data    = np.zeros((CSI_DATA_INDEX,), dtype=np.float64)
        fft_gain_data    = np.zeros((CSI_DATA_INDEX,), dtype=np.float64)
        print(f"[INIT] allocated global buffers for {n_complex_cols} complex subcarriers")


# =========================================================
# SERIAL PARSER
# =========================================================
def csi_data_read_parse(port: str, csv_writer, log_file_fd, callback=None):
    global seen_lengths
    ser = serial.Serial(port=port, baudrate=921600, bytesize=8, parity='N', stopbits=1)
    if not ser.isOpen():
        print("open failed")
        return

    print("open success")
    frame_idx       = 0
    sent_colors_once = False

    while True:
        try:
            line = str(ser.readline())
            if not line:
                break

            line = line.lstrip("b'").rstrip("\\r\\n'")

            if "CSI_DATA" not in line:
                log_file_fd.write(line + "\n")
                log_file_fd.flush()
                continue

            csv_reader   = csv.reader(StringIO(line))
            csi_data     = next(csv_reader)
            csi_data_len = int(csi_data[-3])

            if csi_data_len not in seen_lengths:
                seen_lengths.add(csi_data_len)
                print(f"[CSI] new raw length detected: {csi_data_len} → complex={csi_data_len // 2}")

            if len(csi_data) == len(DATA_COLUMNS_NAMES_C5C6):
                columns = DATA_COLUMNS_NAMES_C5C6
                is_c5c6 = True
            elif len(csi_data) == len(DATA_COLUMNS_NAMES):
                columns = DATA_COLUMNS_NAMES
                is_c5c6 = False
            else:
                print("element number is not equal", len(csi_data), len(DATA_COLUMNS_NAMES))
                log_file_fd.write("element number is not equal\n")
                log_file_fd.write(line + "\n")
                log_file_fd.flush()
                continue

            params = {}
            for i, col_name in enumerate(columns):
                if i < len(csi_data):
                    try:
                        if col_name in ["rssi", "rate", "mcs", "bandwidth", "noise_floor",
                                        "fft_gain", "agc_gain", "channel", "sig_len", "len",
                                        "ampdu_cnt", "ant", "secondary_channel", "sig_mode"]:
                            params[col_name] = int(csi_data[i])
                        else:
                            params[col_name] = csi_data[i]
                    except:
                        params[col_name] = csi_data[i]
                else:
                    params[col_name] = ""

            try:
                csi_raw_data = json.loads(csi_data[-1])
            except json.JSONDecodeError:
                print("data is incomplete (json)")
                log_file_fd.write("data is incomplete\n")
                log_file_fd.write(line + "\n")
                log_file_fd.flush()
                continue

            if csi_data_len != len(csi_raw_data):
                print("csi_data_len is not equal", csi_data_len, len(csi_raw_data))
                log_file_fd.write("csi_data_len is not equal\n")
                log_file_fd.write(line + "\n")
                log_file_fd.flush()
                continue

            n_complex = csi_data_len // 2
            ensure_global_buffers(n_complex)

            fft_gain = params.get('fft_gain', 0) if is_c5c6 else 0
            agc_gain = params.get('agc_gain', 0) if is_c5c6 else 0

            csv_writer.writerow(csi_data)

            csi_data_complex[:-1] = csi_data_complex[1:]
            agc_gain_data[:-1]    = agc_gain_data[1:]
            fft_gain_data[:-1]    = fft_gain_data[1:]

            agc_gain_data[-1] = agc_gain
            fft_gain_data[-1] = fft_gain

            for i in range(n_complex):
                re = csi_raw_data[i * 2 + 1]
                im = csi_raw_data[i * 2]
                csi_data_complex[-1, i] = complex(re, im)

            extended_metadata = {
                'frame_idx':    frame_idx,
                'local_ts':     params.get('local_timestamp', ''),
                'csi_data_len': csi_data_len,
                'params':       params,
            }

            if not sent_colors_once:
                sent_colors_once = True
                if   csi_data_len == 106:  colors = generate_subcarrier_colors((0, 25),  (27, 53),   None,       len(csi_raw_data))
                elif csi_data_len == 114:  colors = generate_subcarrier_colors((0, 27),  (29, 56),   None,       len(csi_raw_data))
                elif csi_data_len == 52:   colors = generate_subcarrier_colors((0, 12),  (13, 26),   None,       len(csi_raw_data))
                elif csi_data_len == 234:  colors = generate_subcarrier_colors((0, 28),  (29, 56),   (60, 116),  len(csi_raw_data))
                elif csi_data_len == 490:  colors = generate_subcarrier_colors((0, 61),  (62, 122),  (123, 245), len(csi_raw_data))
                elif csi_data_len == 128:  colors = generate_subcarrier_colors((0, 31),  (32, 63),   None,       len(csi_raw_data))
                elif csi_data_len == 256:  colors = generate_subcarrier_colors((0, 32),  (32, 63),   (64, 128),  len(csi_raw_data))
                elif csi_data_len == 512:  colors = generate_subcarrier_colors((0, 63),  (64, 127),  (128, 256), len(csi_raw_data))
                elif csi_data_len == 384:  colors = generate_subcarrier_colors((0, 63),  (64, 127),  (128, 192), len(csi_raw_data))
                else:
                    print(f"Please add more color schemes for length: {csi_data_len}")
                    colors = None

                extended_metadata['colors'] = colors

            if callback is not None:
                callback(extended_metadata)

            frame_idx += 1

        except Exception as e:
            print(f"Error in serial reading: {e}")
            traceback.print_exc()
            break

    ser.close()


# =========================================================
# THREAD WRAPPER
# =========================================================
class SubThread(QThread):
    data_ready = pyqtSignal(object)

    def __init__(self, serial_port, save_file_name, log_file_name):
        super().__init__()
        self.serial_port    = serial_port
        self.save_file_name = save_file_name
        self.log_file_name  = log_file_name
        self.save_file_fd   = None
        self.log_file_fd    = None
        self.csv_writer     = None

    def run(self):
        try:
            self.save_file_fd = open(self.save_file_name, 'w', newline='')
            self.log_file_fd  = open(self.log_file_name,  'w')
            self.csv_writer   = csv.writer(self.save_file_fd)
            self.csv_writer.writerow(DATA_COLUMNS_NAMES)
            csi_data_read_parse(self.serial_port, self.csv_writer, self.log_file_fd,
                                callback=self.data_ready.emit)
        except Exception as e:
            print(f"Thread error: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            if self.save_file_fd: self.save_file_fd.close()
            if self.log_file_fd:  self.log_file_fd.close()
        except Exception as e:
            print(f"Error closing files: {e}")

    def __del__(self):
        self.cleanup()


# =========================================================
# ENHANCED GUI
# =========================================================
class EnhancedCSIGraphicalWindow(QWidget):
    def __init__(self):
        super().__init__()

        screen          = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        window_width    = int(screen_geometry.width()  * 0.85)
        window_height   = int(screen_geometry.height() * 0.85)

        self.setWindowTitle("Enhanced CSI Viewer v3.0 – Ensemble Anomaly Detection")
        self.resize(window_width, window_height)

        # State variables
        self.recording_session   = False
        self.session_start_time  = None
        self.event_count         = 0
        self.current_session_data = []
        self.current_frame_idx   = 0
        self.current_timestamp   = ""
        self.actual_csi_columns  = None
        self.curve_list          = []
        self.iq_real_curves      = []
        self.iq_imag_curves      = []
        self.iq_colors           = []
        self.deta_len            = 0
        self.current_rf_params   = {}
        self.fps_counter         = 0
        self.last_fps_time       = time.time()

        # Signal processing pipeline
        self.signal_pipeline    = EnhancedSignalProcessingPipeline(window_size=30)
        self.calibrating        = False
        self.calibration_progress = 0

        # Spectrogram state
        self.spectrogram_n_frames      = 150
        self.spectrogram_buffer        = None
        self.spectrogram_subcarrier_range = (0, None)

        # Alert state
        self.alert_enabled          = True
        self.alert_sound_enabled    = True
        self.alert_count            = 0
        self.last_alert_time        = 0.0
        self.alert_cooldown_sec     = 2.0
        self.alert_severity_thresholds = {'low': 0.5, 'medium': 1.0, 'high': 1.5}

        # Auto-mark timer (fires every 1 second when enabled)
        self.auto_mark_timer = QTimer()
        self.auto_mark_timer.timeout.connect(self.auto_mark_event)

        # Model state (legacy single-model)
        self.model           = None
        self.scaler          = None
        self.feature_columns = None
        self.label_encoder   = None
        self.model_path      = None

        # ── 4T AI Engine state ────────────────────────────────────────
        self._4t_engine = None
        self._4t_eliminator = None
        self._4t_loaded = False
        self._4t_enabled = False
        self._4t_model_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '4T_opt', 'models')
        self._4t_last_result = {}
        self._4t_inference_thread = None
        self._4t_inference_lock = threading.Lock()
        self._4t_processing = False  # guard against re-entry
        self._4t_fs = 100.0  # assumed sampling frequency

        # Build UI
        self.setup_plots()
        self.setup_iq_plots()
        self.setup_signal_processing_panel()
        self.setup_data_collection_panel()
        self.setup_real_time_detection_panel()
        self.setup_4t_ai_panel()
        self.setup_rf_parameters_display()
        self.setup_statistics_panel()
        self.setup_spectrogram_panel()
        self.setup_alert_system()
        self.setup_layout()

        self.connect_signals()
        self.setup_timers()

    # ─────────────────────────────────────────────────────────────────
    # TIMERS
    # ─────────────────────────────────────────────────────────────────
    def setup_timers(self):
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)

        self.processing_timer = pg.QtCore.QTimer()
        self.processing_timer.timeout.connect(self.run_signal_processing)
        self.processing_timer.start(200)

        self.stats_timer = pg.QtCore.QTimer()
        self.stats_timer.timeout.connect(self.update_statistics)
        self.stats_timer.start(1000)

        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.cleanup_memory)
        self.memory_timer.start(300000)

    def cleanup_memory(self):
        print("🧹 Performing memory cleanup...")
        if hasattr(self, 'signal_pipeline'):
            self.signal_pipeline.cwt_coefficients_history.clear()
            self.signal_pipeline.cwt_features_history.clear()
            self.signal_pipeline.basic_features_history.clear()
            self.signal_pipeline.anomaly_scores.clear()
            self.signal_pipeline.processing_times.clear()
            self.signal_pipeline.recent_anomalies.clear()
        gc.collect()
        print("✅ Memory cleanup completed")
        self.memory_label.setText(f"Memory: {self.signal_pipeline.get_memory_usage()} MB (Cleaned)")

    # ─────────────────────────────────────────────────────────────────
    # MAIN PLOTS  (amplitude over time + anomaly)
    # ─────────────────────────────────────────────────────────────────
    def setup_plots(self):
        # Amplitude over time
        self.plotWidget_multi_data = pg.PlotWidget()
        self.plotWidget_multi_data.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.plotWidget_multi_data.addLegend()
        self.plotWidget_multi_data.setTitle("Subcarrier Amplitude Over Time")
        self.plotWidget_multi_data.setLabel('left', 'Amplitude')
        self.plotWidget_multi_data.setLabel('bottom', 'Time (packet index)')

        # Anomaly score
        self.plotWidget_anomaly = pg.PlotWidget()
        self.plotWidget_anomaly.addLegend()
        self.plotWidget_anomaly.setTitle("Ensemble Anomaly Score")
        self.plotWidget_anomaly.setLabel('left', 'Combined Score')
        self.plotWidget_anomaly.setLabel('bottom', 'Frame Index')
        self.plotWidget_anomaly.setYRange(-3, 3)
        self.anomaly_curve    = self.plotWidget_anomaly.plot([], name="Score", pen='y')
        self.threshold_line   = pg.InfiniteLine(pos=-0.5, angle=0, pen='r', label="Threshold")
        self.plotWidget_anomaly.addItem(self.threshold_line)

    # ─────────────────────────────────────────────────────────────────
    # IQ REAL / IMAGINARY PLOTS  (new)
    # ─────────────────────────────────────────────────────────────────
    def setup_iq_plots(self):
        """Two plots showing the real (I) and imaginary (Q) parts of the CSI over time."""
        self.iq_group = QGroupBox("📶 IQ Data — Real (I) & Imaginary (Q)")
        layout = QVBoxLayout()

        # Real part
        self.plotWidget_iq_real = pg.PlotWidget()
        self.plotWidget_iq_real.setTitle("I (Real) Component – Subcarriers Over Time")
        self.plotWidget_iq_real.setLabel('left', 'Real Amplitude')
        self.plotWidget_iq_real.setLabel('bottom', 'Packet Index')
        self.plotWidget_iq_real.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.plotWidget_iq_real.addLegend()
        self.plotWidget_iq_real.setMinimumHeight(200)

        # Imaginary part
        self.plotWidget_iq_imag = pg.PlotWidget()
        self.plotWidget_iq_imag.setTitle("Q (Imaginary) Component – Subcarriers Over Time")
        self.plotWidget_iq_imag.setLabel('left', 'Imaginary Amplitude')
        self.plotWidget_iq_imag.setLabel('bottom', 'Packet Index')
        self.plotWidget_iq_imag.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.plotWidget_iq_imag.addLegend()
        self.plotWidget_iq_imag.setMinimumHeight(200)

        # IQ freeze / link axes toggle
        ctrl_layout = QHBoxLayout()
        self.iq_freeze_btn = QPushButton("❄ Freeze IQ")
        self.iq_freeze_btn.setCheckable(True)
        self.iq_freeze_btn.setStyleSheet("QPushButton:checked { background-color: #1565C0; color: white; }")
        self.iq_link_btn   = QPushButton("🔗 Link Y-Axes")
        self.iq_link_btn.setCheckable(True)
        self.iq_link_btn.toggled.connect(self._toggle_iq_link)
        self.iq_n_sub_spin = QSpinBox()
        self.iq_n_sub_spin.setRange(1, 20)
        self.iq_n_sub_spin.setValue(8)
        self.iq_n_sub_spin.setPrefix("Show ")
        self.iq_n_sub_spin.setSuffix(" subcarriers")

        ctrl_layout.addWidget(self.iq_freeze_btn)
        ctrl_layout.addWidget(self.iq_link_btn)
        ctrl_layout.addWidget(self.iq_n_sub_spin)
        ctrl_layout.addStretch()

        layout.addLayout(ctrl_layout)
        layout.addWidget(self.plotWidget_iq_real)
        layout.addWidget(self.plotWidget_iq_imag)
        self.iq_group.setLayout(layout)

    def _toggle_iq_link(self, checked):
        if checked:
            self.plotWidget_iq_imag.setYLink(self.plotWidget_iq_real)
        else:
            self.plotWidget_iq_imag.setYLink(None)

    def _init_iq_curves(self, n_cols):
        """Create IQ curve objects (called when we know the number of subcarriers)."""
        for c in self.iq_real_curves:
            try: self.plotWidget_iq_real.removeItem(c)
            except: pass
        for c in self.iq_imag_curves:
            try: self.plotWidget_iq_imag.removeItem(c)
            except: pass

        self.iq_real_curves = []
        self.iq_imag_curves = []

        n_show = min(n_cols, 20)
        for i in range(n_show):
            hue   = int(360 * i / n_show)
            color = pg.mkColor(QColor.fromHsv(hue, 200, 220))
            pen   = pg.mkPen(color=color, width=1)
            r_curve = self.plotWidget_iq_real.plot([], pen=pen, name=f"Sub {i}")
            q_curve = self.plotWidget_iq_imag.plot([], pen=pen, name=f"Sub {i}")
            self.iq_real_curves.append(r_curve)
            self.iq_imag_curves.append(q_curve)

    def update_iq_plots(self):
        """Push fresh real/imag data to the IQ plots."""
        if csi_data_complex is None or self.actual_csi_columns is None:
            return
        if self.iq_freeze_btn.isChecked():
            return

        try:
            n_show = min(self.iq_n_sub_spin.value(), self.actual_csi_columns,
                         len(self.iq_real_curves))
            for i in range(n_show):
                if i < csi_data_complex.shape[1]:
                    real_data = csi_data_complex[:, i].real
                    imag_data = csi_data_complex[:, i].imag
                    self.iq_real_curves[i].setData(real_data)
                    self.iq_imag_curves[i].setData(imag_data)
        except Exception as e:
            print(f"IQ plot update error: {e}")

    # ─────────────────────────────────────────────────────────────────
    # SPECTROGRAM / WATERFALL
    # ─────────────────────────────────────────────────────────────────
    def setup_spectrogram_panel(self):
        self.spectrogram_group = QGroupBox("📡 Live Spectrogram / Waterfall")
        layout = QVBoxLayout()

        self.spectrogram_plot = pg.PlotWidget()
        self.spectrogram_plot.setTitle("CSI Amplitude Waterfall (time → right)")
        self.spectrogram_plot.setLabel('left', 'Subcarrier Index')
        self.spectrogram_plot.setLabel('bottom', 'Time Frame')
        self.spectrogram_plot.setMinimumHeight(180)

        self.spectrogram_image = pg.ImageItem()
        self.spectrogram_image.setLookupTable(self.create_spectrogram_lut())
        self.spectrogram_plot.addItem(self.spectrogram_image)

        self.spectrogram_legend = pg.GradientWidget(orientation='right')
        self.spectrogram_legend.setFixedWidth(20)
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(self.spectrogram_plot)
        legend_layout.addWidget(self.spectrogram_legend)
        legend_widget = QWidget()
        legend_widget.setLayout(legend_layout)

        ctrl_layout = QHBoxLayout()
        self.spec_frames_spin = QSpinBox()
        self.spec_frames_spin.setRange(50, 500)
        self.spec_frames_spin.setValue(self.spectrogram_n_frames)
        self.spec_frames_spin.setSuffix(" frames")
        self.spec_frames_spin.valueChanged.connect(self._on_spec_frames_changed)

        self.spec_freeze_btn = QPushButton("❄ Freeze")
        self.spec_freeze_btn.setCheckable(True)
        self.spec_freeze_btn.setStyleSheet("QPushButton:checked { background-color: #1565C0; color: white; }")

        self.spec_reset_btn = QPushButton("🔄 Reset")
        self.spec_reset_btn.clicked.connect(self._reset_spectrogram)

        self.spec_info_label = QLabel("Subcarriers: —  |  Range: full")
        self.spec_info_label.setStyleSheet("color: #888; font-size: 10px;")

        ctrl_layout.addWidget(QLabel("History:"))
        ctrl_layout.addWidget(self.spec_frames_spin)
        ctrl_layout.addWidget(self.spec_freeze_btn)
        ctrl_layout.addWidget(self.spec_reset_btn)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.spec_info_label)

        layout.addWidget(legend_widget)
        layout.addLayout(ctrl_layout)
        self.spectrogram_group.setLayout(layout)

    def create_spectrogram_lut(self, n=256):
        lut = np.zeros((n, 3), dtype=np.uint8)
        for i in range(n):
            t = i / (n - 1)
            if t < 0.4:
                r = int(t / 0.4 * 180); g = 0; b = int(t / 0.4 * 60)
            elif t < 0.7:
                p = (t - 0.4) / 0.3
                r = int(180 + p * 75); g = int(p * 80); b = int(60 * (1 - p))
            else:
                p = (t - 0.7) / 0.3
                r = 255; g = int(80 + p * 175); b = int(p * 255)
            lut[i] = [min(r, 255), min(g, 255), min(b, 255)]
        return lut

    def _on_spec_frames_changed(self, value):
        self.spectrogram_n_frames = value
        self.spectrogram_buffer   = None

    def _reset_spectrogram(self):
        self.spectrogram_buffer = None
        self.spectrogram_image.clear()

    def update_spectrogram(self, amp_arr):
        if self.spec_freeze_btn.isChecked():
            return
        try:
            n_sub    = amp_arr.shape[1]
            n_frames = self.spectrogram_n_frames

            if self.spectrogram_buffer is None or self.spectrogram_buffer.shape != (n_frames, n_sub):
                self.spectrogram_buffer = np.zeros((n_frames, n_sub), dtype=np.float32)
                self.spec_info_label.setText(f"Subcarriers: {n_sub}  |  Frames: {n_frames}")

            self.spectrogram_buffer[:-1] = self.spectrogram_buffer[1:]
            self.spectrogram_buffer[-1]  = amp_arr[-1, :n_sub]

            buf  = self.spectrogram_buffer
            vmin, vmax = buf.min(), buf.max()
            display = ((buf - vmin) / (vmax - vmin) * 255).astype(np.uint8) if vmax > vmin \
                      else np.zeros_like(buf, dtype=np.uint8)

            self.spectrogram_image.setImage(display, autoLevels=False, levels=(0, 255))
        except Exception as e:
            print(f"Spectrogram update error: {e}")

    # ─────────────────────────────────────────────────────────────────
    # ALERT SYSTEM
    # ─────────────────────────────────────────────────────────────────
    def setup_alert_system(self):
        self.alert_group = QGroupBox("🔔 Anomaly Alert System")
        layout = QVBoxLayout()

        self.alert_enable_cb = QCheckBox("Enable Anomaly Alerts")
        self.alert_enable_cb.setChecked(True)
        self.alert_enable_cb.stateChanged.connect(self._on_alert_toggle)

        self.alert_sound_cb = QCheckBox("🔊 Play Sound on Alert")
        self.alert_sound_cb.setChecked(True)
        self.alert_sound_cb.stateChanged.connect(self._on_alert_sound_toggle)

        self.alert_indicator = QLabel("  NO ANOMALY  ")
        self.alert_indicator.setAlignment(Qt.AlignCenter)
        self.alert_indicator.setStyleSheet(
            "background-color: #1B5E20; color: white; font-weight: bold; "
            "font-size: 14px; border-radius: 6px; padding: 6px;")

        self.alert_severity_label = QLabel("Severity: —")
        self.alert_severity_label.setAlignment(Qt.AlignCenter)

        cooldown_layout = QHBoxLayout()
        cooldown_layout.addWidget(QLabel("Alert cooldown (s):"))
        self.alert_cooldown_spin = QDoubleSpinBox()
        self.alert_cooldown_spin.setRange(0.5, 30.0)
        self.alert_cooldown_spin.setSingleStep(0.5)
        self.alert_cooldown_spin.setValue(self.alert_cooldown_sec)
        self.alert_cooldown_spin.valueChanged.connect(lambda v: setattr(self, 'alert_cooldown_sec', v))
        cooldown_layout.addWidget(self.alert_cooldown_spin)

        self.alert_count_label = QLabel("Total alerts: 0")
        self.alert_count_label.setStyleSheet("font-weight: bold;")

        self.alert_log = QListWidget()
        self.alert_log.setMaximumHeight(160)
        self.alert_log.setAlternatingRowColors(True)

        self.clear_alerts_btn = QPushButton("🗑 Clear Alert Log")
        self.clear_alerts_btn.clicked.connect(self._clear_alert_log)

        layout.addWidget(self.alert_enable_cb)
        layout.addWidget(self.alert_sound_cb)
        layout.addWidget(self.alert_indicator)
        layout.addWidget(self.alert_severity_label)
        layout.addLayout(cooldown_layout)
        layout.addWidget(self.alert_count_label)
        layout.addWidget(QLabel("Alert Log:"))
        layout.addWidget(self.alert_log)
        layout.addWidget(self.clear_alerts_btn)
        self.alert_group.setLayout(layout)

        self._alert_reset_timer = QTimer()
        self._alert_reset_timer.setSingleShot(True)
        self._alert_reset_timer.timeout.connect(self._reset_alert_indicator)

    def _on_alert_toggle(self, state):   self.alert_enabled       = (state == Qt.Checked)
    def _on_alert_sound_toggle(self, s): self.alert_sound_enabled = (s     == Qt.Checked)

    def _clear_alert_log(self):
        self.alert_log.clear()
        self.alert_count = 0
        self.alert_count_label.setText("Total alerts: 0")

    def _reset_alert_indicator(self):
        self.alert_indicator.setText("  NO ANOMALY  ")
        self.alert_indicator.setStyleSheet(
            "background-color: #1B5E20; color: white; font-weight: bold; "
            "font-size: 14px; border-radius: 6px; padding: 6px;")
        self.alert_severity_label.setText("Severity: —")

    def trigger_alert(self, anomaly_score):
        if not self.alert_enabled:
            return
        now = time.time()
        if now - self.last_alert_time < self.alert_cooldown_sec:
            return

        self.last_alert_time = now
        self.alert_count    += 1

        threshold = self.signal_pipeline.anomaly_threshold
        distance  = abs(anomaly_score - threshold) / (abs(threshold) + 1e-9) if threshold != 0 else abs(anomaly_score)

        if distance >= self.alert_severity_thresholds['high']:
            severity, bg_color, text_color = "HIGH",   "#B71C1C", "white"
        elif distance >= self.alert_severity_thresholds['medium']:
            severity, bg_color, text_color = "MEDIUM", "#E65100", "white"
        else:
            severity, bg_color, text_color = "LOW",    "#F57F17", "black"

        self.alert_indicator.setText("  ⚠  ANOMALY DETECTED  ⚠  ")
        self.alert_indicator.setStyleSheet(
            f"background-color: {bg_color}; color: {text_color}; "
            f"font-weight: bold; font-size: 14px; border-radius: 6px; padding: 6px;")
        self.alert_severity_label.setText(f"Severity: {severity}  |  Score: {anomaly_score:.4f}")
        self.alert_count_label.setText(f"Total alerts: {self.alert_count}")

        ts    = datetime.now().strftime("%H:%M:%S")
        item  = QListWidgetItem(f"{ts}  [{severity}]  score={anomaly_score:.4f}")
        color_map = {"HIGH": "#FF5252", "MEDIUM": "#FF9800", "LOW": "#FFC107"}
        item.setForeground(QColor(color_map[severity]))
        self.alert_log.insertItem(0, item)
        while self.alert_log.count() > 200:
            self.alert_log.takeItem(self.alert_log.count() - 1)

        if self.alert_sound_enabled:
            try: QApplication.beep()
            except: pass

        self._alert_reset_timer.start(2000)
        print(f"🔔 Alert [{severity}]: score={anomaly_score:.4f}")

    # ─────────────────────────────────────────────────────────────────
    # SIGNAL PROCESSING PANEL
    # ─────────────────────────────────────────────────────────────────
    def setup_signal_processing_panel(self):
        self.processing_group = QGroupBox("Signal Processing Pipeline")
        layout = QVBoxLayout()

        self.calibrate_btn = QPushButton("🎯 Calibrate Empty Room (Ensemble)")
        self.calibrate_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.calibration_status = QLabel("Calibration: Not started")
        self.calibration_progress_bar = QProgressBar()
        self.calibration_progress_bar.setRange(0, 100)

        self.hampel_checkbox = QCheckBox("🔧 Enable Hampel Filtering")
        self.hampel_checkbox.setChecked(True)
        self.hampel_window_spin = QSpinBox()
        self.hampel_window_spin.setRange(3, 15)
        self.hampel_window_spin.setValue(5)
        self.hampel_window_spin.setSuffix(" window")

        self.anomaly_threshold_slider = QSlider(Qt.Horizontal)
        self.anomaly_threshold_slider.setRange(-300, 0)
        self.anomaly_threshold_slider.setValue(-50)
        self.anomaly_threshold_label = QLabel("Threshold: -0.50")

        # Adaptive threshold info
        self.adaptive_thresh_label = QLabel("Adaptive threshold: enabled")
        self.adaptive_thresh_label.setStyleSheet("color: #00BCD4; font-size: 10px;")

        self.cwt_checkbox = QCheckBox("🌀 Enable Selective CWT (for classifier features)")
        self.cwt_checkbox.setChecked(True)

        self.ensemble_info_label = QLabel("Models: IF + LOF + EE")
        self.ensemble_info_label.setStyleSheet("color: #8BC34A; font-size: 10px;")

        self.processing_status = QLabel("🟡 Status: Idle")
        self.anomaly_status    = QLabel("🟢 Anomaly: None")
        self.segment_info      = QLabel("📊 Segment: No active segment")

        layout.addWidget(QLabel("<b>1. Calibration (≥150 empty frames):</b>"))
        layout.addWidget(self.calibrate_btn)
        layout.addWidget(self.calibration_status)
        layout.addWidget(self.calibration_progress_bar)

        layout.addWidget(QLabel("<b>2. Hampel Filter:</b>"))
        hampel_layout = QHBoxLayout()
        hampel_layout.addWidget(self.hampel_checkbox)
        hampel_layout.addWidget(QLabel("Window:"))
        hampel_layout.addWidget(self.hampel_window_spin)
        layout.addLayout(hampel_layout)

        layout.addWidget(QLabel("<b>3. Ensemble Anomaly Detection:</b>"))
        layout.addWidget(self.ensemble_info_label)
        layout.addWidget(QLabel("Manual Threshold Override:"))
        layout.addWidget(self.anomaly_threshold_slider)
        layout.addWidget(self.anomaly_threshold_label)
        layout.addWidget(self.adaptive_thresh_label)

        layout.addWidget(QLabel("<b>4. Selective CWT (Classifier Features):</b>"))
        layout.addWidget(self.cwt_checkbox)

        layout.addWidget(QLabel("<b>Status:</b>"))
        layout.addWidget(self.processing_status)
        layout.addWidget(self.anomaly_status)
        layout.addWidget(self.segment_info)
        layout.addStretch()
        self.processing_group.setLayout(layout)

    # ─────────────────────────────────────────────────────────────────
    # DATA COLLECTION  (updated labels)
    # ─────────────────────────────────────────────────────────────────
    def setup_data_collection_panel(self):
        self.annotation_group = QGroupBox("📝 Data Collection & Labeling")

        # ── Object types (extended) ────────────────────────────────────
        self.object_type = QComboBox()
        self.object_type.addItems([
            "metal_can", "bottles", "glass", "wood",
            "ordinary_object", "glass_can", "metal",
            "plastic", "plastic_bottles",
            "person", "multiple_people", "pet",
            "vehicle", "no_object", "other",
        ])
        self.object_type.setCurrentText("ordinary_object")

        # ── Activity ──────────────────────────────────────────────────
        self.activity_type = QComboBox()
        self.activity_type.addItems([
            "stationary", "moving", "rotating",
            "walking", "running", "sitting", "standing",
            "falling", "gesturing", "breathing", "none",
        ])
        self.activity_type.setCurrentText("stationary")

        # ── Distance ─────────────────────────────────────────────────
        self.distance_input = QSpinBox()
        self.distance_input.setRange(1, 50)
        self.distance_input.setSuffix(" meters")
        self.distance_input.setValue(1)

        # ── Environment (extended) ────────────────────────────────────
        self.environment_type = QComboBox()
        self.environment_type.addItems([
            "outdoor", "class", "lab", "indoor",
            "empty_area", "office", "hallway",
            "furnished_room", "other",
        ])
        self.environment_type.setCurrentText("lab")

        # ── Session controls ──────────────────────────────────────────
        self.start_session_btn = QPushButton("🎬 Start Recording Session")
        self.start_session_btn.setStyleSheet("background-color: #2196F3; color: white;")

        self.stop_session_btn = QPushButton("⏹️ Stop Recording Session")
        self.stop_session_btn.setStyleSheet("background-color: #F44336; color: white;")
        self.stop_session_btn.setEnabled(False)

        self.mark_event_btn = QPushButton("📍 Mark Current Event")
        self.mark_event_btn.setEnabled(False)

        self.auto_mark_checkbox = QCheckBox("🤖 Auto-mark every 1s")

        self.session_info   = QLabel("Session: Not started")
        self.event_counter  = QLabel("Events: 0")
        self.status_label   = QLabel("Frames: 0 | CSI: ? | TS: -")

        self.events_list = QListWidget()
        self.events_list.setMaximumHeight(150)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Object Type:"))
        layout.addWidget(self.object_type)
        layout.addWidget(QLabel("Activity:"))
        layout.addWidget(self.activity_type)
        layout.addWidget(QLabel("Distance (meters):"))
        layout.addWidget(self.distance_input)
        layout.addWidget(QLabel("Environment:"))
        layout.addWidget(self.environment_type)
        layout.addWidget(self.start_session_btn)
        layout.addWidget(self.stop_session_btn)
        layout.addWidget(self.mark_event_btn)
        layout.addWidget(self.auto_mark_checkbox)
        layout.addWidget(QLabel("Marked Events:"))
        layout.addWidget(self.events_list)
        layout.addWidget(self.session_info)
        layout.addWidget(self.event_counter)
        layout.addWidget(self.status_label)
        self.annotation_group.setLayout(layout)

    # ─────────────────────────────────────────────────────────────────
    # REAL-TIME DETECTION PANEL  (train model removed)
    # ─────────────────────────────────────────────────────────────────
    def setup_real_time_detection_panel(self):
        self.detection_group = QGroupBox("🤖 Real-Time Classification")
        layout = QVBoxLayout()

        self.load_model_btn = QPushButton("📂 Load Classification Model (Default)")
        self.load_model_btn.setStyleSheet("background-color: #9C27B0; color: white;")

        self.load_detection_model_btn = QPushButton("📁 Load Detection Model (Browse)")
        self.load_detection_model_btn.setStyleSheet("background-color: #3F51B5; color: white;")
        self.load_detection_model_btn.setToolTip("Browse and load a pre-trained model")

        self.model_status     = QLabel("Model: Not loaded")
        self.model_path_label = QLabel("Path: None")
        self.model_path_label.setStyleSheet("color: #666; font-size: 10px;")

        self.detection_toggle = QCheckBox("🔍 Enable Real-time Classification")

        self.prediction_label = QLabel("Classification: -")
        self.prediction_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.confidence_label = QLabel("Confidence: -")
        self.confidence_bar   = QProgressBar()
        self.confidence_bar.setRange(0, 100)

        self.features_label = QLabel("Features: Not extracted")

        self.results_list = QListWidget()
        self.results_list.setMaximumHeight(120)

        layout.addWidget(self.load_model_btn)
        layout.addWidget(self.load_detection_model_btn)
        layout.addWidget(self.model_status)
        layout.addWidget(self.model_path_label)
        layout.addWidget(self.detection_toggle)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.confidence_bar)
        layout.addWidget(self.features_label)
        layout.addWidget(QLabel("Recent Classifications:"))
        layout.addWidget(self.results_list)
        self.detection_group.setLayout(layout)

    # ─────────────────────────────────────────────────────────────────
    # 4T AI PREDICTIONS PANEL
    # ─────────────────────────────────────────────────────────────────
    def setup_4t_ai_panel(self):
        self._4t_group = QGroupBox("4T AI Predictions (4 Approaches)")
        layout = QVBoxLayout()

        # ── Load / Enable controls ────────────────────────────────────
        self._4t_load_btn = QPushButton("Load 4T Models")
        self._4t_load_btn.setStyleSheet(
            "background-color: #00897B; color: white; font-weight: bold; padding: 6px;")
        self._4t_load_btn.setToolTip("Load all 4 trained approach models from 4T_opt/models/")
        self._4t_load_btn.clicked.connect(self._4t_load_models)

        self._4t_browse_btn = QPushButton("Browse Model Folder...")
        self._4t_browse_btn.setStyleSheet("background-color: #546E7A; color: white;")
        self._4t_browse_btn.clicked.connect(self._4t_browse_models)

        self._4t_enable_cb = QCheckBox("Enable Real-Time 4T Predictions")
        self._4t_enable_cb.setEnabled(False)
        self._4t_enable_cb.stateChanged.connect(self._4t_toggle_enabled)

        # ── Status labels ─────────────────────────────────────────────
        self._4t_status = QLabel("Status: Not loaded")
        self._4t_status.setStyleSheet("color: #999;")

        self._4t_approach_status = {}
        approach_names = [
            ('env',    'Approach 1 (Environment)'),
            ('obj',    'Approach 2 (Object)'),
            ('dist',   'Approach 3 (Distance)'),
            ('person', 'Approach 4 (Person/Activity)'),
        ]
        status_layout = QGridLayout()
        for row, (key, name) in enumerate(approach_names):
            lbl = QLabel(f"  {name}:")
            val = QLabel("--")
            val.setStyleSheet("color: #999; font-size: 10px;")
            status_layout.addWidget(lbl, row, 0)
            status_layout.addWidget(val, row, 1)
            self._4t_approach_status[key] = val

        # ── Prediction display ────────────────────────────────────────
        pred_frame = QFrame()
        pred_frame.setFrameStyle(QFrame.StyledPanel)
        pred_frame.setStyleSheet(
            "QFrame { background-color: #263238; border-radius: 8px; padding: 8px; }")
        pred_layout = QGridLayout(pred_frame)

        header_style = "color: #80CBC4; font-size: 10px; font-weight: bold;"
        value_style  = "color: #FFFFFF; font-size: 14px; font-weight: bold;"

        # Row 0: Environment
        pred_layout.addWidget(self._make_label("ENVIRONMENT", header_style), 0, 0)
        self._4t_env_label = QLabel("--")
        self._4t_env_label.setStyleSheet(value_style)
        pred_layout.addWidget(self._4t_env_label, 0, 1)

        # Row 1: Object
        pred_layout.addWidget(self._make_label("OBJECT", header_style), 1, 0)
        self._4t_obj_label = QLabel("--")
        self._4t_obj_label.setStyleSheet(value_style)
        pred_layout.addWidget(self._4t_obj_label, 1, 1)

        # Row 2: Distance
        pred_layout.addWidget(self._make_label("DISTANCE", header_style), 2, 0)
        self._4t_dist_label = QLabel("--")
        self._4t_dist_label.setStyleSheet(value_style)
        pred_layout.addWidget(self._4t_dist_label, 2, 1)

        # Row 3: Person
        pred_layout.addWidget(self._make_label("PERSON", header_style), 3, 0)
        self._4t_person_label = QLabel("--")
        self._4t_person_label.setStyleSheet(value_style)
        pred_layout.addWidget(self._4t_person_label, 3, 1)

        # Row 4: Activity
        pred_layout.addWidget(self._make_label("ACTIVITY", header_style), 4, 0)
        self._4t_act_label = QLabel("--")
        self._4t_act_label.setStyleSheet(value_style)
        pred_layout.addWidget(self._4t_act_label, 4, 1)

        # Row 5: Confidence + Latency
        pred_layout.addWidget(self._make_label("CONFIDENCE", header_style), 5, 0)
        self._4t_conf_label = QLabel("--")
        self._4t_conf_label.setStyleSheet("color: #FFD54F; font-size: 11px;")
        pred_layout.addWidget(self._4t_conf_label, 5, 1)

        pred_layout.addWidget(self._make_label("LATENCY", header_style), 6, 0)
        self._4t_latency_label = QLabel("--")
        self._4t_latency_label.setStyleSheet("color: #90A4AE; font-size: 11px;")
        pred_layout.addWidget(self._4t_latency_label, 6, 1)

        # History list
        self._4t_history = QListWidget()
        self._4t_history.setMaximumHeight(100)
        self._4t_history.setStyleSheet("font-size: 10px;")

        # Assemble layout
        layout.addWidget(self._4t_load_btn)
        layout.addWidget(self._4t_browse_btn)
        layout.addWidget(self._4t_status)
        layout.addLayout(status_layout)
        layout.addWidget(self._4t_enable_cb)
        layout.addWidget(QLabel("<b>Live Predictions:</b>"))
        layout.addWidget(pred_frame)
        layout.addWidget(QLabel("History:"))
        layout.addWidget(self._4t_history)
        layout.addStretch()
        self._4t_group.setLayout(layout)

    def _make_label(self, text, style):
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        return lbl

    # ─────────────────────────────────────────────────────────────────
    # 4T MODEL LOADING
    # ─────────────────────────────────────────────────────────────────
    def _4t_load_models(self):
        if not _4T_ENGINE_AVAILABLE:
            QMessageBox.warning(self, "4T AI Not Available",
                "4T approach modules could not be imported.\n"
                "Ensure 4T_opt/ folder is present with all approach files.")
            return
        self._4t_do_load(self._4t_model_root)

    def _4t_browse_models(self):
        if not _4T_ENGINE_AVAILABLE:
            QMessageBox.warning(self, "4T AI Not Available",
                "4T approach modules could not be imported.")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Select 4T Model Root Folder", self._4t_model_root)
        if folder:
            self._4t_model_root = folder
            self._4t_do_load(folder)

    def _4t_do_load(self, model_root):
        """4T real-time inference engine has been removed from the project.

        The GUI panel is kept for layout consistency; this method is now a
        no-op that simply informs the user that real-time 4T inference is
        unavailable. Offline training/evaluation is still performed via
        run_all_4t.py.
        """
        self._4t_loaded = False
        self._4t_engine = None
        self._4t_eliminator = None
        self._4t_status.setText("Status: Real-time engine removed (offline only)")
        self._4t_status.setStyleSheet("color: #BDBDBD;")
        for key in self._4t_approach_status:
            self._4t_approach_status[key].setText("Disabled")
            self._4t_approach_status[key].setStyleSheet(
                "color: #9E9E9E; font-weight: bold; font-size: 10px;")
        self._4t_enable_cb.setEnabled(False)
        self._4t_enable_cb.setChecked(False)
        QMessageBox.information(self, "4T Real-Time Inference Removed",
            "The real-time inference engine (realtime_inference_4t.py) has "
            "been removed from the pipeline.\n\n"
            "Use run_all_4t.py for offline training/evaluation of the "
            "4 approaches.")

    def _4t_toggle_enabled(self, state):
        self._4t_enabled = (state == Qt.Checked) and self._4t_loaded
        if self._4t_enabled:
            print("[4T] Real-time predictions ENABLED")
        else:
            print("[4T] Real-time predictions DISABLED")
            self._4t_env_label.setText("--")
            self._4t_obj_label.setText("--")
            self._4t_dist_label.setText("--")
            self._4t_person_label.setText("--")
            self._4t_act_label.setText("--")
            self._4t_conf_label.setText("--")
            self._4t_latency_label.setText("--")

    # ─────────────────────────────────────────────────────────────────
    # 4T REAL-TIME INFERENCE  (disabled — realtime_inference_4t.py removed)
    # ─────────────────────────────────────────────────────────────────
    def _4t_run_inference(self, amplitude_segment, phase_segment):
        """
        Real-time 4T inference has been removed from the project. This
        method is a permanent no-op; use run_all_4t.py for offline
        training and evaluation of the four approaches.
        """
        return

    # ─────────────────────────────────────────────────────────────────
    # RF PARAMETERS DISPLAY
    # ─────────────────────────────────────────────────────────────────
    def setup_rf_parameters_display(self):
        self.rf_group = QGroupBox("📡 RF Parameters")
        layout = QGridLayout()

        self.rf_labels = {}
        params_to_display = [
            ("mac",               "MAC Address:"),
            ("rssi",              "RSSI (dBm):"),
            ("rate",              "Rate (Mbps):"),
            ("mcs",               "MCS Index:"),
            ("bandwidth",         "Bandwidth:"),
            ("noise_floor",       "Noise Floor:"),
            ("channel",           "Channel:"),
            ("ant",               "Antenna:"),
            ("sig_mode",          "Signal Mode:"),
            ("secondary_channel", "Secondary Channel:"),
        ]

        for row, (param_id, param_label) in enumerate(params_to_display):
            label = QLabel(param_label)
            value = QLabel("-")
            value.setStyleSheet("font-weight: bold;")
            layout.addWidget(label, row, 0)
            layout.addWidget(value, row, 1)
            self.rf_labels[param_id] = value

        self.csi_len_label = QLabel("CSI Length:")
        self.csi_len_value = QLabel("-")
        self.csi_len_value.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.csi_len_label, len(params_to_display), 0)
        layout.addWidget(self.csi_len_value, len(params_to_display), 1)
        self.rf_group.setLayout(layout)

    # ─────────────────────────────────────────────────────────────────
    # STATISTICS PANEL
    # ─────────────────────────────────────────────────────────────────
    def setup_statistics_panel(self):
        self.stats_group = QGroupBox("Real-time Statistics")
        layout = QVBoxLayout()

        self.fps_label             = QLabel("FPS: 0")
        self.processing_time_label = QLabel("Processing: 0 ms")
        self.anomaly_rate_label    = QLabel("Anomaly Rate: 0.0%")
        self.memory_label          = QLabel("Memory: 0 MB")
        self.pipeline_status_label = QLabel("Pipeline: ⚠️ Not calibrated")
        self.pipeline_status_label.setStyleSheet("color: orange;")
        self.adaptive_thr_display  = QLabel("Adaptive Thr: N/A")
        self.adaptive_thr_display.setStyleSheet("color: #00BCD4;")

        self.export_btn = QPushButton("💾 Export Data")
        self.export_btn.setStyleSheet("background-color: #607D8B; color: white;")
        self.export_btn.clicked.connect(self.export_processed_data)

        self.save_pipeline_btn = QPushButton("💾 Save Pipeline State")
        self.save_pipeline_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.save_pipeline_btn.clicked.connect(self.save_pipeline_state)

        self.load_pipeline_btn = QPushButton("📂 Load Pipeline State")
        self.load_pipeline_btn.setStyleSheet("background-color: #FF9800; color: white;")
        self.load_pipeline_btn.clicked.connect(self.load_pipeline_state)

        layout.addWidget(QLabel("<b>Performance:</b>"))
        layout.addWidget(self.fps_label)
        layout.addWidget(self.processing_time_label)
        layout.addWidget(self.anomaly_rate_label)

        layout.addWidget(QLabel("<b>Memory & Status:</b>"))
        layout.addWidget(self.memory_label)
        layout.addWidget(self.pipeline_status_label)
        layout.addWidget(self.adaptive_thr_display)

        layout.addWidget(QLabel("<b>Data Management:</b>"))
        layout.addWidget(self.export_btn)
        layout.addWidget(self.save_pipeline_btn)
        layout.addWidget(self.load_pipeline_btn)
        layout.addStretch()
        self.stats_group.setLayout(layout)

    # ─────────────────────────────────────────────────────────────────
    # LAYOUT
    # ─────────────────────────────────────────────────────────────────
    def setup_layout(self):
        scroll    = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        main_layout = QHBoxLayout(container)

        # Left column – amplitude + anomaly + spectrogram  (38%)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(self.plotWidget_multi_data)
        left_layout.addWidget(self.plotWidget_anomaly)
        left_layout.addWidget(self.spectrogram_group)
        main_layout.addWidget(left_widget, 38)

        # Middle column – IQ plots + RF params + alerts  (32%)
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.addWidget(self.iq_group)
        middle_layout.addWidget(self.rf_group)
        middle_layout.addWidget(self.alert_group)
        main_layout.addWidget(middle_widget, 32)

        # Right column – controls  (30%)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self.processing_group)
        right_layout.addWidget(self.annotation_group)
        right_layout.addWidget(self.detection_group)
        right_layout.addWidget(self.stats_group)
        if hasattr(self, '_4t_group'):
            right_layout.addWidget(self._4t_group)
        main_layout.addWidget(right_widget, 30)

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.addWidget(scroll)

    # ─────────────────────────────────────────────────────────────────
    # SIGNAL CONNECTIONS
    # ─────────────────────────────────────────────────────────────────
    def connect_signals(self):
        self.calibrate_btn.clicked.connect(self.start_calibration)
        self.hampel_checkbox.stateChanged.connect(self.toggle_hampel_filter)
        self.anomaly_threshold_slider.valueChanged.connect(self.update_anomaly_threshold)
        self.cwt_checkbox.stateChanged.connect(self.toggle_cwt_processing)

        self.start_session_btn.clicked.connect(self.start_recording_session)
        self.stop_session_btn.clicked.connect(self.stop_recording_session)
        self.mark_event_btn.clicked.connect(self.mark_event)
        self.auto_mark_checkbox.stateChanged.connect(self.toggle_auto_mark)

        self.load_model_btn.clicked.connect(self.load_classification_model)
        self.load_detection_model_btn.clicked.connect(self.load_detection_model_from_file)
        self.detection_toggle.stateChanged.connect(self.toggle_detection)

    # ─────────────────────────────────────────────────────────────────
    # AUTO-MARK TIMER HELPERS
    # ─────────────────────────────────────────────────────────────────
    def toggle_auto_mark(self, state):
        """Start or stop the 1-second auto-mark timer."""
        if state == Qt.Checked:
            if self.recording_session:
                self.auto_mark_timer.start(1000)
        else:
            self.auto_mark_timer.stop()

    def auto_mark_event(self):
        """Called by the 1-second timer — mark an event automatically."""
        if self.recording_session:
            self.mark_event()

    # ─────────────────────────────────────────────────────────────────
    # SIGNAL PROCESSING CONTROLS
    # ─────────────────────────────────────────────────────────────────
    def start_calibration(self):
        self.calibrating = True
        self.calibration_progress = 0
        self.signal_pipeline.empty_room_features = None
        self.signal_pipeline.calibration_samples = 0
        self.signal_pipeline.is_calibrated       = False
        self.signal_pipeline.clean_score_buffer.clear()

        self.calibration_status.setText("Calibration: In progress...")
        self.calibration_status.setStyleSheet("color: orange;")
        self.calibrate_btn.setEnabled(False)

        QMessageBox.information(
            self, "Calibration Started",
            "Ensure the room is empty (no movement).\n"
            "Calibration collects 150 frames for the ensemble baseline.\n\n"
            "Models: Isolation Forest + LOF + Elliptic Envelope"
        )

    def toggle_hampel_filter(self, state):
        self.signal_pipeline.hampel_window = self.hampel_window_spin.value()

    def toggle_cwt_processing(self, state):
        self.signal_pipeline.cwt_enabled = (state == Qt.Checked)
        flag = "enabled" if self.signal_pipeline.cwt_enabled else "disabled"
        print(f"CWT processing {flag}")

    def update_anomaly_threshold(self, value):
        threshold = value / 100.0
        self.signal_pipeline.anomaly_threshold = threshold
        self.threshold_line.setValue(threshold)
        self.anomaly_threshold_label.setText(f"Threshold: {threshold:.2f}")

    # ─────────────────────────────────────────────────────────────────
    # SIGNAL PROCESSING LOOP
    # ─────────────────────────────────────────────────────────────────
    def run_signal_processing(self):
        if csi_data_complex is None or self.actual_csi_columns is None:
            return

        try:
            amplitude_frame = np.abs(csi_data_complex[-1, :self.actual_csi_columns])
            phase_frame     = np.angle(csi_data_complex[-1, :self.actual_csi_columns])

            cwt_features, cwt_coefficients, is_anomaly, anomaly_score, amplitude_segment = \
                self.signal_pipeline.process_frame(amplitude_frame, phase_frame, self.current_rf_params)

            # Calibration progress
            if self.calibrating and amplitude_segment is not None:
                calibrated = self.signal_pipeline.calibrate_empty_room(
                    amplitude_segment,
                    np.array(self.signal_pipeline.phase_buffer),
                    self.current_rf_params
                )
                progress = int((self.signal_pipeline.calibration_samples /
                                self.signal_pipeline.calibration_target) * 100)
                self.calibration_progress_bar.setValue(progress)

                if calibrated:
                    self.calibrating = False
                    self.calibrate_btn.setEnabled(True)
                    self.calibration_status.setText("Calibration: Complete ✅")
                    self.calibration_status.setStyleSheet("color: green;")
                    QMessageBox.information(
                        self, "Calibration Complete",
                        "Ensemble anomaly detector trained!\n"
                        "(IsolationForest + LOF + EllipticEnvelope)"
                    )

            # Update anomaly plot
            if self.signal_pipeline.anomaly_scores:
                scores = list(self.signal_pipeline.anomaly_scores)
                self.anomaly_curve.setData(scores)

            if is_anomaly:
                self.anomaly_status.setText(f"🔴 Anomaly: DETECTED (Score: {anomaly_score:.4f})")
                self.anomaly_status.setStyleSheet("color: red; font-weight: bold;")
                self.processing_status.setText("🔴 Status: Anomaly processing")
                self.segment_info.setText("📊 Segment: Active")
                self.trigger_alert(anomaly_score)
            else:
                self.anomaly_status.setText(f"🟢 Anomaly: None (Score: {anomaly_score:.4f})")
                self.anomaly_status.setStyleSheet("color: green;")
                self.processing_status.setText("🟡 Status: Monitoring")
                self.segment_info.setText("📊 Segment: None")

            # Classification (use CWT features if available, else basic features)
            features_for_clf = None
            if cwt_features is not None:
                features_for_clf = cwt_features
                self.features_label.setText(f"Features: {len(cwt_features)} CWT extracted")
            elif is_anomaly and len(self.signal_pipeline.basic_features_history) > 0:
                features_for_clf = self.signal_pipeline.basic_features_history[-1]
                self.features_label.setText(f"Features: {len(features_for_clf)} basic extracted")

            if is_anomaly and features_for_clf is not None and self.model is not None:
                self.run_classification(features_for_clf)

            # ── 4T AI inference (runs every processing cycle when enabled) ──
            if self._4t_enabled and amplitude_segment is not None:
                try:
                    phase_seg = np.array(self.signal_pipeline.phase_buffer)
                    self._4t_run_inference(amplitude_segment, phase_seg)
                except Exception as e4t:
                    print(f"[4T] Inference error: {e4t}")

        except Exception as e:
            print(f"Error in signal processing: {e}")
            traceback.print_exc()

    # ─────────────────────────────────────────────────────────────────
    # DATA UPDATE  (plots, IQ, spectrogram)
    # ─────────────────────────────────────────────────────────────────
    def update_data(self):
        global csi_data_complex, agc_gain_data, fft_gain_data

        if csi_data_complex is None or self.actual_csi_columns is None:
            return

        try:
            amp_arr = np.abs(csi_data_complex)

            # Spectrogram
            self.update_spectrogram(amp_arr)

            # Amplitude curves (limit to 20 subcarriers for performance)
            n_curves = min(len(self.curve_list), self.actual_csi_columns, 20)
            for i in range(n_curves):
                if i < amp_arr.shape[1]:
                    self.curve_list[i].setData(amp_arr[:, i])

            # IQ real / imaginary
            self.update_iq_plots()

        except Exception as e:
            print(f"Error in update_data: {e}")

    # ─────────────────────────────────────────────────────────────────
    # INCOMING CSI METADATA
    # ─────────────────────────────────────────────────────────────────
    def handle_new_meta(self, meta):
        try:
            self.fps_counter += 1

            colors        = meta.get('colors')
            frame_idx     = meta['frame_idx']
            local_ts      = meta['local_ts']
            csi_data_len  = meta['csi_data_len']
            params        = meta['params']

            self.current_frame_idx    = frame_idx
            self.current_timestamp    = local_ts
            self.current_rf_params    = params
            n_complex = csi_data_len // 2

            self.update_rf_parameters_display(params)

            if self.actual_csi_columns is None:
                self.init_csi_buffers_and_plots(n_complex)
            elif self.actual_csi_columns != n_complex:
                print(f"[WARN] CSI length changed {self.actual_csi_columns} → {n_complex}")
                self.init_csi_buffers_and_plots(n_complex)
                return

            if colors is not None:
                self.iq_colors = colors

            rssi    = params.get('rssi', 'N/A')
            rate    = params.get('rate', 'N/A')
            channel = params.get('channel', 'N/A')
            self.status_label.setText(
                f"Frames: {frame_idx} | CSI: {n_complex} subcarriers | "
                f"RSSI: {rssi} dBm | Rate: {rate} Mbps | Ch: {channel}"
            )

        except Exception as e:
            print(f"Error in handle_new_meta: {e}")
            traceback.print_exc()

    def update_rf_parameters_display(self, params):
        for param_id, label_widget in self.rf_labels.items():
            value = params.get(param_id, 'N/A')
            if param_id == 'rssi' and value != 'N/A':
                try:
                    rssi_val = int(value)
                    color    = "green" if rssi_val > -50 else ("orange" if rssi_val > -70 else "red")
                    label_widget.setText(f"<span style='color:{color}'>{value}</span>")
                except:
                    label_widget.setText(str(value))
            else:
                label_widget.setText(str(value))

        if csi_data_complex is not None and self.actual_csi_columns is not None:
            self.csi_len_value.setText(f"{self.actual_csi_columns} subcarriers")

    def init_csi_buffers_and_plots(self, n_cols):
        try:
            self.actual_csi_columns = n_cols

            # Clear amplitude curves
            for c in self.curve_list:
                try: self.plotWidget_multi_data.removeItem(c)
                except: pass
            self.curve_list = []

            max_curves = min(n_cols, 20)
            for i in range(max_curves):
                hue   = int(360 * i / max_curves)
                color = pg.mkColor(QColor.fromHsv(hue, 180, 220))
                pen   = pg.mkPen(color=color, width=1)
                c = self.plotWidget_multi_data.plot([], pen=pen, name=f"Sub {i}")
                self.curve_list.append(c)

            self.deta_len = n_cols

            # Initialise IQ curves
            self._init_iq_curves(n_cols)

            # Reset pipeline buffers
            self.signal_pipeline.amplitude_buffer = deque(maxlen=self.signal_pipeline.window_size)
            self.signal_pipeline.phase_buffer     = deque(maxlen=self.signal_pipeline.window_size)

            print(f"✅ Initialised {max_curves} amplitude + IQ curves for {n_cols} subcarriers")

        except Exception as e:
            print(f"Error in init_csi_buffers_and_plots: {e}")
            traceback.print_exc()

    # ─────────────────────────────────────────────────────────────────
    # STATISTICS UPDATE
    # ─────────────────────────────────────────────────────────────────
    def update_statistics(self):
        try:
            current_time = time.time()
            elapsed      = current_time - self.last_fps_time
            if elapsed > 0:
                fps = self.fps_counter / elapsed
                self.fps_label.setText(f"FPS: {fps:.1f}")
                self.fps_counter   = 0
                self.last_fps_time = current_time

            metrics = self.signal_pipeline.get_performance_metrics()
            self.processing_time_label.setText(
                f"Processing: {metrics['avg_processing_time'] * 1000:.1f} ms")
            self.anomaly_rate_label.setText(
                f"Anomaly Rate: {metrics['anomaly_rate'] * 100:.1f}%")
            self.memory_label.setText(
                f"Memory: {metrics['memory_usage_mb']} MB")

            thr = self.signal_pipeline.anomaly_threshold
            self.adaptive_thr_display.setText(f"Adaptive Thr: {thr:.4f}")
            self.threshold_line.setValue(thr)
            self.anomaly_threshold_label.setText(f"Threshold: {thr:.2f}")

            if self.signal_pipeline.is_calibrated:
                self.pipeline_status_label.setText("Pipeline: ✅ Calibrated (Ensemble)")
                self.pipeline_status_label.setStyleSheet("color: green;")
            else:
                self.pipeline_status_label.setText("Pipeline: ⚠️ Not calibrated")
                self.pipeline_status_label.setStyleSheet("color: orange;")

        except Exception as e:
            print(f"Error updating statistics: {e}")

    # ─────────────────────────────────────────────────────────────────
    # PIPELINE STATE  (save / load)
    # ─────────────────────────────────────────────────────────────────
    def save_pipeline_state(self):
        try:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Pipeline State", "", "Joblib Files (*.joblib)")
            if filepath:
                if not filepath.endswith('.joblib'):
                    filepath += '.joblib'
                if self.signal_pipeline.save_pipeline_state(filepath):
                    QMessageBox.information(self, "Success", f"Pipeline saved to {filepath}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save pipeline: {e}")

    def load_pipeline_state(self):
        try:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Load Pipeline State", "", "Joblib Files (*.joblib)")
            if filepath and self.signal_pipeline.load_pipeline_state(filepath):
                QMessageBox.information(self, "Success", f"Pipeline loaded from {filepath}")
                self.calibration_status.setText("Calibration: Loaded from file")
                self.calibration_status.setStyleSheet("color: green;")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load pipeline: {e}")

    # ─────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────
    def export_processed_data(self):
        try:
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = f"exports/export_{timestamp}"
            os.makedirs(export_dir, exist_ok=True)

            if self.signal_pipeline.anomaly_scores:
                scores = list(self.signal_pipeline.anomaly_scores)
                np.save(f"{export_dir}/anomaly_scores.npy", scores)
                np.savetxt(f"{export_dir}/anomaly_scores.csv", scores, delimiter=",")

            if self.signal_pipeline.basic_features_history:
                feats = np.array(list(self.signal_pipeline.basic_features_history))
                np.save(f"{export_dir}/basic_features.npy", feats)

            if self.signal_pipeline.cwt_features_history:
                cwt_feats = list(self.signal_pipeline.cwt_features_history)
                np.save(f"{export_dir}/cwt_features.npy", cwt_feats)

            if self.current_session_data:
                with open(f"{export_dir}/session_data.json", 'w') as f:
                    json.dump(self.current_session_data, f, indent=2)

            metadata = {
                'timestamp':           timestamp,
                'total_frames':        self.current_frame_idx,
                'anomaly_threshold':   self.signal_pipeline.anomaly_threshold,
                'calibration_samples': self.signal_pipeline.calibration_samples,
                'session_id':          self.session_start_time or 'none',
                'events_recorded':     self.event_count,
                'ensemble_weights':    self.signal_pipeline.ensemble_weights.tolist(),
            }
            with open(f"{export_dir}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            QMessageBox.information(
                self, "Export Complete",
                f"✅ Data exported to {export_dir}\n\n"
                f"Contains:\n"
                f"- Anomaly scores (CSV + NPY)\n"
                f"- Basic features (NPY)\n"
                f"- CWT features (NPY, if any)\n"
                f"- Session metadata (JSON)"
            )
            print(f"✅ Data exported to {export_dir}")

        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export: {e}")

    # ─────────────────────────────────────────────────────────────────
    # DATA COLLECTION  (session recording)
    # ─────────────────────────────────────────────────────────────────
    def start_recording_session(self):
        self.recording_session    = True
        self.session_start_time   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.event_count          = 0
        self.current_session_data = []
        self.events_list.clear()

        session_id = f"session_{self.session_start_time}"
        self.session_info.setText(f"Session: {session_id}")
        self.event_counter.setText("Events: 0")

        self.start_session_btn.setEnabled(False)
        self.stop_session_btn.setEnabled(True)
        self.mark_event_btn.setEnabled(True)

        # Start auto-mark timer if checkbox is already checked
        if self.auto_mark_checkbox.isChecked():
            self.auto_mark_timer.start(1000)

        os.makedirs(f"data/{session_id}", exist_ok=True)
        print(f"Started recording session: {session_id}")

    def mark_event(self):
        if not self.recording_session:
            QMessageBox.warning(self, "Warning", "Start a recording session first!")
            return

        csi_shape = str(csi_data_complex.shape) if csi_data_complex is not None else "No data"

        # Prefer CWT features, fall back to basic features
        current_features = None
        if len(self.signal_pipeline.cwt_features_history) > 0:
            current_features = self.signal_pipeline.cwt_features_history[-1].tolist()
            feat_type = "CWT"
        elif len(self.signal_pipeline.basic_features_history) > 0:
            current_features = self.signal_pipeline.basic_features_history[-1].tolist()
            feat_type = "basic"
        else:
            feat_type = "none"

        event_data = {
            "session_id":      self.session_start_time,
            "event_id":        self.event_count,
            "frame_index":     self.current_frame_idx,
            "timestamp":       self.current_timestamp,
            "object_type":     self.object_type.currentText(),
            "activity":        self.activity_type.currentText(),
            "distance":        self.distance_input.value(),
            "environment":     self.environment_type.currentText(),
            "csi_shape":       csi_shape,
            "utc_time":        datetime.now().isoformat(),
            "rf_parameters":   self.current_rf_params.copy(),
            "anomaly_score":   float(self.signal_pipeline.anomaly_scores[-1])
                               if self.signal_pipeline.anomaly_scores else 0.0,
            "features":        current_features,
            "feature_type":    feat_type,
            "in_anomaly_window": self.signal_pipeline.in_anomaly_window,
        }

        self.current_session_data.append(event_data)
        self.event_count += 1
        self.event_counter.setText(f"Events: {self.event_count}")

        event_text = (f"{self.event_count}: {event_data['object_type']} | "
                      f"{event_data['activity']} | {event_data['distance']}m | "
                      f"{event_data['environment']}")
        if current_features:
            event_text += f" [{feat_type}: {len(current_features)} feat]"
        self.events_list.addItem(event_text)

        print(f"Marked event {self.event_count}: {event_data['object_type']}")

    def stop_recording_session(self):
        if not self.recording_session:
            return

        self.recording_session = False
        self.start_session_btn.setEnabled(True)
        self.stop_session_btn.setEnabled(False)
        self.mark_event_btn.setEnabled(False)

        # Stop auto-mark timer and uncheck checkbox
        self.auto_mark_timer.stop()
        self.auto_mark_checkbox.setChecked(False)

        session_id = f"session_{self.session_start_time}"
        out_dir    = f"data/{session_id}"

        # ── metadata.json ─────────────────────────────────────────────
        with open(f"{out_dir}/metadata.json", "w") as f:
            json.dump(self.current_session_data, f, indent=2)

        # ── events.csv ────────────────────────────────────────────────
        with open(f"{out_dir}/events.csv", "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["event_id", "frame_index", "timestamp", "object_type",
                             "activity", "distance", "environment", "utc_time", "feature_type"])
            for ev in self.current_session_data:
                writer.writerow([
                    ev["event_id"], ev["frame_index"], ev["timestamp"],
                    ev["object_type"], ev["activity"], ev["distance"],
                    ev["environment"], ev["utc_time"], ev.get("feature_type", "")
                ])

        # ── NumPy arrays (raw CSI + gain data) ───────────────────────
        if csi_data_complex is not None:
            np.save(f"{out_dir}/csi_data.npy", csi_data_complex)
        if agc_gain_data is not None:
            np.save(f"{out_dir}/agc_data.npy", agc_gain_data)
        if fft_gain_data is not None:
            np.save(f"{out_dir}/fft_data.npy", fft_gain_data)

        # ── session_info.json ─────────────────────────────────────────
        session_info = {
            "session_id":     session_id,
            "total_frames":   self.current_frame_idx,
            "csi_shape":      str(csi_data_complex.shape) if csi_data_complex is not None else None,
            "events":         self.event_count,
            "recording_end":  datetime.now().isoformat(),
        }
        with open(f"{out_dir}/session_info.json", "w") as f:
            json.dump(session_info, f, indent=2)

        # ── rf_summary.json ───────────────────────────────────────────
        if self.current_rf_params:
            with open(f"{out_dir}/rf_summary.json", "w") as f:
                json.dump(self.current_rf_params, f, indent=2)

        self.session_info.setText(f"Session: Saved {session_id}")
        QMessageBox.information(
            self, "Session Saved",
            f"Session '{session_id}' saved to {out_dir}\n\n"
            f"Files written:\n"
            f"  • metadata.json  ({self.event_count} events)\n"
            f"  • events.csv\n"
            f"  • csi_data.npy\n"
            f"  • agc_data.npy\n"
            f"  • fft_data.npy\n"
            f"  • session_info.json\n"
            f"  • rf_summary.json"
        )
        print(f"Session {session_id} saved to {out_dir}")

    # ─────────────────────────────────────────────────────────────────
    # MACHINE LEARNING  (load model only – no training)
    # ─────────────────────────────────────────────────────────────────
    def load_classification_model(self):
        try:
            model_path   = 'cwt_classification_model.pkl'
            scaler_path  = 'cwt_feature_scaler.pkl'
            feat_path    = 'cwt_feature_columns.pkl'
            enc_path     = 'cwt_label_encoder.pkl'

            if not all(os.path.exists(p) for p in [model_path, scaler_path, feat_path, enc_path]):
                QMessageBox.warning(self, "Warning",
                                    "Default model files not found.\n"
                                    "Use 'Load Detection Model' to browse for a model.")
                return

            self.model           = joblib.load(model_path)
            self.scaler          = joblib.load(scaler_path)
            self.feature_columns = joblib.load(feat_path)
            self.label_encoder   = joblib.load(enc_path)
            self.model_path      = "Default location"

            self.model_status.setText("Model: ✅ Loaded (Default)")
            self.model_status.setStyleSheet("color: green; font-weight: bold;")
            self.model_path_label.setText(f"Path: {self.model_path}")
            print("✅ Classification model loaded from default location")

        except Exception as e:
            self.model_status.setText("Model: ❌ Failed")
            self.model_status.setStyleSheet("color: red;")
            QMessageBox.warning(self, "Error", f"Failed to load model: {e}")

    def load_detection_model_from_file(self):
        try:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Select Detection Model", "",
                "Joblib / Pickle Files (*.joblib *.pkl);;All Files (*)"
            )
            if not filepath:
                return

            self.model = joblib.load(filepath)
            directory  = os.path.dirname(filepath)
            filename   = os.path.basename(filepath)

            # Helper to try several naming conventions
            def _try_load(names):
                for name in names:
                    p = os.path.join(directory, name)
                    if os.path.exists(p):
                        try:
                            return joblib.load(p)
                        except:
                            continue
                return None

            stem = filename.replace('.joblib', '').replace('.pkl', '')
            self.scaler = _try_load([
                f"{stem}_scaler.joblib", f"{stem}_scaler.pkl",
                "cwt_feature_scaler.pkl", "scaler.joblib",
            ])
            self.feature_columns = _try_load([
                f"{stem}_features.joblib", f"{stem}_features.pkl",
                "cwt_feature_columns.pkl", "features.joblib",
            ])
            self.label_encoder = _try_load([
                f"{stem}_encoder.joblib", f"{stem}_encoder.pkl",
                "cwt_label_encoder.pkl", "encoder.joblib",
            ])

            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                if hasattr(self.model, 'classes_'):
                    self.label_encoder.classes_ = self.model.classes_
                else:
                    self.label_encoder.classes_ = np.array(['unknown'])

            self.model_path = filepath
            self.model_status.setText("Model: ✅ Loaded (Custom)")
            self.model_status.setStyleSheet("color: green; font-weight: bold;")
            self.model_path_label.setText(f"Path: {os.path.basename(filepath)}")

            n_classes = len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else 0
            QMessageBox.information(
                self, "Model Loaded",
                f"Model: {type(self.model).__name__}\n"
                f"Classes: {n_classes}\n"
                f"Scaler: {'✅' if self.scaler else '❌'}\n"
                f"Feature cols: {'✅' if self.feature_columns is not None else '❌'}"
            )

        except Exception as e:
            self.model_status.setText("Model: ❌ Failed")
            self.model_status.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}\n\n{traceback.format_exc()}")

    def run_classification(self, feature_vector):
        try:
            fv = np.array(feature_vector).reshape(1, -1)

            if self.scaler is not None:
                fv = self.scaler.transform(fv)

            prediction_encoded = self.model.predict(fv)[0]

            if hasattr(self.label_encoder, 'inverse_transform'):
                try:
                    prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
                except:
                    prediction = str(prediction_encoded)
            else:
                prediction = str(prediction_encoded)

            if hasattr(self.model, 'predict_proba'):
                confidence = float(np.max(self.model.predict_proba(fv)[0]))
            else:
                confidence = 1.0

            self.prediction_label.setText(f"Classification: {prediction}")
            self.confidence_label.setText(f"Confidence: {confidence:.1%}")
            self.confidence_bar.setValue(int(confidence * 100))

            color = "green" if confidence > 0.9 else ("orange" if confidence > 0.7 else "red")
            self.prediction_label.setStyleSheet(f"color: {color}; font-weight: bold;")

            ts         = datetime.now().strftime("%H:%M:%S")
            result_text = f"{ts} | {prediction} | {confidence:.1%}"
            self.results_list.addItem(result_text)
            if self.results_list.count() > 10:
                self.results_list.takeItem(0)

        except Exception as e:
            print(f"Classification error: {e}")

    def toggle_detection(self, state):
        if state == Qt.Checked:
            if self.model is None:
                QMessageBox.warning(self, "Warning", "Load a model first!")
                self.detection_toggle.setChecked(False)
                return
            print("✅ Real-time classification enabled")
        else:
            self.prediction_label.setText("Classification: -")
            self.confidence_label.setText("Confidence: -")
            self.confidence_bar.setValue(0)
            print("❌ Real-time classification disabled")


# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    if sys.version_info < (3, 6):
        print("Python version should >= 3.6")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Enhanced CSI Viewer – Ensemble Anomaly Detection"
    )
    parser.add_argument('-p', '--port',  dest='port',       required=True,
                        help="Serial port (e.g. COM4 or /dev/ttyUSB0)")
    parser.add_argument('-s', '--store', dest='store_file', default='./csi_data.csv',
                        help="CSV file to store valid CSI lines")
    parser.add_argument('-l', '--log',   dest='log_file',   default='./csi_data_log.txt',
                        help="File for invalid / other serial output")
    parser.add_argument('--load-pipeline', dest='load_pipeline', action='store_true',
                        help="Load saved pipeline state on startup")

    args = parser.parse_args()

    os.makedirs("data",    exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont("Segoe UI", 9))

    window = EnhancedCSIGraphicalWindow()

    if args.load_pipeline and os.path.exists("pipeline_state.joblib"):
        window.signal_pipeline.load_pipeline_state()
        window.calibration_status.setText("Calibration: Loaded from file")
        window.calibration_status.setStyleSheet("color: green;")

    subthread = SubThread(args.port, args.store_file, args.log_file)
    subthread.data_ready.connect(window.handle_new_meta)
    subthread.start()

    window.show()
    sys.exit(app.exec())
