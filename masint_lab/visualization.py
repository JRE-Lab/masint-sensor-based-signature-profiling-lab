diff --git a/masint_lab/visualization.py b/masint_lab/visualization.py
new file mode 100644
index 0000000000000000000000000000000000000000..0772f21c0612c5dbf0b125ca7d2b5b37b8fc8ad5
--- /dev/null
+++ b/masint_lab/visualization.py
@@ -0,0 +1,66 @@
+"""Plotting utilities for lab outputs."""
+
+from __future__ import annotations
+
+from pathlib import Path
+
+import matplotlib.pyplot as plt
+import numpy as np
+
+
+def _group_by_label(
+    signals: np.ndarray,
+    labels: np.ndarray,
+    target_names: list[str],
+) -> dict[int, np.ndarray]:
+    grouped: dict[int, np.ndarray] = {}
+    for idx, _ in enumerate(target_names):
+        grouped[idx] = signals[labels == idx]
+    return grouped
+
+
+def plot_sample_signals(
+    signals: np.ndarray,
+    labels: np.ndarray,
+    target_names: list[str],
+    sample_rate: int,
+    output_dir: Path,
+) -> None:
+    grouped = _group_by_label(signals, labels, target_names)
+    fig, axes = plt.subplots(len(target_names), 1, figsize=(8, 6), sharex=True)
+    if len(target_names) == 1:
+        axes = [axes]
+    for idx, name in enumerate(target_names):
+        sample = grouped[idx][0]
+        t = np.arange(len(sample)) / sample_rate
+        axes[idx].plot(t, sample, color="steelblue")
+        axes[idx].set_title(f"Processed signal: {name}")
+        axes[idx].set_ylabel("Amplitude")
+    axes[-1].set_xlabel("Time (s)")
+    fig.tight_layout()
+    fig.savefig(output_dir / "sample_signals.png")
+    plt.close(fig)
+
+
+def plot_sample_spectra(
+    signals: np.ndarray,
+    labels: np.ndarray,
+    target_names: list[str],
+    sample_rate: int,
+    output_dir: Path,
+) -> None:
+    grouped = _group_by_label(signals, labels, target_names)
+    fig, axes = plt.subplots(len(target_names), 1, figsize=(8, 6), sharex=True)
+    if len(target_names) == 1:
+        axes = [axes]
+    for idx, name in enumerate(target_names):
+        sample = grouped[idx][0]
+        spectrum = np.abs(np.fft.rfft(sample))
+        freqs = np.fft.rfftfreq(len(sample), d=1.0 / sample_rate)
+        axes[idx].plot(freqs, spectrum, color="darkorange")
+        axes[idx].set_title(f"Spectrum: {name}")
+        axes[idx].set_ylabel("Magnitude")
+    axes[-1].set_xlabel("Frequency (Hz)")
+    fig.tight_layout()
+    fig.savefig(output_dir / "sample_spectra.png")
+    plt.close(fig)
