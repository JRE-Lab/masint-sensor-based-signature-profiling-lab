diff --git a/masint_lab/features.py b/masint_lab/features.py
new file mode 100644
index 0000000000000000000000000000000000000000..d7e17f3ea39235d4e3c0ef19c639f9393e9d2b36
--- /dev/null
+++ b/masint_lab/features.py
@@ -0,0 +1,33 @@
+"""Feature extraction for MASINT signals."""
+
+from __future__ import annotations
+
+import numpy as np
+
+
+def rms(signal: np.ndarray) -> float:
+    return float(np.sqrt(np.mean(np.square(signal))))
+
+
+def spectral_features(signal: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
+    spectrum = np.fft.rfft(signal)
+    magnitude = np.abs(spectrum)
+    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
+    if np.sum(magnitude) == 0:
+        return 0.0, 0.0, 0.0
+    centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude))
+    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude)))
+    dominant_freq = float(freqs[np.argmax(magnitude)])
+    return centroid, bandwidth, dominant_freq
+
+
+def extract_features(signal: np.ndarray, sample_rate: int) -> np.ndarray:
+    centroid, bandwidth, dominant = spectral_features(signal, sample_rate)
+    peak = float(np.max(np.abs(signal)))
+    return np.array([
+        rms(signal),
+        peak,
+        centroid,
+        bandwidth,
+        dominant,
+    ])
