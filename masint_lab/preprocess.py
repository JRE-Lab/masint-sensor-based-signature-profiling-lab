diff --git a/masint_lab/preprocess.py b/masint_lab/preprocess.py
new file mode 100644
index 0000000000000000000000000000000000000000..079c60a2f5dc256c6d49bff3e587c33401caee04
--- /dev/null
+++ b/masint_lab/preprocess.py
@@ -0,0 +1,30 @@
+"""Pre-processing utilities for sensor signals."""
+
+from __future__ import annotations
+
+import numpy as np
+
+
+def detrend(signal: np.ndarray) -> np.ndarray:
+    return signal - np.mean(signal)
+
+
+def normalize(signal: np.ndarray) -> np.ndarray:
+    max_val = np.max(np.abs(signal))
+    if max_val == 0:
+        return signal
+    return signal / max_val
+
+
+def smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
+    if window <= 1:
+        return signal
+    kernel = np.ones(window) / window
+    return np.convolve(signal, kernel, mode="same")
+
+
+def preprocess(signal: np.ndarray, smooth_window: int = 5) -> np.ndarray:
+    processed = detrend(signal)
+    processed = smooth(processed, window=smooth_window)
+    processed = normalize(processed)
+    return processed
