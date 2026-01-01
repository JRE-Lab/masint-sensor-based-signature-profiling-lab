diff --git a/masint_lab/classifier.py b/masint_lab/classifier.py
new file mode 100644
index 0000000000000000000000000000000000000000..8a101d992e20a4e43793d1d7a2e382a50cd81926
--- /dev/null
+++ b/masint_lab/classifier.py
@@ -0,0 +1,30 @@
+"""Simple centroid classifier for signature profiles."""
+
+from __future__ import annotations
+
+from dataclasses import dataclass
+
+import numpy as np
+
+
+@dataclass
+class CentroidClassifier:
+    centroids: np.ndarray | None = None
+
+    def fit(self, features: np.ndarray, labels: np.ndarray) -> "CentroidClassifier":
+        unique_labels = np.unique(labels)
+        centroids = []
+        for label in unique_labels:
+            centroids.append(np.mean(features[labels == label], axis=0))
+        self.centroids = np.stack(centroids)
+        return self
+
+    def predict(self, features: np.ndarray) -> np.ndarray:
+        if self.centroids is None:
+            raise ValueError("Classifier is not fitted.")
+        distances = np.linalg.norm(features[:, None, :] - self.centroids[None, :, :], axis=2)
+        return np.argmin(distances, axis=1)
+
+    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
+        predictions = self.predict(features)
+        return float(np.mean(predictions == labels))
