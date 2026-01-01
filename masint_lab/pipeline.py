diff --git a/masint_lab/pipeline.py b/masint_lab/pipeline.py
new file mode 100644
index 0000000000000000000000000000000000000000..7516b81c421a4387a64352889a90d20f9dca4725
--- /dev/null
+++ b/masint_lab/pipeline.py
@@ -0,0 +1,116 @@
+"""End-to-end pipeline for the MASINT lab."""
+
+from __future__ import annotations
+
+from dataclasses import dataclass
+from pathlib import Path
+from typing import Dict
+
+import numpy as np
+
+from masint_lab.classifier import CentroidClassifier
+from masint_lab.data_generation import generate_dataset
+from masint_lab.features import extract_features
+from masint_lab.preprocess import preprocess
+
+
+@dataclass
+class LabConfig:
+    samples_per_target: int = 120
+    sample_rate: int = 2000
+    duration_s: float = 1.0
+    train_split: float = 0.75
+    smooth_window: int = 7
+    seed: int = 42
+
+
+@dataclass
+class LabResults:
+    accuracy: float
+    labels: list[str]
+    confusion_matrix: np.ndarray
+
+
+
+def train_test_split(
+    signals: np.ndarray,
+    labels: np.ndarray,
+    train_split: float,
+    rng: np.random.Generator,
+) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
+    indices = np.arange(len(signals))
+    rng.shuffle(indices)
+    cutoff = int(len(indices) * train_split)
+    train_idx, test_idx = indices[:cutoff], indices[cutoff:]
+    return signals[train_idx], signals[test_idx], labels[train_idx], labels[test_idx]
+
+
+def build_features(signals: np.ndarray, sample_rate: int, smooth_window: int) -> np.ndarray:
+    features = []
+    for signal in signals:
+        processed = preprocess(signal, smooth_window=smooth_window)
+        features.append(extract_features(processed, sample_rate))
+    return np.vstack(features)
+
+
+def evaluate(
+    predictions: np.ndarray,
+    labels: np.ndarray,
+    num_labels: int,
+) -> np.ndarray:
+    matrix = np.zeros((num_labels, num_labels), dtype=int)
+    for pred, actual in zip(predictions, labels):
+        matrix[actual, pred] += 1
+    return matrix
+
+
+def run_lab(output_dir: Path, config: LabConfig) -> LabResults:
+    output_dir.mkdir(parents=True, exist_ok=True)
+    signals, labels, target_names = generate_dataset(
+        samples_per_target=config.samples_per_target,
+        sample_rate=config.sample_rate,
+        duration_s=config.duration_s,
+        seed=config.seed,
+    )
+    rng = np.random.default_rng(config.seed)
+    train_signals, test_signals, train_labels, test_labels = train_test_split(
+        signals,
+        labels,
+        train_split=config.train_split,
+        rng=rng,
+    )
+    train_features = build_features(train_signals, config.sample_rate, config.smooth_window)
+    test_features = build_features(test_signals, config.sample_rate, config.smooth_window)
+
+    classifier = CentroidClassifier().fit(train_features, train_labels)
+    accuracy = classifier.score(test_features, test_labels)
+    predictions = classifier.predict(test_features)
+    confusion_matrix = evaluate(predictions, test_labels, len(target_names))
+
+    np.save(output_dir / "train_features.npy", train_features)
+    np.save(output_dir / "test_features.npy", test_features)
+    np.save(output_dir / "test_labels.npy", test_labels)
+    np.save(output_dir / "predictions.npy", predictions)
+
+    return LabResults(
+        accuracy=accuracy,
+        labels=target_names,
+        confusion_matrix=confusion_matrix,
+    )
+
+
+def format_results(results: LabResults) -> str:
+    header = "Confusion Matrix (rows=actual, cols=predicted)"
+    labels = " | ".join(results.labels)
+    lines = [header, labels]
+    for row in results.confusion_matrix:
+        lines.append(" | ".join(str(val) for val in row))
+    return "\n".join(lines)
+
+
+def to_summary_dict(results: LabResults) -> Dict[str, object]:
+    return {
+        "accuracy": results.accuracy,
+        "labels": results.labels,
+        "confusion_matrix": results.confusion_matrix.tolist(),
+    }
