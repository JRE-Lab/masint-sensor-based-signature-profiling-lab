diff --git a/run_lab.py b/run_lab.py
new file mode 100644
index 0000000000000000000000000000000000000000..dab9a677f561dec66d0fe65e11e1a4c07b5f79be
--- /dev/null
+++ b/run_lab.py
@@ -0,0 +1,94 @@
+"""Run the MASINT Sensor-Based Signature Profiling Lab."""
+
+from __future__ import annotations
+
+import argparse
+import json
+from pathlib import Path
+
+import matplotlib.pyplot as plt
+import numpy as np
+
+from masint_lab.data_generation import generate_dataset
+from masint_lab.pipeline import LabConfig, format_results, run_lab, to_summary_dict
+from masint_lab.preprocess import preprocess
+from masint_lab.visualization import plot_sample_signals, plot_sample_spectra
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(description="Run the MASINT signature profiling lab.")
+    parser.add_argument("--output", default="outputs", help="Directory for generated artifacts.")
+    parser.add_argument("--samples", type=int, default=120, help="Samples per target class.")
+    parser.add_argument("--sample-rate", type=int, default=2000, help="Sample rate in Hz.")
+    parser.add_argument("--duration", type=float, default=1.0, help="Duration per sample in seconds.")
+    parser.add_argument("--train-split", type=float, default=0.75, help="Fraction of data used for training.")
+    parser.add_argument("--smooth-window", type=int, default=7, help="Window size for moving average.")
+    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
+    return parser.parse_args()
+
+
+def plot_confusion_matrix(labels: list[str], matrix: np.ndarray, output_dir: Path) -> None:
+    fig, ax = plt.subplots(figsize=(6, 5))
+    ax.imshow(matrix, cmap="Blues")
+    ax.set_xticks(range(len(labels)))
+    ax.set_yticks(range(len(labels)))
+    ax.set_xticklabels(labels)
+    ax.set_yticklabels(labels)
+    ax.set_xlabel("Predicted")
+    ax.set_ylabel("Actual")
+    for i in range(matrix.shape[0]):
+        for j in range(matrix.shape[1]):
+            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
+    ax.set_title("Confusion Matrix")
+    fig.tight_layout()
+    fig.savefig(output_dir / "confusion_matrix.png")
+    plt.close(fig)
+
+
+def main() -> None:
+    args = parse_args()
+    config = LabConfig(
+        samples_per_target=args.samples,
+        sample_rate=args.sample_rate,
+        duration_s=args.duration,
+        train_split=args.train_split,
+        smooth_window=args.smooth_window,
+        seed=args.seed,
+    )
+    output_dir = Path(args.output)
+    results = run_lab(output_dir, config)
+    summary = to_summary_dict(results)
+
+    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
+
+    print("Accuracy:", f"{results.accuracy:.2%}")
+    print(format_results(results))
+
+    plot_confusion_matrix(results.labels, results.confusion_matrix, output_dir)
+    sample_signals, sample_labels, _ = generate_dataset(
+        samples_per_target=1,
+        sample_rate=config.sample_rate,
+        duration_s=config.duration_s,
+        seed=config.seed,
+    )
+    processed_samples = np.vstack(
+        [preprocess(signal, smooth_window=config.smooth_window) for signal in sample_signals]
+    )
+    plot_sample_signals(
+        processed_samples,
+        sample_labels,
+        results.labels,
+        config.sample_rate,
+        output_dir,
+    )
+    plot_sample_spectra(
+        processed_samples,
+        sample_labels,
+        results.labels,
+        config.sample_rate,
+        output_dir,
+    )
+
+
+if __name__ == "__main__":
+    main()
