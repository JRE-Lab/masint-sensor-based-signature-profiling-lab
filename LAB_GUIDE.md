diff --git a/LAB_GUIDE.md b/LAB_GUIDE.md
new file mode 100644
index 0000000000000000000000000000000000000000..ff3a7515df561cafb5a416320096a6d91ddd345b
--- /dev/null
+++ b/LAB_GUIDE.md
@@ -0,0 +1,79 @@
+# MASINT Sensor-Based Signature Profiling Lab Guide
+
+This guide provides a full walkthrough for building and evaluating a simple MASINT signature profiling workflow using synthetic data. It is designed for classroom or self-paced use and can be adapted to real sensors.
+
+## Learning objectives
+- Explain how MASINT signatures differ from traditional identity-based intelligence.
+- Collect or generate baseline signatures for multiple target classes.
+- Apply preprocessing to reduce noise and normalize signals.
+- Extract meaningful features for classification.
+- Train and evaluate a classifier using a confusion matrix.
+
+## Prerequisites
+- Python 3.10+ installed.
+- Familiarity with time-series signals and basic FFT concepts.
+
+## Materials
+- Laptop with Python.
+- Optional: a microphone or accelerometer (for real data extension).
+
+## Lab overview
+You will generate synthetic signatures for three targets (vehicle, generator, footsteps), process them, build a feature library, and classify unknown signals. The lab saves reproducible artifacts so you can inspect results after each run.
+
+## Part 1: Configure the lab
+1. Create a virtual environment and install dependencies:
+   ```bash
+   python -m venv .venv
+   source .venv/bin/activate
+   pip install -r requirements.txt
+   ```
+2. Inspect target profiles in `masint_lab/data_generation.py`:
+   - `base_freqs`: dominant frequencies.
+   - `harmonics`: harmonic richness.
+   - `noise_level`: background noise.
+   - `modulation_freq`: low-frequency modulation.
+
+## Part 2: Run the pipeline
+Run the lab with default settings:
+```bash
+python run_lab.py --output outputs
+```
+
+Outputs include:
+- `summary.json` with accuracy and confusion matrix.
+- `confusion_matrix.png` for quick evaluation.
+- `sample_signals.png` and `sample_spectra.png` for target signature inspection.
+- Feature arrays and predictions for deeper analysis.
+
+## Part 3: Inspect results
+1. Open `outputs/confusion_matrix.png` and verify that diagonal counts dominate.
+2. Review `outputs/sample_signals.png` and `outputs/sample_spectra.png` to see how targets differ.
+3. Inspect `summary.json` to capture accuracy for your lab report.
+
+## Part 4: Experimentation checklist
+Try changing one parameter at a time and record effects:
+- Increase `--samples` to improve accuracy.
+- Change `--smooth-window` to reduce noise.
+- Adjust `--train-split` to see how training size influences performance.
+- Modify target profiles to create overlapping signatures.
+
+## Part 5: Report template
+Include the following in your lab report:
+- Objective and hypothesis.
+- Dataset configuration (sample rate, duration, number of samples).
+- Preprocessing steps.
+- Feature list.
+- Classification method.
+- Accuracy and confusion matrix.
+- Observations and limitations.
+
+## Optional extension: Real sensor data
+1. Collect 5–10 recordings per target (e.g., different machines).
+2. Store raw recordings under `data/raw/` as NumPy arrays or WAV files.
+3. Update `masint_lab/data_generation.py` or add a loader to read real data.
+4. Re-run the pipeline and compare with synthetic results.
+
+## Troubleshooting
+- If plots do not save, ensure `outputs/` is writable.
+- If accuracy is low, verify that the number of samples is sufficient and targets are distinct.
+- For reproducibility, keep the same `--seed` across runs.
