diff --git a/masint_lab/data_generation.py b/masint_lab/data_generation.py
new file mode 100644
index 0000000000000000000000000000000000000000..97b047784920c913e1a914f55d934421e7f8ed78
--- /dev/null
+++ b/masint_lab/data_generation.py
@@ -0,0 +1,82 @@
+"""Synthetic signal generation for MASINT signatures."""
+
+from __future__ import annotations
+
+from dataclasses import dataclass
+from typing import Dict, List, Tuple
+
+import numpy as np
+
+
+@dataclass(frozen=True)
+class TargetProfile:
+    name: str
+    base_freqs: Tuple[float, ...]
+    harmonics: int
+    noise_level: float
+    modulation_freq: float
+
+
+PROFILES: Dict[str, TargetProfile] = {
+    "vehicle": TargetProfile(
+        name="vehicle",
+        base_freqs=(45.0, 90.0),
+        harmonics=3,
+        noise_level=0.25,
+        modulation_freq=2.5,
+    ),
+    "generator": TargetProfile(
+        name="generator",
+        base_freqs=(60.0, 120.0),
+        harmonics=4,
+        noise_level=0.2,
+        modulation_freq=1.2,
+    ),
+    "footsteps": TargetProfile(
+        name="footsteps",
+        base_freqs=(18.0, 36.0),
+        harmonics=2,
+        noise_level=0.35,
+        modulation_freq=3.5,
+    ),
+}
+
+
+def generate_signal(
+    profile: TargetProfile,
+    sample_rate: int,
+    duration_s: float,
+    rng: np.random.Generator,
+) -> np.ndarray:
+    """Generate a single synthetic sensor trace."""
+    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
+    signal = np.zeros_like(t)
+    for base in profile.base_freqs:
+        for harmonic in range(1, profile.harmonics + 1):
+            phase = rng.uniform(0, 2 * np.pi)
+            signal += (1 / harmonic) * np.sin(2 * np.pi * base * harmonic * t + phase)
+
+    modulation = 1 + 0.2 * np.sin(2 * np.pi * profile.modulation_freq * t)
+    signal *= modulation
+
+    noise = rng.normal(scale=profile.noise_level, size=t.shape)
+    return signal + noise
+
+
+def generate_dataset(
+    samples_per_target: int,
+    sample_rate: int,
+    duration_s: float,
+    seed: int = 42,
+) -> Tuple[np.ndarray, np.ndarray, List[str]]:
+    """Generate labeled samples for each target profile."""
+    rng = np.random.default_rng(seed)
+    signals: List[np.ndarray] = []
+    labels: List[int] = []
+    target_names = list(PROFILES.keys())
+    for idx, name in enumerate(target_names):
+        profile = PROFILES[name]
+        for _ in range(samples_per_target):
+            signals.append(generate_signal(profile, sample_rate, duration_s, rng))
+            labels.append(idx)
+    return np.array(signals), np.array(labels), target_names
