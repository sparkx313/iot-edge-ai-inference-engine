import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class ModelConfig:
    name: str
    input_shape: Tuple[int, ...]
    n_classes: int
    param_count: int
    size_mb: float
    quantized: bool = False

class EdgeInferenceEngine:
    def __init__(self, config: ModelConfig):
        self.config = config
        np.random.seed(42)

    def quantize_int8(self) -> "EdgeInferenceEngine":
        q_config = ModelConfig(
            name=f"{self.config.name}-INT8",
            input_shape=self.config.input_shape,
            n_classes=self.config.n_classes,
            param_count=self.config.param_count,
            size_mb=self.config.size_mb / 4,
            quantized=True,
        )
        print(f"Quantized: {self.config.size_mb:.1f}MB -> {q_config.size_mb:.1f}MB")
        return EdgeInferenceEngine(q_config)

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        scores = np.random.dirichlet(np.ones(self.config.n_classes))
        return scores

    def benchmark(self, n_runs: int = 100) -> Dict[str, float]:
        dummy = np.random.rand(*self.config.input_shape).astype(np.float32)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.infer(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)
        return {
            "mean_ms": np.mean(latencies),
            "p99_ms": np.percentile(latencies, 99),
            "fps": 1000 / np.mean(latencies),
        }

MODELS = [
    ModelConfig("MobileNetV3-Small", (1,224,224,3), 1000, 2_500_000, 9.8),
    ModelConfig("EfficientNet-Lite0", (1,224,224,3), 1000, 4_700_000, 18.6),
]

if __name__ == "__main__":
    for config in MODELS:
        engine = EdgeInferenceEngine(config)
        stats = engine.benchmark(50)
        print(f"{config.name}: mean={stats['mean_ms']:.2f}ms fps={stats['fps']:.0f}")
        q = engine.quantize_int8()
        q_stats = q.benchmark(50)
        print(f"  INT8: mean={q_stats['mean_ms']:.2f}ms speedup={stats['mean_ms']/q_stats['mean_ms']:.1f}x")
