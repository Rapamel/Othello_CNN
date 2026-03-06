import numpy as np
from numpy.typing import NDArray
from typing import Dict

class Filter:
    def __init__(self, weights: NDArray, bias: float) -> None:
        self.weights = weights
        self.bias = bias
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = 0.0

    def forward(self, kernel: NDArray) -> float:
        return (kernel * self.weights).sum() + self.bias

    def update(self, learning_rate: float):
        self.weights -= learning_rate * self.gradient_weights
        self.gradient_bias -= learning_rate * self.gradient_bias

    def get_state_dict(self) -> Dict[str, NDArray|float]:
        return {"weights" : self.weights, "bias": self.bias}
    
    def load_state_dict(self, state_dict : Dict[str, NDArray|float]) -> None:
        self.weights = state_dict["weights"]
        self.bias = state_dict["bias"]

    @classmethod
    def generate_filter(
        cls,
        kernel_size: int,
        nb_in_channels: int,
        variance: float,
        generator: np.random.Generator,
    ) -> "Filter":
        weights = generator.normal(
            0, np.sqrt(variance), (kernel_size, kernel_size, nb_in_channels)
        )
        return cls(weights, 0.0)
