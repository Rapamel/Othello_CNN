from .filters import Filter
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any

class InvalidShape(Exception):
    pass


class ConvLayer:
    def __init__(
        self,
        nb_in_channels: int,
        nb_out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        seed: int,
    ) -> None:
        if padding == -1:
            padding = (kernel_size - 1) // 2
        self.padding = padding
        self.stride = stride
        self.nb_in_channels = nb_in_channels
        self.nb_out_channels = nb_out_channels
        self.kernel_size = kernel_size

        if seed == -1:
            seed = np.random.randint(0, int(1e9))
        rng = np.random.default_rng(seed=seed)
        target_variance = 2 / (kernel_size * kernel_size * nb_in_channels)
        self.filters = [
            Filter.generate_filter(
                kernel_size, nb_in_channels, target_variance, rng
            )
            for i in range(nb_out_channels)
        ]
        self.last_input_vector = None

    def forward(self, input_vector: NDArray) -> NDArray:
        if (
            input_vector.ndim != 3
            or input_vector.shape[2] != self.nb_in_channels
        ):
            raise InvalidShape
        width = input_vector.shape[1]
        height = input_vector.shape[0]
        p = self.padding
        k = self.kernel_size

        width_pad = width + 2 * p
        height_pad = height + 2 * p
        padded_input_vector = padding(input_vector, p)
        width_out = (width_pad - self.kernel_size) // self.stride + 1
        height_out = (height_pad - self.kernel_size) // self.stride + 1

        output_vector = np.zeros((height_out, width_out, self.nb_out_channels))
        for i in range(height_out):
            for j in range(width_out):
                row = i * self.stride
                col = j * self.stride
                kernel = padded_input_vector[row : row + k, col : col + k, :]
                for f, filter in enumerate(self.filters):
                    output_vector[i, j, f] = filter.forward(kernel)

        self.last_input_vector = input_vector
        return output_vector

    def backward(self, later_gradient: NDArray) -> NDArray:
        assert isinstance(self.last_input_vector, np.ndarray)
        padded_last_input_vector = padding(
            self.last_input_vector, self.padding
        )
        height_in, width_in, depth_in = padded_last_input_vector.shape
        height_out, width_out, depth_out = later_gradient.shape
        output_gradient = np.zeros((height_in, width_in, self.nb_in_channels))

        for f, filter in enumerate(self.filters):
            filter.gradient_bias += later_gradient[:, :, f].sum()

            for a in range(self.kernel_size):
                for b in range(self.kernel_size):
                    for c in range(self.nb_in_channels):
                        for i in range(height_out):
                            for j in range(width_out):
                                row = i * self.stride
                                col = j * self.stride
                                filter.gradient_weights[a, b, c] += (
                                    padded_last_input_vector[
                                        row + a, col + b, c
                                    ]
                                    * later_gradient[i, j, f]
                                )
                                output_gradient[row + a, col + b, c] += (
                                    filter.weights[a, b, c]
                                    * later_gradient[i, j, f]
                                )
        return crop_padding(output_gradient, self.padding)

    def reset_gradients(self) -> None:
        for filter in self.filters:
            filter.gradient_bias = 0.0
            filter.gradient_weights = np.zeros(filter.gradient_weights.shape)

    def update(self, learning_rate: float) -> None:
        for filter in self.filters:
            filter.update(learning_rate)
    
    def get_state_dict(self) -> Dict[str, Dict[str, Any]]:
        return_dict = {}
        for i,filter in enumerate(self.filters):
            return_dict[f"filter{i}"] = filter.get_state_dict()
        return return_dict
    
    def load_state_dict(self, state_dict : Dict[str, Dict[str, Any]]) -> None:
        for i,filter in enumerate(self.filters):
            filter.load_state_dict(state_dict[f"filter{i}"])



def padding(input: np.ndarray, p: int) -> np.ndarray:
    height, width, depth = input.shape
    width_pad = width + 2 * p
    height_pad = height + 2 * p
    output = np.zeros((height_pad, width_pad, depth))
    if p > 0:
        output[p:-p, p:-p, :] = input
    else:
        output = input
    return output


def crop_padding(input: np.ndarray, p: int) -> np.ndarray:
    if p > 0:
        output = input[p:-p, p:-p, :]
    else:
        output = input
    return output


class ReluLayer:
    def __init__(
        self,
    ) -> None:
        self.mask = None

    def forward(self, input_vector: NDArray) -> NDArray:
        if input_vector.ndim != 3:
            raise InvalidShape
        output_vector = self.relu(input_vector)
        self.mask = input_vector > 0
        return output_vector

    def backward(self, later_gradient: NDArray) -> NDArray:
        assert isinstance(self.mask, np.ndarray)
        output_gradient = later_gradient * self.mask
        return output_gradient

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        r = np.maximum(x, 0)
        assert isinstance(r, np.ndarray)
        return r
