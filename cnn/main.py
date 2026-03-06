from .layer import ConvLayer, ReluLayer

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Dict,Any, cast, Self


class ConvolutionNN:
    def __init__(
        self,
        nb_blocks: int,
        nb_channel: int,
        kernel_size: int,
        padding: int = -1,
        stride: int = 1,
        seed: int = -1,
    ) -> None:
        assert nb_blocks > 1 and nb_channel > 0

        if seed == -1:
            seed = np.random.randint(0, int(1e9))
        rng = np.random.default_rng(seed=seed)

        self.nb_layer = nb_blocks
        self.relu_layers = [ReluLayer() for _ in range(nb_blocks)]
        self.kernel_size = kernel_size
        self.nb_channel = nb_channel
        self.stride = stride
        self.padding = padding
        self.conv_layers = [
            ConvLayer(
                nb_channel,
                nb_channel,
                kernel_size,
                padding,
                stride,
                rng.integers(0, int(1e9)),
            )
            for _ in range(nb_blocks)
        ]
        self.conv_layers[0] = ConvLayer(
            2,
            nb_channel,
            kernel_size,
            padding,
            stride,
            rng.integers(0, int(1e9)),
        )
        self.final_layer = ConvLayer(
            nb_channel,
            1,
            kernel_size,
            padding,
            stride,
            rng.integers(0, int(1e9)),
        )

    def forward_pass(self, input_vector: NDArray) -> NDArray:
        vector = input_vector
        for i in range(self.nb_layer):
            vector = self.conv_layers[i].forward(vector)
            vector = self.relu_layers[i].forward(vector)
        vector = self.final_layer.forward(vector)
        return vector

    def backward_pass(self, loss_gradient: NDArray) -> None:
        vector = loss_gradient
        vector = self.final_layer.backward(vector)
        for i in reversed(range(self.nb_layer)):
            vector = self.relu_layers[i].backward(vector)
            vector = self.conv_layers[i].backward(vector)

    def update(self, learning_rate: float = 1) -> None:
        for layer in self.conv_layers:
            layer.update(learning_rate)
            layer.reset_gradients()
        self.final_layer.update(learning_rate)
        self.final_layer.reset_gradients()

    def reset_gradients(self) -> None:
        for layer in self.conv_layers:
            layer.reset_gradients()
        self.final_layer.reset_gradients()

    def get_state_dict(self) -> Dict[str, Any]:
        return_dict : Dict[str, Dict[str,Any]]= {
            "meta": {
                "nb_layer": self.nb_layer,
                "kernel_size": self.kernel_size,
                "nb_channel": self.nb_channel,
                "padding": self.padding, 
                "stride": self.stride,                
            }
        }
        for i,layer in enumerate(self.conv_layers):
            return_dict[f"layer{i}"] = layer.get_state_dict()
        return_dict["final_layer"] = self.final_layer.get_state_dict()
        return flatten_dict(return_dict, "/")

    @classmethod
    def load_state_dict(cls, state : dict) -> Self:
        state_dict = unflatten_dict(state, "/")
        metadata = state_dict["meta"]
        nb_layer = metadata["nb_layer"]
        kernel_size = metadata["kernel_size"]
        nb_channel = metadata["nb_channel"]
        padding = metadata["padding"]
        stride = metadata["stride"]
        cnn = cls(nb_layer, nb_channel, kernel_size, padding, stride)
        for i, layer in enumerate(cnn.conv_layers):
            layer.load_state_dict(state_dict[f"layer{i}"])
        cnn.final_layer.load_state_dict(state_dict["final_layer"])
        return cnn

    
def flatten_dict(d : dict, sep_keys: str = "/")-> dict:
    df = pd.json_normalize(d, sep=sep_keys) 
    return df.to_dict(orient="records")[0]

def unflatten_dict(d : Dict[str, Any], sep_keys: str = "/")-> Dict[str, Any]:
    new_dict = {}
    for key in d.keys():
        value = d[key]
        sub_keys = key.split(sep_keys)
        if len(sub_keys) == 1:
            new_dict[key] = value
        else :
            # Build nested dicts instead of appending to a list.
            new_dict.setdefault(sub_keys[0], {})
            new_dict[sub_keys[0]][sep_keys.join(sub_keys[1:])] = value
    for key in new_dict.keys():
        if isinstance(new_dict[key],dict):
            new_dict[key] = unflatten_dict(new_dict[key])
    return new_dict
