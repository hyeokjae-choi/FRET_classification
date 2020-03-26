import random
import numpy as np


class TransformerModule:
    def __init__(self, x_length, operate, noise_scale=0.1):
        self.x_length = x_length
        self.operate = operate
        self.noise_sclae = noise_scale

    def run(self, x):
        n = np.array(self.noise_sclae * np.random.rand(self.x_length), dtype=np.float32)
        x_t = x + n
        return x_t
