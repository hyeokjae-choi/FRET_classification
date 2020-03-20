import numpy as np

np.random.seed(2020)


def augment_trace(trace, noise_scale=0.01):

    if noise_scale is not None:
        length = trace.shape[1]

        r_trace = trace[0]
        b_trace = trace[1]
        e_trace = trace[2]

        noise = noise_scale * 2.0 * (np.random.rand(length) - 0.5)

        r_noise = r_trace + np.random.rand(length)

    return 1.0
