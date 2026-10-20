import numpy as np


def conv1d(x: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    """
    1D cross-correlation ("convolution" in the deep-learning sense: the
    kernel is NOT flipped) with "valid" or "same" zero padding.

    x: shape (n,). kernel: shape (k,), k odd. mode: "valid" or "same".

    "valid": no padding, output length n - k + 1.
    "same": zero-pad x by k // 2 on each side first, output length n.

    Slide the kernel across the (possibly padded) signal without flipping
    it, taking sum(window * kernel) at each position.
    """
    # TODO: build the padded (or unpadded) signal first, then one sliding-window loop.
    if mode == "same":
        p = len(kernel)//2
        x_padded = np.pad(x, (p,p), mode = "constant")
    else:
        x_padded = x

    n, k = len(x_padded), len(kernel)
    out = np.empty(n-k+1)
    for i in range(n-k+1):
        window = x_padded[i : i+k]
        out[i] = np.sum(window*kernel)

    return out

    pass
