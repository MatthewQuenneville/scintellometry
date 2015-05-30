import numpy as np


# the high mag value for 2-bit reconstruction
OPTIMAL_2BIT_HIGH = 3.3359
FOUR_BIT_1_SIGMA = 2.95


def init_luts():
    """Set up the look-up tables for levels as a function of input byte."""
    lut2level = np.array([-1.0, 1.0], dtype=np.float32)
    lut4level = np.array([-OPTIMAL_2BIT_HIGH, -1.0, 1.0, OPTIMAL_2BIT_HIGH],
                         dtype=np.float32)
    lut16level = (np.arange(16) - 8.)/FOUR_BIT_1_SIGMA

    b = np.arange(256)[:, np.newaxis]
    # 1-bit mode
    i = np.arange(8)
    lut1bit = lut2level[(b >> i) & 1]
    # 2-bit mode
    i = np.arange(0, 8, 2)
    lut2bit = lut4level[(b >> i) & 3]
    # 4-bit mode
    i = np.arange(0, 8, 4)
    lut4bit = lut16level[(b >> i) & 0xf]
    return lut1bit, lut2bit, lut4bit

lut1bit, lut2bit, lut4bit = init_luts()


# Decoders keyed by bits_per_sample, complex_data:
DECODERS = {
    (2, False): lambda x: lut2bit[x].ravel(),
    (4, True): lambda x: lut2bit[x].reshape(-1, 2).view(np.complex64).squeeze()
}


def encode_2bit(values):
    if values.dtype.kind == 'c':
        values = values.astype(np.complex64).view(np.float32)

    values = values.reshape(-1, 4)

    bitvalues = np.sign(np.trunc(values / 2.)).astype(np.int32)
    # value < -2:     -1
    # -2 < value < 2:  0
    # value > 2:       1
    bitvalues += np.where(values < 0, 1, 2)
    # value < -2:      0
    # -2 < value < 0:  1
    #  0 < value < 2:  2
    #  value > 2:      3
    return (bitvalues << np.arange(0, 8, 2)).sum(-1).astype(np.uint8)

ENCODERS = {
    (2, False): encode_2bit,
    (4, True): encode_2bit
}
