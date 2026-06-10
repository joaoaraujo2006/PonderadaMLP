import numpy as np


def cross_entropy(Yv, Yp):
    Yp = np.clip(Yp, 1e-8, 1 - 1e-8)
    return -np.mean(np.sum(Yv * np.log(Yp), axis=1))


def derivada_cross_entropy(Yv, Yp):
    return Yp - Yv
