import numpy as np


def vanilla_sgd_step(params, grads, moments, lr: float, itr: int):
    for name in params:
        params[name] = params[name] - lr * grads[name]


def adam_step(params, grads, moments, lr: float, itr: int):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for name in params:
        moments["m"][name] = beta1 * moments["m"][name] + (1 - beta1) * grads[name]
        moments["v"][name] = beta2 * moments["v"][name] + (1 - beta2) * (grads[name] ** 2)
        m_hat = moments["m"][name] / (1 - beta1 ** itr)
        v_hat = moments["v"][name] / (1 - beta2 ** itr)
        params[name] = params[name] - lr * m_hat / (np.sqrt(v_hat) + eps)
