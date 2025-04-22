import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true, axis=None):
    return np.mean(np.abs(pred - true), axis=axis)


def MSE(pred, true, axis=None):
    return np.mean((pred - true) ** 2, axis=axis)


def RMSE(pred, true, axis=None):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true, axis=None):
    return np.mean(np.abs((pred - true) / true), axis=axis)


def MSPE(pred, true, axis=None):
    return np.mean(np.square((pred - true) / true), axis=axis)


def metric(pred, true, axis=None):
    mae = MAE(pred, true, axis)
    mse = MSE(pred, true, axis)
    rmse = RMSE(pred, true, axis)
    mape = MAPE(pred, true, axis)
    mspe = MSPE(pred, true, axis)

    return mae, mse, rmse, mape, mspe
