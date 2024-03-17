import numpy as np


def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    numerator = np.sum((sim - obs) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    nse_val = 1 - numerator / denominator

    return nse_val


def cal_rmse(obs: np.array, sim: np.array):
    # only consider time steps, where observations are available
    # sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    # obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    # sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    # obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    rmse = np.sqrt(np.mean(np.square(sim - obs)))
    return rmse


def cal_mae(obs: np.array, sim: np.array):
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    # sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    # obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    # sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    # obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    mae = np.mean(np.abs(obs-sim))
    return mae


def cal_nse_rmse_mae(test_y, predict_y):
    """计算nse,rmse,mae"""
    # 计算NSE
    nse = calc_nse(test_y, predict_y)
    # 计算RMSE
    rmse = cal_rmse(test_y, predict_y)
    mae = cal_mae(test_y, predict_y)
    return nse, rmse, mae


if __name__ == '__main__':
    y = np.array([1, 2, 2, 3])
    y_hat = np.array([1, 2, 2, 4])
    print(cal_nse_rmse_mae(y, y_hat))