import numpy as np


def cal_rae(obs: np.array, sim: np.array):
    """
    计算相对平均绝对误差（RAE）。

    参数:
    actual_values -- 实际值的列表或NumPy数组。
    predicted_values -- 预测值的列表或NumPy数组。

    返回:
    RAE -- 计算得到的相对平均绝对误差。
    """
    # 计算实际值的平均值
    mean_actual = np.mean(obs)
    
    # 计算绝对误差的总和
    abs_errors = np.abs(obs - sim)
    
    # 计算每个实际值与平均值的差的绝对值
    mean_abs_deviation = np.mean(np.abs(obs - mean_actual))
    
    # 计算RAE
    RAE = abs_errors.mean() / mean_abs_deviation
    
    return RAE

if __name__ == '__main__':
    # 示例使用
    actual = np.array([1, 2, 3, 4, 5])
    predicted = np.array([1.1, 1.9, 3.1, 4.0, 4.9])
    RAE = cal_rae(actual, predicted)
    print(f"The RAE is: {RAE}")