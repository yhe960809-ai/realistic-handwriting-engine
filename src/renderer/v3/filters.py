"""可分离卷积。

``np.apply_along_axis(np.convolve, ...)`` 是 Python 层的逐行循环：一张 A4 页
要跑上百万次 ``np.convolve``，占了 v3 渲染约三分之一的时间。这里改成"按核长
做 K 次整数组移位累加"，数学上与 ``np.convolve(..., "valid")`` 等价，但全部
运算留在 numpy 内部，且内存只多一份临时数组（不像 sliding_window_view 会放大
K 倍，整页图放不下）。
"""

from __future__ import annotations

import math

import numpy as np


def derivative(values: np.ndarray) -> np.ndarray:
    """中心差分，等价于 ``np.gradient`` 的一维默认行为。

    轨迹层每笔要算四次导数，一次渲染上万次调用；``np.gradient`` 支持任意轴、
    任意间距与多阶精度，那套通用逻辑的开销比这里的算术本身高一个量级。
    """
    out = np.empty_like(values, dtype=np.float64)
    if values.size < 2:
        out[...] = 0.0
        return out
    out[1:-1] = (values[2:] - values[:-2]) * 0.5
    out[0] = values[1] - values[0]
    out[-1] = values[-1] - values[-2]
    return out


def derivative_rows(points: np.ndarray) -> np.ndarray:
    """对 [N, 2] 点列沿轴 0 做中心差分。"""
    out = np.empty_like(points, dtype=np.float64)
    if points.shape[0] < 2:
        out[...] = 0.0
        return out
    out[1:-1] = (points[2:] - points[:-2]) * 0.5
    out[0] = points[1] - points[0]
    out[-1] = points[-1] - points[-2]
    return out


def quantile(values: np.ndarray, q: float) -> float:
    """``np.quantile`` 的轻量替代。

    轨迹与墨迹层每个字要取好几次分位数，一次渲染上万次调用。``np.quantile``
    单次约 0.6ms 几乎全花在参数校验与派发上，实际排序只占零头；这里直接用
    ``np.partition`` 取第 k 小，语义等价于 ``method="lower"``。
    """
    size = values.size
    if size == 0:
        return 0.0
    if size == 1:
        return float(values.flat[0])
    index = int(q * (size - 1))
    return float(np.partition(values.reshape(-1), index)[index])


def median(values: np.ndarray) -> float:
    return quantile(values, 0.5)


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(sigma * 3.0)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def convolve_1d(field: np.ndarray, kernel: np.ndarray, axis: int, pad_mode: str) -> np.ndarray:
    """沿 ``axis`` 做一维卷积，边界按 ``pad_mode`` 补齐，输出与输入同形。"""
    radius = (len(kernel) - 1) // 2
    if radius == 0:
        return field * float(kernel[0])

    pad_width = [(0, 0)] * field.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(field, pad_width, mode=pad_mode)

    length = field.shape[axis]
    dtype = np.result_type(field.dtype, np.float32)
    out = np.zeros(field.shape, dtype=dtype)
    # 每个抽头复用同一块暂存，而不是让 ``slice * weight`` 现分配一个新数组。
    # 一次模糊有十几个抽头、一页字要模糊上千次，这些临时数组加起来是 GB 级的
    # 分配与回收。累加顺序不变，输出逐位一致。
    scratch = np.empty(field.shape, dtype=dtype)
    slicer: list[slice] = [slice(None)] * field.ndim
    for offset, weight in enumerate(kernel):
        if weight == 0.0:
            continue
        slicer[axis] = slice(offset, offset + length)
        np.multiply(padded[tuple(slicer)], weight, out=scratch)
        out += scratch
    return out


def gaussian_blur_2d(
    field: np.ndarray, sigma: float, *, pad_mode: str = "constant"
) -> np.ndarray:
    """先横后纵的可分离高斯模糊。``sigma`` 过小时原样返回。"""
    if sigma <= 1e-3:
        return field
    kernel = gaussian_kernel_1d(sigma)
    blurred = convolve_1d(field, kernel, axis=1, pad_mode=pad_mode)
    return convolve_1d(blurred, kernel, axis=0, pad_mode=pad_mode)
