"""
Общие геометрические функции для работы с проекциями в цветовом пространстве CIE XYZ.
"""
import numpy as np


EPS = 1e-12


def intersect_ray_with_plane(P, planeRGB):
    """
    Вычисляет пересечение луча OP с плоскостью, заданной тремя точками RGB.

    Args:
        P: точка в пространстве (numpy array)
        planeRGB: кортеж из трех точек (R, G, B), определяющих плоскость

    Returns:
        Точка пересечения луча OP с плоскостью RGB

    Raises:
        ValueError: если точки R,G,B коллинеарны или луч параллелен плоскости
    """
    BR = planeRGB[0] - planeRGB[2]
    BG = planeRGB[1] - planeRGB[2]
    n = np.cross(BR, BG)
    if np.dot(n, n) < EPS:
        raise ValueError("Точки R,G,B коллинеарны: плоскость не определена.")

    n_dot_P = np.dot(n, P)
    if abs(n_dot_P) < EPS:
        raise ValueError("Луч OP параллелен плоскости: пересечения нет или их бесконечно много.")

    t = np.dot(n, planeRGB[2]) / n_dot_P
    return t * P


def solve_in_plane_basis(BR, BG, BH):
    """
    Находит координаты вектора BH в базисе векторов BR и BG на плоскости.

    Решает систему: BH = u*BR + v*BG

    Args:
        BR: базисный вектор 1 (numpy array)
        BG: базисный вектор 2 (numpy array)
        BH: вектор, который нужно разложить по базису (numpy array)

    Returns:
        Кортеж (u, v) - координаты BH в базисе (BR, BG)

    Raises:
        ValueError: если BR и BG линейно зависимы
    """
    aa = np.dot(BR, BR)
    bb = np.dot(BG, BG)
    ab = np.dot(BR, BG)
    ha = np.dot(BH, BR)
    hb = np.dot(BH, BG)

    den = aa * bb - ab * ab
    if abs(den) < EPS:
        raise ValueError("BR и BG линейно зависимы: базис на плоскости не определён.")

    u = (ha * bb - hb * ab) / den
    v = (hb * aa - ha * ab) / den
    return u, v
