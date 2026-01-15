import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from typing import Tuple, List
import ciexyz31 as cie

dynamic_obj = {'3d':[], 'cmf':[], 'xy_plane':[]}
Vec = Tuple[float, float, float]
EPS = 1e-12

def add(a: Vec, b: Vec) -> Vec:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def sub(a: Vec, b: Vec) -> Vec:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def mul(k: float, a: Vec) -> Vec:
    return (k*a[0], k*a[1], k*a[2])

def dot(a: Vec, b: Vec) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def norm2(a: Vec) -> float:
    return dot(a, a)



def intersect_ray_with_plane(P: Vec, planeRGB) -> Vec:
    BR = sub(planeRGB[0], planeRGB[2])
    BG = sub(planeRGB[1], planeRGB[2])
    n = cross(BR, BG)
    if norm2(n) < EPS:
        raise ValueError("Точки R,G,B коллинеарны: плоскость не определена.")

    n_dot_P = dot(n, P)
    if abs(n_dot_P) < EPS:
        raise ValueError("Луч OP параллелен плоскости: пересечения нет или их бесконечно много.")

    t = dot(n, planeRGB[2]) / n_dot_P
    return mul(t, P)


def solve_in_plane_basis(BR: Vec, BG: Vec, BH: Vec) -> Tuple[float, float]:
    aa = dot(BR, BR)
    bb = dot(BG, BG)
    ab = dot(BR, BG)
    ha = dot(BH, BR)
    hb = dot(BH, BG)

    den = aa * bb - ab * ab
    if abs(den) < EPS:
        raise ValueError("BR и BG линейно зависимы: базис на плоскости не определён.")

    u = (ha * bb - hb * ab) / den
    v = (hb * aa - ha * ab) / den
    return u, v


def draw_points(sub_plt, points, color , m = '.'):
    for i in range(len(points)):
        draw_point(sub_plt, points[i], f"", color, marker=m, s=5)



def draw_point(sub_plt, pt: Vec, name: str, color: str, marker="o", s=30):
    p = sub_plt.scatter([pt[0]], [pt[1]], [pt[2]], s=s, color=color, marker=marker)
    label = sub_plt.text(pt[0], pt[1], pt[2], f" {name}", color=color)
    return p, label


def draw_vector(sub_plt, O: Vec, V: Vec, name: str, color: str, lw=2.0, ls="-", alpha=0.9):
    q = sub_plt.quiver(
        O[0], O[1], O[2],
        V[0], V[1], V[2],
        color=color,
        linewidth=lw,
        linestyle=ls,
        alpha=alpha,
        arrow_length_ratio=0.1
    )
    E = add(O, mul(0.5, V))
    txt = sub_plt.text(E[0], E[1], E[2], f" {name}", color=color)
    return q, txt


def draw_plane(sub_plt, R: Vec, G: Vec, B: Vec, Zero: Vec):
    draw_point(sub_plt, R, "R", "red")
    draw_point(sub_plt, G, "G", "green")
    draw_point(sub_plt, B, "B", "blue")
    draw_point(sub_plt, Zero, "0", "black", s=50)

    poly = Poly3DCollection([[R, G, B]], alpha=0.25, facecolor="gray", edgecolor="none")
    sub_plt.add_collection3d(poly)
    sub_plt.plot(
        [R[0], G[0], B[0], R[0]],
        [R[1], G[1], B[1], R[1]],
        [R[2], G[2], B[2], R[2]],
        color="gray", linewidth=0
    )


def draw_points_curve(sub_plt, points, color="purple", lw=1.5, alpha=0.7, label="Спектр"):
    """Соединяет точки из массива points линией"""
    if len(points) < 2:
        return
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    
    sub_plt.plot(xs, ys, zs, color=color, linewidth=lw, alpha=alpha, label=label)


def draw_3d_coords(sub_plt):
    x0, x1 = sub_plt.get_xlim()
    y0, y1 = sub_plt.get_ylim()
    z0, z1 = sub_plt.get_zlim()
    sub_plt.plot([x0, x1], [0, 0], [0, 0], color="black", linewidth=1.6, alpha=0.4)
    sub_plt.plot([0, 0], [y0, y1], [0, 0], color="black", linewidth=1.6, alpha=0.4)
    sub_plt.plot([0, 0], [0, 0], [z0, z1], color="black", linewidth=1.6, alpha=0.4)

    sub_plt.set_xlabel("X")
    sub_plt.set_ylabel("Y")
    sub_plt.set_zlabel("Z")


def draw_scale_plot(sub_plt, pts_all):
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    zs = [p[2] for p in pts_all]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
    span = max(xmax-xmin, ymax-ymin, zmax-zmin) * 0.6 + 1e-9
    sub_plt.set_xlim(cx - span, cx + span)
    sub_plt.set_ylim(cy - span, cy + span)
    sub_plt.set_zlim(cz - span, cz + span)


def draw_2d_CMF_plot(ax_2d, points, project_point_i, wavelengths):
    """Рисует 2D график зависимости координат X,Y,Z от длины волны"""

    ax_2d.clear()
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    
    ax_2d.plot(wavelengths, xs, 'r-', label='R', linewidth=1.5, alpha=0.8)
    ax_2d.plot(wavelengths, ys, 'g-', label='G', linewidth=1.5, alpha=0.8)
    ax_2d.plot(wavelengths, zs, 'b-', label='B', linewidth=1.5, alpha=0.8)
    
    # Подсветка выбранной точки
    ax_2d.scatter([wavelengths[project_point_i]], [xs[project_point_i]], color='red', s=80, zorder=5, marker='o')
    ax_2d.scatter([wavelengths[project_point_i]], [ys[project_point_i]], color='green', s=80, zorder=5, marker='o')
    ax_2d.scatter([wavelengths[project_point_i]], [zs[project_point_i]], color='blue', s=80, zorder=5, marker='o')
    
    ax_2d.axvline(wavelengths[project_point_i], color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax_2d.set_xlabel('Длина волны (нм)')
    ax_2d.set_ylabel('Отклик')
    ax_2d.set_title('CIE 1931 2-deg observer, CMF (Color matching Functions)')
    ax_2d.legend(loc='best')
    ax_2d.grid(True, alpha=0.3)


def draw_3d_XYZ_plot_static(points: List[Vec], Hs, R: Vec, G: Vec, B: Vec, project_point_i: int = 0, fig=None, sub_plt=None):
    Zero = (0.0, 0.0, 0.0)
    draw_3d_coords(sub_plt)
    draw_plane(sub_plt, R, G, B, Zero)
    draw_points(sub_plt, points, 'pink', '.')
    draw_points(sub_plt, Hs, 'black', 's')
    draw_points_curve(sub_plt, points)
    pts_all = [Zero, R, G, B] + points + Hs
    draw_scale_plot(sub_plt, pts_all)


def draw_projection(sub_plt, Zero, P, H, c = "tab:blue", idx = 0):
    '''
        Рисовать проекцию из Zero --- P --- H
    '''
    plot, = sub_plt.plot(
        [Zero[0], max(H[0], P[0])],
        [Zero[1], max(H[1], P[1])],
        [Zero[2], max(H[2], P[2])],
        color=c, linewidth=1.0, alpha=0.6, linestyle="--"
    )
    p1, label1 = draw_point(sub_plt, P, f"{cie.get_L(idx)}", c, marker="o", s=10)
    p2, label2 = draw_point(sub_plt, H, f"H", c, marker="o", s=15)
    dynamic_obj['3d'].extend([p1, label1, p2, label2, plot])


def draw_BR_BG_decomposition(sub_plt, B, xBR, yBG, BH, M, idx: int = 1):
    q1, txt1 = draw_vector(sub_plt, B, xBR, f"", "red", lw=2, alpha=1)
    q2, txt2 = draw_vector(sub_plt, M, yBG, f"", "green", lw=2, alpha=1)
    q3, txt3 = draw_vector(sub_plt, B, BH, f"", "black", lw=2, ls="-", alpha=1)
    dynamic_obj['3d'].extend([q1, txt1, q2, txt2, q3, txt3])


def draw_3d_XYZ_plot_dynamic(points: List[Vec], Hs, B, xBR, yBG, BH, M, i: int = 0, fig=None, sub_plt=None):
    Zero = (0.0, 0.0, 0.0)
    draw_projection(sub_plt, Zero, points[i], Hs[i], idx=i)
    draw_BR_BG_decomposition(sub_plt, B, xBR, yBG, BH, M, idx=i)

def draw_2d_xy_plot_static(ax_2d, h_2d_list, selected_idx, label_info, R,G,B):
    ax_2d.clear()

    # Оформление осей
    ax_2d.set_xlabel('x (компонента вдоль BR)')
    ax_2d.set_ylabel('y (компонента вдоль BG)')
    ax_2d.set_title('Проекция на плоскость RGB - xy chromaticity diagram')
    ax_2d.legend(loc='best', fontsize=9)
    ax_2d.grid(True, alpha=0.3, linestyle='--')
    ax_2d.set_aspect('equal')
    ax_2d.axhline(0, alpha=0.4)
    ax_2d.axvline(0, alpha=0.4)

    # 1. Рисуем плоскость RGB
    poly = PolyCollection([[R, G, B]], facecolors=["gray"], edgecolors='none', alpha=0.4)
    ax_2d.add_collection(poly)

    ax_2d.scatter([B[0]], [B[1]], color='blue', s=10, label='B (0,0)', zorder=6)
    ax_2d.scatter([R[0]], [R[1]], color='red', s=50, label='R (1,0)', zorder=5)
    ax_2d.scatter([G[0]], [G[1]], color='green', s=50, label='G (0,1)', zorder=5)
    ax_2d.text(R[0], R[1], '  R',  color='red')
    ax_2d.text(G[0], G[1], '  G',  color='green')
    ax_2d.text(B[0], B[1], '  B',  color='blue')

    # 2. Рисуем спектральную кривую
    if len(h_2d_list) > 1:
        xs, ys = zip(*h_2d_list)
        ax_2d.plot(xs, ys, 'black', linewidth=2, alpha=0.6, label='Спектральная кривая', zorder=2)
    
    # 3. Рисуем все точки H
    h_x, h_y = zip(*h_2d_list)
    ax_2d.scatter(h_x, h_y, color='black', s=10, zorder=3, marker='s', alpha=0.6)
    

def draw_2d_xy_plot_dynamic(ax_2d, B, BH, xBR, yBG, H_label = 'H'):
    # Изменяемая часть    
    p = ax_2d.scatter([BH[0]], [BH[1]], color='darkblue', s=10, zorder=5, marker='o')

    txt = ax_2d.text(BH[0], BH[1], H_label, fontsize=10, color='darkblue', fontweight='bold', va='bottom')

    # Векторы 
    q1 = ax_2d.quiver(B[0], B[1], BH[0], BH[1], 
         angles='xy', scale_units='xy', scale=1, 
         width=0.005, color='black', alpha=1)
    q2 = ax_2d.quiver(B[0], B[1], xBR[0], xBR[1],
         angles='xy', scale_units='xy', scale=1, 
         width=0.005, color='red', alpha=1)
    q3 = ax_2d.quiver(xBR[0], xBR[1], yBG[0], yBG[1],
         angles='xy', scale_units='xy', scale=1, 
         width=0.005, color='green', alpha=1)

    dynamic_obj['xy_plane'].extend([p, txt, q1, q2, q3])
    

def main():
    """
    Интерактивная визуализация с ползунком выбора точки
    """
    initial_point = 39
    # Начальные расчеты
    R = (1.0, 0.0, 0.0)
    G = (0.0, 1.0, 0.0)
    B = (0.0, 0.0, 1.0)

    R_2d = (1.0, 0.0)
    G_2d = (0.0, 1.0)
    B_2d = (0.0, 0.0)
    planeRGB = (R, G, B)
    n = 1

    points = cie.get_every_n_points(n)
    wavelengths = [cie.get_L(i) for i in range(0, len(cie.cieL), n)]
    Hs = [intersect_ray_with_plane(P, planeRGB) for P in points] 
    BR = sub(R, B)
    BG = sub(G, B)

    h_2d_points = []
    for H in Hs:
        u, v = solve_in_plane_basis(BR, BG, sub(H, B))
        h_2d_points.append((u, v))

    BR = sub(planeRGB[0], planeRGB[2])
    BG = sub(planeRGB[1], planeRGB[2])
    BH = sub(Hs[initial_point], planeRGB[2])
    x, y = solve_in_plane_basis(BR, BG, BH)
    xBR = mul(u, BR)
    yBG = mul(v, BG)
    M = add(B, xBR)


    # Инициализация графиков
    fig = plt.figure(figsize=(20, 7))
    
    sub_plt_cmf = fig.add_subplot(131)
    sub_plt_XYZ = fig.add_subplot(132, projection="3d")
    sub_plt_XYZ.view_init(elev=30, azim=17)
    sub_plt_xy = fig.add_subplot(133)
    sub_plt_XYZ.set_title("ciexyz")
    plt.subplots_adjust(bottom=0.15, left=0.04, right=0.98, top=0.95, wspace=0.3)
    
    draw_2d_CMF_plot(sub_plt_cmf, points, initial_point, wavelengths)
    draw_3d_XYZ_plot_static(points, Hs, R, G, B, initial_point, fig, sub_plt_XYZ)
    draw_2d_xy_plot_static(sub_plt_xy, h_2d_points, initial_point,  f"{cie.get_L(initial_point)} нм", R_2d, G_2d, B_2d)
    
    fig.canvas.draw_idle()

    # Создаём слайдер для подсветки точки на всех трёх графиках 
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax_slider, 
        'Точка спектра', 
        0, 
        len(points) - 1, 
        valinit=initial_point, 
        valstep=1,
        color='lightblue'
    )
    def update_scene(val):
        # Отрисовка выбранной точки на графиках
        point_idx = int(slider.val)

        # Расчеты
        BH = sub(Hs[point_idx], B)
        u, v = solve_in_plane_basis(BR, BG, BH)
        xBR = mul(u, BR)
        yBG = mul(v, BG)
        M = add(B, xBR)

        # Перерисовка 1
        sub_plt_cmf.cla()
        draw_2d_CMF_plot(sub_plt_cmf, points, point_idx, wavelengths)
        # Перерисовка 2
        for obj in dynamic_obj['3d']:
            obj.remove()
        dynamic_obj['3d'].clear()
        draw_3d_XYZ_plot_dynamic(points, Hs, B, xBR, yBG, BH, M, point_idx, fig, sub_plt_XYZ)
        # Перерисовка 3
        for obj in dynamic_obj['xy_plane']:
            obj.remove()
        dynamic_obj['xy_plane'].clear()
        draw_2d_xy_plot_dynamic(sub_plt_xy, B_2d, BH, xBR, yBG, H_label = f"H ({cie.get_L(point_idx)} nm)\n x={x:.2f}, y={y:.2f}")
        fig.canvas.draw_idle()

    slider.on_changed(update_scene)
    update_scene(initial_point)
    plt.show()


if __name__ == "__main__":
    main()
