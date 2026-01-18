import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from typing import Tuple, List
import ciexyz31 as cie
from vis_interface import VisInterface, Control

# ============================================
# ВЫБОР БИБЛИОТЕКИ ВИЗУАЛИЗАЦИИ
# ============================================
# Доступные варианты: 'matplotlib' или 'plotly'
# matplotlib - классическая библиотека с Qt5Agg backend, поддерживает интерактивные слайдеры
# plotly - WebGL визуализация, интерактивные 3D графики в браузере
USE_BACKEND = 'plotly'  # Измените на 'plotly' для использования Plotly
# ============================================

SHOW_PROJECTION = 1
SHOW_DECOMPOSITION = 1


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


def draw_points(vis, points, color , m = '.'):
    for i in range(len(points)):
        draw_point(vis, points[i], f"", color, marker=m, s=5)


def draw_point(vis, pt: Vec, name: str, color: str, marker="o", s=30):
    p = vis.scatter(pt[0], pt[1], pt[2], s=s, color=color, marker=marker)
    label = vis.text(pt[0], pt[1], pt[2], f" {name}", color=color)
    return p, label


def draw_vector(vis, O: Vec, V: Vec, name: str, color: str, lw=2.0, ls="-", alpha=0.9):
    q = vis.quiver(O[0], O[1], O[2], V[0], V[1], V[2],
                   color=color, linewidth=lw, linestyle=ls, alpha=alpha, arrow_length_ratio=0.1)
    E = add(O, mul(0.5, V))
    txt = vis.text(E[0], E[1], E[2], f" {name}", color=color)
    return q, txt


def draw_plane(vis, R: Vec, G: Vec, B: Vec, Zero: Vec):
    draw_point(vis, R, "R", "red")
    draw_point(vis, G, "G", "green")
    draw_point(vis, B, "B", "blue")
    draw_point(vis, Zero, "0", "black", s=50)
    poly = Poly3DCollection([[R, G, B]], alpha=0.25, facecolor="gray", edgecolor="none")
    vis.add_collection_3d(poly)
    vis.plot([R[0], G[0], B[0], R[0]], [R[1], G[1], B[1], R[1]], [R[2], G[2], B[2], R[2]],
             color="gray", linewidth=0)


def draw_points_curve(vis, points, color="purple", lw=1.5, alpha=0.7, label="Спектр"):
    if len(points) < 2:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    vis.plot(xs, ys, zs, color=color, linewidth=lw, alpha=alpha, label=label)


def draw_3d_coords(vis):
    xlim, ylim, zlim = vis.get_limits()
    x0, x1 = xlim
    y0, y1 = ylim
    z0, z1 = zlim
    vis.plot([x0, x1], [0, 0], [0, 0], color="red", linewidth=4, alpha=0.4)
    vis.plot([0, 0], [y0, y1], [0, 0], color="green", linewidth=4, alpha=0.4)
    vis.plot([0, 0], [0, 0], [z0, z1], color="blue", linewidth=4, alpha=0.4)
    vis.set_labels("R", "G", "B")


def draw_scale_plot(vis, pts_all):
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    zs = [p[2] for p in pts_all]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
    span = max(xmax-xmin, ymax-ymin, zmax-zmin) * 0.6 + 1e-9
    vis.set_limits((cx - span, cx + span), (cy - span, cy + span), (cz - span, cz + span))


def draw_2d_CMF_plot(vis, points, project_point_i, wavelengths):
    vis.clear()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    vis.plot(wavelengths, xs, 'r-', label='R', linewidth=1.5, alpha=0.8)
    vis.plot(wavelengths, ys, 'g-', label='G', linewidth=1.5, alpha=0.8)
    vis.plot(wavelengths, zs, 'b-', label='B', linewidth=1.5, alpha=0.8)
    vis.scatter(wavelengths[project_point_i], xs[project_point_i], color='red', s=80, zorder=5, marker='o')
    vis.scatter(wavelengths[project_point_i], ys[project_point_i], color='green', s=80, zorder=5, marker='o')
    vis.scatter(wavelengths[project_point_i], zs[project_point_i], color='blue', s=80, zorder=5, marker='o')
    vis.axvline(wavelengths[project_point_i], color='gray', linestyle='--', linewidth=1, alpha=0.5)
    vis.set_labels('Длина волны (нм)', 'Отклик')
    vis.set_title('CIE 1931 2-deg observer, CMF (Color matching Functions)')
    vis.legend(loc='best')
    vis.grid(True, alpha=0.3)


def draw_3d_XYZ_plot_static(points: List[Vec], Hs, R: Vec, G: Vec, B: Vec, vis):
    Zero = (0.0, 0.0, 0.0)
    draw_3d_coords(vis)
    draw_points(vis, points, 'black', '.')
    draw_points_curve(vis, points, lw=10, alpha=0.1)
    if SHOW_PROJECTION:
        draw_plane(vis, R, G, B, Zero)
        draw_points(vis, Hs, 'black', 's')
    pts_all = [Zero, R, G, B] + points + Hs
    draw_scale_plot(vis, pts_all)


def draw_projection(vis, Zero, P, H, c="tab:blue", idx=0):
    plot = vis.plot([Zero[0], max(H[0], P[0])], [Zero[1], max(H[1], P[1])],
                    [Zero[2], max(H[2], P[2])], color=c, linewidth=1.0, alpha=0.6, linestyle="--")[0]
    p1, label1 = draw_point(vis, P, f"{cie.get_L(idx)}", c, marker="o", s=10)
    dynamic_obj['3d'].extend([plot, p1, label1])
    if SHOW_PROJECTION:
        p2, label2 = draw_point(vis, H, "H", c, marker="o", s=15)
        dynamic_obj['3d'].extend([p2, label2])


def draw_BR_BG_decomposition(vis, B, xBR, yBG, BH, M):
    q1, txt1 = draw_vector(vis, B, xBR, "", "red", lw=2, alpha=1)
    q2, txt2 = draw_vector(vis, M, yBG, "", "green", lw=2, alpha=1)
    q3, txt3 = draw_vector(vis, B, BH, "", "black", lw=2, ls="-", alpha=1)
    dynamic_obj['3d'].extend([q1, txt1, q2, txt2, q3, txt3])


def draw_3d_XYZ_plot_dynamic(points: List[Vec], Hs, B, xBR, yBG, BH, M, i: int, vis):
    Zero = (0.0, 0.0, 0.0)
    draw_projection(vis, Zero, points[i], Hs[i], idx=i)
    if SHOW_DECOMPOSITION:
        draw_BR_BG_decomposition(vis, B, xBR, yBG, BH, M)

def draw_2d_xy_plot_static(vis, h_2d_list, R, G, B):
    vis.clear()
    vis.set_labels('x (компонента вдоль BR)', 'y (компонента вдоль BG)')
    vis.set_title('Проекция на плоскость RGB - xy chromaticity diagram')
    vis.legend(loc='best', fontsize=9)
    vis.grid(True, alpha=0.3, linestyle='--')
    vis.set_aspect('equal')
    vis.axhline(0, alpha=0.4)
    vis.axvline(0, alpha=0.4)
    poly = PolyCollection([[R, G, B]], facecolors=["gray"], edgecolors='none', alpha=0.4)
    vis.add_collection(poly)
    vis.scatter(B[0], B[1], color='blue', s=10, label='B (0,0)', zorder=6)
    vis.scatter(R[0], R[1], color='red', s=50, label='R (1,0)', zorder=5)
    vis.scatter(G[0], G[1], color='green', s=50, label='G (0,1)', zorder=5)
    vis.text_2d(R[0], R[1], '  R', color='red')
    vis.text_2d(G[0], G[1], '  G', color='green')
    vis.text_2d(B[0], B[1], '  B', color='blue')
    if len(h_2d_list) > 1:
        xs, ys = zip(*h_2d_list)
        vis.plot(xs, ys, 'black', linewidth=2, alpha=0.6, label='Спектральная кривая', zorder=2)
    h_x, h_y = zip(*h_2d_list)
    vis.scatter_multiple(h_x, h_y, color='black', s=10, zorder=3, marker='s', alpha=0.6)


def draw_2d_xy_plot_dynamic(vis, B, BH, xBR, yBG, H_label='H'):
    p = vis.scatter(BH[0], BH[1], color='darkblue', s=10, zorder=5, marker='o')
    txt = vis.text_2d(BH[0], BH[1], H_label, fontsize=10, color='darkblue', fontweight='bold', va='bottom')
    q1 = vis.quiver_2d(B[0], B[1], BH[0], BH[1], angles='xy', scale_units='xy',
                       scale=1, width=0.005, color='black', alpha=1)
    q2 = vis.quiver_2d(B[0], B[1], xBR[0], xBR[1], angles='xy', scale_units='xy',
                       scale=1, width=0.005, color='red', alpha=1)
    q3 = vis.quiver_2d(xBR[0], xBR[1], yBG[0], yBG[1], angles='xy', scale_units='xy',
                       scale=1, width=0.005, color='green', alpha=1)
    dynamic_obj['xy_plane'].extend([p, txt, q1, q2, q3])
    

def main():
    # Проверка выбранной библиотеки визуализации
    if USE_BACKEND == 'plotly':
        print("Запуск визуализации с использованием Plotly (WebGL)...")
        try:
            import main_plotly
            main_plotly.main()
            return
        except ImportError as e:
            print(f"Ошибка импорта Plotly модуля: {e}")
            print("Убедитесь, что установлены необходимые зависимости: pip install plotly")
            print("Переключение на matplotlib backend...")
    elif USE_BACKEND != 'matplotlib':
        print(f"Неизвестный backend '{USE_BACKEND}', использую matplotlib по умолчанию")

    # Запуск с matplotlib backend
    print("Запуск визуализации с использованием Matplotlib (Qt5Agg)...")

    initial_point = 39
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

    BH = sub(Hs[initial_point], B)
    x, y = solve_in_plane_basis(BR, BG, BH)
    xBR = mul(x, BR)
    yBG = mul(y, BR)
    M = add(B, xBR)

    fig = plt.figure(figsize=(20, 7))
    ax_cmf = fig.add_subplot(131)
    ax_3d = fig.add_subplot(132, projection="3d")
    ax_3d.view_init(elev=30, azim=17)
    ax_3d.set_title("ciexyz")
    ax_xy = fig.add_subplot(133)
    plt.subplots_adjust(bottom=0.15, left=0.04, right=0.98, top=0.95, wspace=0.3)

    vis_cmf = VisInterface(ax_cmf)
    vis_3d = VisInterface(ax_3d)
    vis_xy = VisInterface(ax_xy)

    draw_2d_CMF_plot(vis_cmf, points, initial_point, wavelengths)
    draw_3d_XYZ_plot_static(points, Hs, R, G, B, vis_3d)
    draw_2d_xy_plot_static(vis_xy, h_2d_points, R_2d, G_2d, B_2d)
    fig.canvas.draw_idle()

    slider = Control(fig, [0.2, 0.05, 0.6, 0.03], 'Точка спектра', 0, len(points) - 1, initial_point, 1)

    def update_scene(val):
        point_idx = int(slider.get_val())
        BH = sub(Hs[point_idx], B)
        u, v = solve_in_plane_basis(BR, BG, BH)
        xBR = mul(u, BR)
        yBG = mul(v, BG)
        M = add(B, xBR)

        draw_2d_CMF_plot(vis_cmf, points, point_idx, wavelengths)
        for obj in dynamic_obj['3d']:
            obj.remove()
        dynamic_obj['3d'].clear()
        draw_3d_XYZ_plot_dynamic(points, Hs, B, xBR, yBG, BH, M, point_idx, vis_3d)
        for obj in dynamic_obj['xy_plane']:
            obj.remove()
        dynamic_obj['xy_plane'].clear()
        draw_2d_xy_plot_dynamic(vis_xy, B_2d, BH, xBR, yBG,
                               H_label=f"H ({cie.get_L(point_idx)} nm)\n x={u:.2f}, y={v:.2f}")
        fig.canvas.draw_idle()

    slider.on_changed(update_scene)
    update_scene(initial_point)
    plt.show()


if __name__ == "__main__":
    main()
