import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import numpy as np
from typing import List
import ciexyz31 as cie


SHOW_PROJECTION = 1
SHOW_DECOMPOSITION = 1


EPS = 1e-12


def create_arrow_3d(O, V, color: str, lw: float = 2.0) -> dict:
    """Создаёт стрелку в 3D: линия от O до O+V с конусом на конце"""
    E = O + V

    # Направление вектора для конуса
    V_len = np.linalg.norm(V)
    if V_len < EPS:
        return {}

    # Коэффициент для размера конуса
    arrow_ratio = 0.15
    cone_base = arrow_ratio * V

    # Точка основания конуса
    cone_base_point = E - cone_base

    # Создаём конус используя Cone в plotly
    arrow_data = {
        'line': {
            'x': [O[0], cone_base_point[0]],
            'y': [O[1], cone_base_point[1]],
            'z': [O[2], cone_base_point[2]],
            'color': color,
            'width': lw
        },
        'cone': {
            'x': [E[0]],
            'y': [E[1]],
            'z': [E[2]],
            'u': [V[0] * arrow_ratio],
            'v': [V[1] * arrow_ratio],
            'w': [V[2] * arrow_ratio],
            'color': color,
            'sizemode': 'scaled',
            'sizeref': 2
        }
    }
    return arrow_data


def create_arrow_2d(O: tuple, V: tuple, color: str) -> dict:
    """Создаёт стрелку в 2D: линия от O до O+V с конусом на конце"""
    E = (O[0] + V[0], O[1] + V[1])

    arrow_ratio = 0.15
    cone_base = (O[0] + V[0] * (1 - arrow_ratio), O[1] + V[1] * (1 - arrow_ratio))

    arrow_data = {
        'line': {
            'x': [O[0], cone_base[0]],
            'y': [O[1], cone_base[1]],
            'color': color,
            'width': 2
        },
        'marker': {
            'x': [E[0]],
            'y': [E[1]],
            'symbol': 'triangle-up',
            'size': 10,
            'color': color
        }
    }
    return arrow_data


def intersect_ray_with_plane(P, planeRGB):
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




def create_cmf_plot(points, project_point_i, wavelengths):
    """Создаёт 2D график зависимости координат X,Y,Z от длины волны"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    fig = go.Figure()

    # Кривые
    fig.add_trace(go.Scatter(x=wavelengths, y=xs, mode='lines', name='R',
                            line=dict(color='red', width=1.5), opacity=0.8))
    fig.add_trace(go.Scatter(x=wavelengths, y=ys, mode='lines', name='G',
                            line=dict(color='green', width=1.5), opacity=0.8))
    fig.add_trace(go.Scatter(x=wavelengths, y=zs, mode='lines', name='B',
                            line=dict(color='blue', width=1.5), opacity=0.8))

    # Выбранные точки
    fig.add_trace(go.Scatter(x=[wavelengths[project_point_i]], y=[xs[project_point_i]],
                            mode='markers', marker=dict(size=10, color='red'),
                            showlegend=False))
    fig.add_trace(go.Scatter(x=[wavelengths[project_point_i]], y=[ys[project_point_i]],
                            mode='markers', marker=dict(size=10, color='green'),
                            showlegend=False))
    fig.add_trace(go.Scatter(x=[wavelengths[project_point_i]], y=[zs[project_point_i]],
                            mode='markers', marker=dict(size=10, color='blue'),
                            showlegend=False))

    # Вертикальная линия
    fig.add_vline(x=wavelengths[project_point_i], line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title='CIE 1931 2-deg observer, CMF (Color matching Functions)',
        xaxis_title='Длина волны (нм)',
        yaxis_title='Отклик',
        hovermode='closest',
        template='plotly_white'
    )

    return fig


def create_3d_xyz_plot(points, Hs, R, G, B,
                       project_point_i: int = 0, point_idx: int = None,
                       xBR=None, yBG=None, BH=None, M=None):
    """Создаёт 3D график XYZ с динамическими элементами"""
    if point_idx is None:
        point_idx = project_point_i

    Zero = (0.0, 0.0, 0.0)
    fig = go.Figure()

    # Оси координат
    x0, x1 = -0.5, 1.5
    y0, y1 = -0.5, 1.5
    z0, z1 = -0.5, 1.5

    fig.add_trace(go.Scatter3d(x=[x0, x1], y=[0, 0], z=[0, 0], mode='lines',
                              line=dict(color='red', width=6), opacity=0.4, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[y0, y1], z=[0, 0], mode='lines',
                              line=dict(color='green', width=6), opacity=0.4, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[z0, z1], mode='lines',
                              line=dict(color='blue', width=6), opacity=0.4, showlegend=False))

    # Спектральная кривая
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=dict(size=3, color='black'),
                              showlegend=False, opacity=0.5))
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color='purple', width=2),
                              showlegend=False, opacity=0.1))

    # Плоскость RGB
    if SHOW_PROJECTION:
        fig.add_trace(go.Mesh3d(x=[R[0], G[0], B[0]], y=[R[1], G[1], B[1]], z=[R[2], G[2], B[2]],
                               i=[0], j=[1], k=[2], opacity=0.25, color='gray', showlegend=False))

        # Точки Hs
        hs_x = [h[0] for h in Hs]
        hs_y = [h[1] for h in Hs]
        hs_z = [h[2] for h in Hs]
        fig.add_trace(go.Scatter3d(x=hs_x, y=hs_y, z=hs_z, mode='markers',
                                  marker=dict(size=4, color='black', symbol='square'),
                                  showlegend=False))

    # Динамические элементы (проекция)
    P = points[point_idx]
    H = Hs[point_idx]
    fig.add_trace(go.Scatter3d(x=[Zero[0], max(H[0], P[0])], y=[Zero[1], max(H[1], P[1])],
                              z=[Zero[2], max(H[2], P[2])], mode='lines',
                              line=dict(color='lightblue', width=2, dash='dash'), opacity=0.6, showlegend=False))

    fig.add_trace(go.Scatter3d(x=[P[0]], y=[P[1]], z=[P[2]], mode='markers+text',
                              marker=dict(size=5, color='lightblue'),
                              text=[f"{cie.get_L(point_idx)}"], textposition='top center', showlegend=False))

    if SHOW_PROJECTION:
        fig.add_trace(go.Scatter3d(x=[H[0]], y=[H[1]], z=[H[2]], mode='markers',
                                  marker=dict(size=8, color='lightblue'),
                                  showlegend=False))

    # Декомпозиция векторов
    if SHOW_DECOMPOSITION and xBR is not None and yBG is not None and BH is not None and M is not None:
        # Вектор xBR из B
        B_xBR = B + xBR
        fig.add_trace(go.Scatter3d(x=[B[0], B_xBR[0]], y=[B[1], B_xBR[1]],
                                  z=[B[2], B_xBR[2]], mode='lines',
                                  line=dict(color='red', width=3), showlegend=False))

        # Вектор yBG из M
        M = B + xBR
        M_yBG = M + yBG
        fig.add_trace(go.Scatter3d(x=[M[0], M_yBG[0]], y=[M[1], M_yBG[1]],
                                  z=[M[2], M_yBG[2]], mode='lines',
                                  line=dict(color='green', width=3), showlegend=False))

        # Вектор BH из B
        B_BH = B + BH
        fig.add_trace(go.Scatter3d(x=[B[0], B_BH[0]], y=[B[1], B_BH[1]],
                                  z=[B[2], B_BH[2]], mode='lines',
                                  line=dict(color='black', width=3), showlegend=False))

    # Вычисление границ для масштаба
    pts_all = [Zero, R, G, B] + points + Hs
    xs_all = [p[0] for p in pts_all]
    ys_all = [p[1] for p in pts_all]
    zs_all = [p[2] for p in pts_all]
    xmin, xmax = min(xs_all), max(xs_all)
    ymin, ymax = min(ys_all), max(ys_all)
    zmin, zmax = min(zs_all), max(zs_all)

    cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
    span = max(xmax-xmin, ymax-ymin, zmax-zmin) * 0.6 + 1e-9

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[cx - span, cx + span]),
            yaxis=dict(range=[cy - span, cy + span]),
            zaxis=dict(range=[cz - span, cz + span]),
            xaxis_title='R',
            yaxis_title='G',
            zaxis_title='B'
        ),
        title='ciexyz',
        template='plotly_white'
    )
    fig.update_scenes(camera=dict(eye=dict(x=0.8, y=0.8, z=0.6)))

    return fig

def create_2d_xy_plot(h_2d_list, selected_idx, wavelength, B, G, R,
                      BH_coords=None, xBR=None, yBG=None, point_idx=None):
    """Создаёт 2D XY diagram"""
    if point_idx is None:
        point_idx = selected_idx

    fig = go.Figure()

    # Плоскость RGB треугольник
    if len(h_2d_list) > 0:
        fig.add_trace(go.Scatter(x=[R[0], G[0], B[0], R[0]], y=[R[1], G[1], B[1], R[1]],
                                mode='lines', line=dict(color='gray', width=1),
                                fill='toself', fillcolor='rgba(128, 128, 128, 0.2)',
                                showlegend=False))

    # Точки RGB
    fig.add_trace(go.Scatter(x=[B[0]], y=[B[1]], mode='markers+text',
                            marker=dict(size=8, color='blue'),
                            text=['B (0,0)'], textposition='bottom center',
                            name='B (0,0)', showlegend=False))
    fig.add_trace(go.Scatter(x=[R[0]], y=[R[1]], mode='markers+text',
                            marker=dict(size=12, color='red'),
                            text=['R (1,0)'], textposition='bottom center',
                            name='R (1,0)', showlegend=False))
    fig.add_trace(go.Scatter(x=[G[0]], y=[G[1]], mode='markers+text',
                            marker=dict(size=12, color='green'),
                            text=['G (0,1)'], textposition='bottom center',
                            name='G (0,1)', showlegend=False))

    # Спектральная кривая
    if len(h_2d_list) > 1:
        xs, ys = zip(*h_2d_list)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines',
                                line=dict(color='black', width=2),
                                name='Спектральная кривая', opacity=0.6))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers',
                                marker=dict(size=5, color='black', symbol='square'),
                                showlegend=False, opacity=0.6))

    # Динамические элементы (выбранная точка и векторы)
    if BH_coords is not None and xBR is not None and yBG is not None:
        # BH_coords это (u, v) - координаты на плоскости
        fig.add_trace(go.Scatter(x=[BH_coords[0]], y=[BH_coords[1]], mode='markers+text',
                                marker=dict(size=8, color='darkblue'),
                                text=[f'H ({wavelength} nm)'], textposition='top center',
                                showlegend=False))

        # Вектор BH
        fig.add_trace(go.Scatter(x=[B[0], B[0] + BH_coords[0]], y=[B[1], B[1] + BH_coords[1]],
                                mode='lines', line=dict(color='black', width=2),
                                showlegend=False))

        # Вектор xBR
        fig.add_trace(go.Scatter(x=[B[0], B[0] + xBR[0]], y=[B[1], B[1] + xBR[1]],
                                mode='lines', line=dict(color='red', width=2),
                                showlegend=False))

        # Вектор yBG
        xBR_end = B + xBR
        yBG_end = xBR_end + yBG
        fig.add_trace(go.Scatter(x=[xBR_end[0], yBG_end[0]],
                                y=[xBR_end[1], yBG_end[1]],
                                mode='lines', line=dict(color='green', width=2),
                                showlegend=False))

    fig.update_layout(
        title='Проекция на плоскость RGB - xy chromaticity diagram',
        xaxis_title='x (компонента вдоль BR)',
        yaxis_title='y (компонента вдоль BG)',
        hovermode='closest',
        template='plotly_white',
        xaxis=dict(scaleanchor='y', scaleratio=1),
        yaxis=dict(scaleanchor='x', scaleratio=1)
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)

    return fig
    

def main():
    """
    Интерактивное веб-приложение с Dash и plotly для визуализации
    """
    initial_point = 39
    # Начальные расчеты
    R = np.array([1.0, 0.0, 0.0])
    G = np.array([0.0, 1.0, 0.0])
    B = np.array([0.0, 0.0, 1.0])

    R_2d = np.array([1.0, 0.0])
    G_2d = np.array([0.0, 1.0])
    B_2d = np.array([0.0, 0.0])
    planeRGB = (R, G, B)
    n = 1

    points = [np.array(p) for p in cie.get_every_n_points(n)]
    wavelengths = [cie.get_L(i) for i in range(0, len(cie.cieL), n)]
    Hs = [intersect_ray_with_plane(P, planeRGB) for P in points]
    BR = R - B
    BG = G - B

    h_2d_points = []
    for H in Hs:
        u, v = solve_in_plane_basis(BR, BG, H - B)
        h_2d_points.append((u, v))

    # Инициализация Dash приложения
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("CIE XYZ Color Visualization", style={'textAlign': 'center', 'marginBottom': 30}),

        html.Div([
            html.Label("Выберите точку спектра:", style={'fontSize': 18, 'fontWeight': 'bold'}),
            dcc.Slider(
                id='spectrum-slider',
                min=0,
                max=len(points) - 1,
                value=initial_point,
                step=1,
                marks={0: '0', len(points)-1: str(len(points)-1)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'margin': '20px 20px 40px 20px'}),

        html.Div([
            dcc.Graph(id='cmf-plot', style={'width': '33%', 'display': 'inline-block'}),
            dcc.Graph(id='xyz-plot', style={'width': '33%', 'display': 'inline-block'}),
            dcc.Graph(id='xy-plot', style={'width': '33%', 'display': 'inline-block'}),
        ], style={'display': 'flex'}),

        dcc.Store(id='data-store', data={
            'points': [list(p) for p in points],
            'Hs': [list(h) for h in Hs],
            'wavelengths': wavelengths,
            'h_2d_points': h_2d_points,
            'BR': list(BR),
            'BG': list(BG),
            'B': list(B),
            'R': list(R),
            'G': list(G),
            'R_2d': R_2d,
            'G_2d': G_2d,
            'B_2d': B_2d,
        })
    ])

    @app.callback(
        [Output('cmf-plot', 'figure'),
         Output('xyz-plot', 'figure'),
         Output('xy-plot', 'figure')],
        Input('spectrum-slider', 'value'),
        Input('data-store', 'data')
    )
    def update_plots(point_idx, data):
        points_data = [np.array(p) for p in data['points']]
        Hs_data = [np.array(h) for h in data['Hs']]
        wavelengths_data = data['wavelengths']
        h_2d_points_data = data['h_2d_points']
        BR_data = np.array(data['BR'])
        BG_data = np.array(data['BG'])
        B_data = np.array(data['B'])
        R_data = np.array(data['R'])
        G_data = np.array(data['G'])
        R_2d_data = np.array(data['R_2d'])
        G_2d_data = np.array(data['G_2d'])
        B_2d_data = np.array(data['B_2d'])

        # CMF plot
        fig_cmf = create_cmf_plot(points_data, point_idx, wavelengths_data)

        # Расчеты для XYZ и XY plots
        BH = Hs_data[point_idx] - B_data
        u, v = solve_in_plane_basis(BR_data, BG_data, BH)
        xBR = u * BR_data
        yBG = v * BG_data
        M = B_data + xBR

        # XYZ 3D plot
        fig_xyz = create_3d_xyz_plot(
            points_data, Hs_data, R_data, G_data, B_data,
            project_point_i=point_idx, point_idx=point_idx,
            xBR=xBR, yBG=yBG, BH=BH, M=M
        )

        # XY 2D plot
        # Вычисляем 2D координаты для векторов
        BR_2d = R_2d_data - B_2d_data
        BG_2d = G_2d_data - B_2d_data
        xBR_2d = u * BR_2d
        yBG_2d = v * BG_2d

        fig_xy = create_2d_xy_plot(
            h_2d_points_data, point_idx, wavelengths_data[point_idx],
            B_2d_data, G_2d_data, R_2d_data,
            BH_coords=(u, v), xBR=xBR_2d, yBG=yBG_2d, point_idx=point_idx
        )

        return fig_cmf, fig_xyz, fig_xy

    app.run(debug=True)


if __name__ == "__main__":
    main()
