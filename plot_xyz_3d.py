import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import numpy as np
import ciexyz31 as cie
from geometry_utils import intersect_ray_with_plane, solve_in_plane_basis


SHOW_PROJECTION = 1
SHOW_DECOMPOSITION = 1


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


def main():
    """
    Интерактивное веб-приложение с Dash для визуализации 3D XYZ пространства
    """
    initial_point = 39
    # Начальные расчеты
    R = np.array([1.0, 0.0, 0.0])
    G = np.array([0.0, 1.0, 0.0])
    B = np.array([0.0, 0.0, 1.0])

    planeRGB = (R, G, B)
    n = 1

    points = [np.array(p) for p in cie.get_every_n_points(n)]
    wavelengths = [cie.get_L(i) for i in range(0, len(cie.cieL), n)]
    Hs = [intersect_ray_with_plane(P, planeRGB) for P in points]
    BR = R - B
    BG = G - B

    # Инициализация Dash приложения
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("CIE XYZ 3D Visualization", style={'textAlign': 'center', 'marginBottom': 30}),

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
            dcc.Graph(id='xyz-plot', style={'width': '100%'}),
        ]),

        dcc.Store(id='data-store', data={
            'points': [list(p) for p in points],
            'Hs': [list(h) for h in Hs],
            'wavelengths': wavelengths,
            'BR': list(BR),
            'BG': list(BG),
            'B': list(B),
            'R': list(R),
            'G': list(G),
        })
    ])

    @app.callback(
        Output('xyz-plot', 'figure'),
        Input('spectrum-slider', 'value'),
        Input('data-store', 'data')
    )
    def update_plot(point_idx, data):
        points_data = [np.array(p) for p in data['points']]
        Hs_data = [np.array(h) for h in data['Hs']]
        BR_data = np.array(data['BR'])
        BG_data = np.array(data['BG'])
        B_data = np.array(data['B'])
        R_data = np.array(data['R'])
        G_data = np.array(data['G'])

        # Расчеты для XYZ plot
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

        return fig_xyz

    app.run(debug=True)


if __name__ == "__main__":
    main()
