import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import numpy as np
import ciexyz31 as cie


EPS = 1e-12


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
    Интерактивное веб-приложение с Dash для визуализации 2D xy chromaticity diagram
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
        html.H1("CIE XY Chromaticity Diagram", style={'textAlign': 'center', 'marginBottom': 30}),

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
            dcc.Graph(id='xy-plot', style={'width': '100%'}),
        ]),

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
            'R_2d': list(R_2d),
            'G_2d': list(G_2d),
            'B_2d': list(B_2d),
        })
    ])

    @app.callback(
        Output('xy-plot', 'figure'),
        Input('spectrum-slider', 'value'),
        Input('data-store', 'data')
    )
    def update_plot(point_idx, data):
        points_data = [np.array(p) for p in data['points']]
        Hs_data = [np.array(h) for h in data['Hs']]
        wavelengths_data = data['wavelengths']
        h_2d_points_data = data['h_2d_points']
        BR_data = np.array(data['BR'])
        BG_data = np.array(data['BG'])
        B_data = np.array(data['B'])
        R_2d_data = np.array(data['R_2d'])
        G_2d_data = np.array(data['G_2d'])
        B_2d_data = np.array(data['B_2d'])

        # Расчеты для XY plot
        BH = Hs_data[point_idx] - B_data
        u, v = solve_in_plane_basis(BR_data, BG_data, BH)

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

        return fig_xy

    app.run(debug=True)


if __name__ == "__main__":
    main()
