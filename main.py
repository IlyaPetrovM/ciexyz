import dash
from dash import dcc, html, Input, Output
import numpy as np
import ciexyz31 as cie
from geometry_utils import intersect_ray_with_plane, solve_in_plane_basis, EPS
from plot_cmf import create_cmf_plot
from plot_xyz_3d import create_3d_xyz_plot
from plot_xy_2d import create_2d_xy_plot


SHOW_PROJECTION = 1
SHOW_DECOMPOSITION = 1

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
