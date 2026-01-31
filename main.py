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
initial_point = 39
n = 1




def init_app(num_points, initial_point):
    """
    Создает layout для Dash приложения
    """
    # Инициализация Dash приложения
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("CIE XYZ Color Visualization", style={'textAlign': 'center', 'marginBottom': 30}),

        html.Div([
            html.Label("Выберите точку спектра:", style={'fontSize': 18, 'fontWeight': 'bold'}),
            dcc.Slider(
                id='spectrum-slider',
                min=0,
                max=num_points - 1,
                value=initial_point,
                step=1,
                marks={0: '0', num_points-1: str(num_points-1)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'margin': '20px 20px 40px 20px'}),

        html.Div([
            dcc.Graph(id='cmf-plot', style={'width': '33%', 'display': 'inline-block'}),
            dcc.Graph(id='xyz-plot', style={'width': '33%', 'display': 'inline-block'}),
            dcc.Graph(id='xy-plot', style={'width': '33%', 'display': 'inline-block'}),
        ], style={'display': 'flex'}),
    ])
    return app


def main():
    """
    Интерактивное веб-приложение с Dash и plotly для визуализации
    """
    R = np.array([1.0, 0.0, 0.0])
    G = np.array([0.0, 1.0, 0.0])
    B = np.array([0.0, 0.0, 1.0])

    R_2d = np.array([1.0, 0.0])
    G_2d = np.array([0.0, 1.0])
    B_2d = np.array([0.0, 0.0])

    planeRGB = (R, G, B)
    
    BR = R - B
    BG = G - B

    # импортируем данные откликов эксперимента 31 года
    wavelengths = [cie.get_L(i) for i in range(0, len(cie.cieL), n)]
    cmf_data = [np.array(p) for p in cie.get_every_n_points(n)]

    # Находим проекцию каждой точки cmf_data на плоскость RGB
    cmf_projection = [intersect_ray_with_plane(point, planeRGB) for point in cmf_data]

    # для базисных векторов BR, BG находим значения x и y - новые координаты точек
    h_2d_points = []
    for H in cmf_projection:
        BH = H - B
        x, y = solve_in_plane_basis(BR, BG, BH)
        h_2d_points.append((x, y))

    app = init_app(len(cmf_data), initial_point)

    @app.callback(
        [Output('cmf-plot', 'figure'),
         Output('xyz-plot', 'figure'),
         Output('xy-plot', 'figure')],
        Input('spectrum-slider', 'value')
    )
    def update_plots(point_idx):

        BH = cmf_projection[point_idx] - B
        x, y = solve_in_plane_basis(BR, BG, BH)

        B_BH = B + BH
        M = B + (x * BR)
        B_xBR = B + (x * BR)
        M_yBG = M + (y * BG)

        # в 2d пространстве
        BR_2d = R_2d - B_2d
        BG_2d = G_2d - B_2d
        xBR_2d = x * BR_2d
        yBG_2d = y * BG_2d

        fig_cmf = create_cmf_plot(cmf_data, point_idx, wavelengths)

        fig_xyz = create_3d_xyz_plot(
            cmf_data, cmf_projection, R, G, B,
            point_idx, B_BH, B_xBR, M_yBG, M
        )

        fig_xy = create_2d_xy_plot(
            h_2d_points,
            wavelengths[point_idx],
            B_2d, G_2d, R_2d,
            BH_coords=(x, y), 
            xBR=xBR_2d, 
            yBG=yBG_2d
        )

        return fig_cmf, fig_xyz, fig_xy

    app.run(debug=True)


if __name__ == "__main__":
    main()
