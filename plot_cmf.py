import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import numpy as np
import ciexyz31 as cie


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


def main():
    """
    Интерактивное веб-приложение с Dash для визуализации CMF графика
    """
    initial_point = 39
    n = 1

    points = [np.array(p) for p in cie.get_every_n_points(n)]
    wavelengths = [cie.get_L(i) for i in range(0, len(cie.cieL), n)]

    # Инициализация Dash приложения
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("CIE 1931 Color Matching Functions", style={'textAlign': 'center', 'marginBottom': 30}),

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
            dcc.Graph(id='cmf-plot', style={'width': '100%'}),
        ]),

        dcc.Store(id='data-store', data={
            'points': [list(p) for p in points],
            'wavelengths': wavelengths,
        })
    ])

    @app.callback(
        Output('cmf-plot', 'figure'),
        Input('spectrum-slider', 'value'),
        Input('data-store', 'data')
    )
    def update_plot(point_idx, data):
        points_data = [np.array(p) for p in data['points']]
        wavelengths_data = data['wavelengths']

        # CMF plot
        fig_cmf = create_cmf_plot(points_data, point_idx, wavelengths_data)

        return fig_cmf

    app.run(debug=True)


if __name__ == "__main__":
    main()
