import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import numpy as np
import ciexyz31 as cie
import csv
from geometry_utils import intersect_ray_with_plane, solve_in_plane_basis


def read_csv_chromaticity(filename):
    """Читает координаты chromaticity из CSV файла"""
    wavelengths = []
    x_coords = []
    y_coords = []

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                # Формат: wavelength, x, y, z
                wavelength = float(row[0])
                x = float(row[1])
                y = float(row[2])
                wavelengths.append(wavelength)
                x_coords.append(x)
                y_coords.append(y)

    return wavelengths, x_coords, y_coords


def calculate_and_save_differences(h_2d_points, wavelengths, csv_data, output_filename='differences.txt'):
    """
    Вычисляет разность между координатами из двух источников и сохраняет в txt файл.

    Args:
        h_2d_points: список кортежей (x, y) из расчетных данных (черная кривая)
        wavelengths: список длин волн для расчетных данных
        csv_data: кортеж (csv_wavelengths, csv_x, csv_y) из CSV файла (красная кривая)
        output_filename: имя выходного файла
    """
    csv_wavelengths, csv_x, csv_y = csv_data

    # Создаем словари для быстрого поиска по длине волны
    csv_dict = {wl: (x, y) for wl, x, y in zip(csv_wavelengths, csv_x, csv_y)}
    calc_dict = {wl: (x, y) for wl, (x, y) in zip(wavelengths, h_2d_points)}

    # Находим общие длины волн
    common_wavelengths = sorted(set(wavelengths) & set(csv_wavelengths))

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("Разность между расчетными данными (черная кривая) и данными из CSV (красная кривая)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Wavelength (nm)':<20} {'Δx':<20} {'Δy':<20} {'Euclidean distance':<20}\n")
        f.write("-" * 80 + "\n")

        total_dx = 0
        total_dy = 0
        total_euclidean = 0
        count = 0

        for wl in common_wavelengths:
            calc_x, calc_y = calc_dict[wl]
            csv_x_val, csv_y_val = csv_dict[wl]

            dx = calc_x - csv_x_val
            dy = calc_y - csv_y_val
            euclidean_dist = np.sqrt(dx**2 + dy**2)

            f.write(f"{wl:<20.1f} {dx:<20.8f} {dy:<20.8f} {euclidean_dist:<20.8f}\n")

            total_dx += abs(dx)
            total_dy += abs(dy)
            total_euclidean += euclidean_dist
            count += 1

        f.write("-" * 80 + "\n")
        f.write(f"\nСтатистика по {count} точкам:\n")
        f.write(f"Средняя абсолютная разность по X: {total_dx/count:.8f}\n")
        f.write(f"Средняя абсолютная разность по Y: {total_dy/count:.8f}\n")
        f.write(f"Среднее евклидово расстояние: {total_euclidean/count:.8f}\n")

        # Вычисляем максимальные отклонения
        max_dx = max((abs(calc_dict[wl][0] - csv_dict[wl][0]) for wl in common_wavelengths))
        max_dy = max((abs(calc_dict[wl][1] - csv_dict[wl][1]) for wl in common_wavelengths))
        max_euclidean = max((np.sqrt((calc_dict[wl][0] - csv_dict[wl][0])**2 +
                                     (calc_dict[wl][1] - csv_dict[wl][1])**2)
                            for wl in common_wavelengths))

        f.write(f"\nМаксимальная абсолютная разность по X: {max_dx:.8f}\n")
        f.write(f"Максимальная абсолютная разность по Y: {max_dy:.8f}\n")
        f.write(f"Максимальное евклидово расстояние: {max_euclidean:.8f}\n")

    print(f"Разности сохранены в файл: {output_filename}")
    print(f"Обработано {count} общих точек")
    print(f"Среднее евклидово расстояние: {total_euclidean/count:.8f}")


def create_2d_xy_plot(h_2d_list, wavelength, B, G, R,
                      BH_coords=None, xBR=None, yBG=None, csv_data=None):
    """Создаёт 2D XY diagram"""

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

    # Данные из CSV файла (красным цветом)
    if csv_data is not None:
        csv_wavelengths, csv_x, csv_y = csv_data
        fig.add_trace(go.Scatter(x=csv_x, y=csv_y, mode='lines',
                                line=dict(color='red', width=2),
                                name='CIE 1931 из CSV', opacity=0.7))
        fig.add_trace(go.Scatter(x=csv_x, y=csv_y, mode='markers',
                                marker=dict(size=4, color='red', symbol='circle'),
                                showlegend=False, opacity=0.7))

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

    # Чтение данных из CSV файла
    csv_wavelengths, csv_x, csv_y = read_csv_chromaticity('cccie31.csv')
    csv_data = (csv_wavelengths, csv_x, csv_y)

    # Вычисление и сохранение разностей
    calculate_and_save_differences(h_2d_points, wavelengths, csv_data, 'differences.txt')

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
            'csv_data': csv_data,
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
        csv_data_from_store = data['csv_data']

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
            BH_coords=(u, v), xBR=xBR_2d, yBG=yBG_2d, point_idx=point_idx,
            csv_data=csv_data_from_store
        )

        return fig_xy

    app.run(debug=True)


if __name__ == "__main__":
    main()
