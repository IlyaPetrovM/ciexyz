from manim import *
import numpy as np
import ciexyz31 as cie


class CMFPlot(Scene):
    def construct(self):
        # Получаем данные из библиотеки CIE
        n = 1  # Шаг выборки
        points = [np.array(p) for p in cie.get_every_n_points(n)]
        wavelengths = [cie.get_L(i) for i in range(0, len(cie.cieL), n)]

        # Извлекаем координаты X, Y, Z
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]

        # Создаём оси графика
        axes = Axes(
            x_range=[360, 830, 50],  # Диапазон длин волн
            y_range=[0, 2, 0.5],     # Диапазон откликов
            x_length=10,             # Ширина графика
            y_length=5,              # Высота графика
            axis_config={"include_numbers": True, "font_size": 24},
            tips=False
        )

        # Подписи осей
        x_label = Text("wavelength, nm", font_size=16).next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = Text("Response", font_size=16).rotate(90 * DEGREES).next_to(axes.y_axis, LEFT, buff=0.3)

        # Заголовок
        title = Text("CIE 1931 2-deg observer, CMF", font_size=36).to_edge(UP, buff=0.5)

        # Создаём кривые для R, G, B компонент
        # Преобразуем данные в точки для Manim
        r_points = [axes.coords_to_point(wavelengths[i], xs[i]) for i in range(len(wavelengths))]
        g_points = [axes.coords_to_point(wavelengths[i], ys[i]) for i in range(len(wavelengths))]
        b_points = [axes.coords_to_point(wavelengths[i], zs[i]) for i in range(len(wavelengths))]

        # Создаём кривые как гладкие линии (толстые, полупрозрачные, без заливки)
        W = 8
        r_curve = VMobject(color=RED, stroke_width=W).set_points_smoothly(r_points).set_opacity(0.5).set_fill(opacity=0)
        g_curve = VMobject(color=GREEN, stroke_width=W).set_points_smoothly(g_points).set_opacity(0.5).set_fill(opacity=0)
        b_curve = VMobject(color=BLUE, stroke_width=W).set_points_smoothly(b_points).set_opacity(0.5).set_fill(opacity=0)

        # Создаём точки (маленькие, непрозрачные)
        r_dots = VGroup(*[Dot(point, radius=0.04, color=RED) for point in r_points])
        g_dots = VGroup(*[Dot(point, radius=0.04, color=GREEN) for point in g_points])
        b_dots = VGroup(*[Dot(point, radius=0.04, color=BLUE) for point in b_points])

        # Легенда
        legend_r = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=RED, stroke_width=4),
            Text("R", font_size=24, color=RED)
        ).arrange(RIGHT, buff=0.2)

        legend_g = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=GREEN, stroke_width=4),
            Text("G", font_size=24, color=GREEN)
        ).arrange(RIGHT, buff=0.2)

        legend_b = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=BLUE, stroke_width=4),
            Text("B", font_size=24, color=BLUE)
        ).arrange(RIGHT, buff=0.2)

        # Группируем легенду
        legend = VGroup(legend_r, legend_g, legend_b).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.to_corner(UR, buff=0.5)

        # Добавляем все объекты на сцену
        self.add(axes, x_label, y_label, title)
        self.add(r_curve, g_curve, b_curve)  # Кривые
        self.add(r_dots, g_dots, b_dots)      # Точки
        self.add(legend)

        # Держим кадр 10 секунд (статичное изображение)
        self.wait(10)
