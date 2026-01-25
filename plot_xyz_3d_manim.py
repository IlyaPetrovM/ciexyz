from manim import *
import numpy as np
import ciexyz31 as cie


EPS = 1e-12
SHOW_PROJECTION = True
SHOW_DECOMPOSITION = True


def intersect_ray_with_plane(P, planeRGB):
    """
    Находит пересечение луча OP с плоскостью RGB

    Args:
        P: точка направления луча из начала координат
        planeRGB: кортеж из трех точек (R, G, B), определяющих плоскость

    Returns:
        точка пересечения H луча с плоскостью
    """
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
    """
    Находит коэффициенты u, v разложения вектора BH по базису BR, BG:
    BH = u*BR + v*BG

    Args:
        BR: вектор от B к R
        BG: вектор от B к G
        BH: вектор от B к H

    Returns:
        (u, v) - коэффициенты разложения
    """
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


class XYZ3DPlot(ThreeDScene):
    def construct(self):
        """Создаёт 3D график XYZ с Manim"""

        # Параметры
        initial_point = 39  # Точка спектра (555 nm)
        n = 1  # Шаг выборки точек

        # Начальные расчеты
        R = np.array([1.0, 0.0, 0.0])
        G = np.array([0.0, 1.0, 0.0])
        B = np.array([0.0, 0.0, 1.0])
        Zero = np.array([0.0, 0.0, 0.0])

        planeRGB = (R, G, B)

        # Получаем точки спектральной кривой
        points = [np.array(p) for p in cie.get_every_n_points(n)]
        wavelengths = [cie.get_L(i) for i in range(0, len(cie.cieL), n)]

        # Вычисляем проекции на плоскость RGB
        Hs = [intersect_ray_with_plane(P, planeRGB) for P in points]

        # Векторы базиса плоскости RGB
        BR = R - B
        BG = G - B

        # Выбранная точка для визуализации
        point_idx = initial_point
        P = points[point_idx]
        H = Hs[point_idx]

        # Разложение вектора BH по базису BR, BG
        BH = H - B
        u, v = solve_in_plane_basis(BR, BG, BH)
        xBR = u * BR
        yBG = v * BG
        M = B + xBR

        # Масштаб для Manim (увеличиваем для лучшей видимости)
        scale = 3.5

        # Настройка камеры
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES, zoom=0.8)

        # === СОЗДАНИЕ ОБЪЕКТОВ ===

        # 1. Оси координат (красная=X/R, зелёная=Y/G, синяя=Z/B)
        axis_length = 1.5 * scale
        x_axis = Arrow3D(start=Zero * scale, end=np.array([axis_length, 0, 0]),
                         color=RED, thickness=0.015, height=0.2, base_radius=0.08)
        y_axis = Arrow3D(start=Zero * scale, end=np.array([0, axis_length, 0]),
                         color=GREEN, thickness=0.015, height=0.2, base_radius=0.08)
        z_axis = Arrow3D(start=Zero * scale, end=np.array([0, 0, axis_length]),
                         color=BLUE, thickness=0.015, height=0.2, base_radius=0.08)

        x_axis.set_opacity(0.4)
        y_axis.set_opacity(0.4)
        z_axis.set_opacity(0.4)

        # Подписи осей
        x_label = Text("R", font_size=36, color=RED).move_to(np.array([axis_length + 0.3, 0, 0]))
        y_label = Text("G", font_size=36, color=GREEN).move_to(np.array([0, axis_length + 0.3, 0]))
        z_label = Text("B", font_size=36, color=BLUE).move_to(np.array([0, 0, axis_length + 0.3]))

        # Всегда направлены к камере
        x_label.add_updater(lambda m: m.rotate(self.camera.get_phi(), axis=RIGHT))
        y_label.add_updater(lambda m: m.rotate(self.camera.get_phi(), axis=RIGHT))
        z_label.add_updater(lambda m: m.rotate(self.camera.get_phi(), axis=RIGHT))

        # 2. Спектральная кривая (точки и линия)
        spectrum_dots = VGroup()
        for p in points:
            dot = Sphere(radius=0.06, color=BLACK).move_to(p * scale)
            dot.set_opacity(0.5)
            spectrum_dots.add(dot)

        # Линия через все точки спектра
        spectrum_points = [p * scale for p in points]
        spectrum_line = VMobject(color=PURPLE, stroke_width=2)
        spectrum_line.set_points_as_corners(spectrum_points)
        spectrum_line.set_opacity(0.3)

        # 3. Плоскость RGB (треугольник)
        if SHOW_PROJECTION:
            rgb_plane = Polygon(
                R * scale, G * scale, B * scale,
                color=GRAY, fill_opacity=0.25, stroke_opacity=0.5
            )

            # Точки проекций Hs на плоскость
            hs_dots = VGroup()
            for h in Hs:
                dot = Sphere(radius=0.05, color=BLACK).move_to(h * scale)
                hs_dots.add(dot)

        # 4. Выделенная точка P и луч от начала координат
        P_dot = Sphere(radius=0.15, color=BLUE_C).move_to(P * scale)

        # Луч от начала координат через P (пунктирная линия)
        ray_end = max(np.linalg.norm(H), np.linalg.norm(P)) * P / np.linalg.norm(P) * scale
        ray_line = DashedLine(
            start=Zero * scale,
            end=ray_end,
            color=BLUE_C,
            stroke_width=3,
            dash_length=0.1
        )
        ray_line.set_opacity(0.6)

        # Подпись к точке P (длина волны)
        p_label = Text(f"{wavelengths[point_idx]} nm", font_size=24, color=BLUE_C)
        p_label.move_to(P * scale + np.array([0, 0, 0.5]))
        p_label.add_updater(lambda m: m.rotate(self.camera.get_phi(), axis=RIGHT))

        # 5. Точка проекции H на плоскости RGB
        if SHOW_PROJECTION:
            H_dot = Sphere(radius=0.2, color=BLUE_C).move_to(H * scale)

        # 6. Декомпозиция векторов
        if SHOW_DECOMPOSITION:
            # Вектор xBR (красный) от B
            vector_xBR = Arrow3D(
                start=B * scale,
                end=(B + xBR) * scale,
                color=RED,
                thickness=0.02,
                height=0.25,
                base_radius=0.1
            )

            # Вектор yBG (зелёный) от M
            vector_yBG = Arrow3D(
                start=M * scale,
                end=(M + yBG) * scale,
                color=GREEN,
                thickness=0.02,
                height=0.25,
                base_radius=0.1
            )

            # Вектор BH (чёрный) от B
            vector_BH = Arrow3D(
                start=B * scale,
                end=(B + BH) * scale,
                color=BLACK,
                thickness=0.02,
                height=0.25,
                base_radius=0.1
            )

        # Заголовок
        title = Text("CIE XYZ 3D Space", font_size=48)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)  # Фиксируем в пространстве экрана


        # === ДОБАВЛЕНИЕ ОБЪЕКТОВ НА СЦЕНУ ===

        # Оси
        self.add(x_axis, y_axis, z_axis)
        self.add(x_label, y_label, z_label)

        # Спектральная кривая
        self.add(spectrum_line)
        self.add(spectrum_dots)

        # Плоскость RGB и проекции
        if SHOW_PROJECTION:
            self.add(rgb_plane)
            self.add(hs_dots)
            self.add(H_dot)

        # Луч и выделенная точка
        self.add(ray_line)
        self.add(P_dot)
        self.add(p_label)

        # Декомпозиция векторов
        if SHOW_DECOMPOSITION:
            self.add(vector_xBR)
            self.add(vector_yBG)
            self.add(vector_BH)

        # Заголовок
        self.add(title)

        # Пауза для отображения
        self.wait(3)
