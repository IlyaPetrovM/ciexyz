import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import Any


class VisInterface:
    """Интерфейс для инкапсуляции вызовов библиотеки отрисовки"""

    def __init__(self, ax):
        self.ax = ax

    def scatter(self, x, y, z=None, **kwargs) -> Any:
        """Отрисовка точек"""
        if z is not None:
            return self.ax.scatter([x], [y], [z], **kwargs)
        return self.ax.scatter([x], [y], **kwargs)

    def scatter_multiple(self, xs, ys, **kwargs) -> Any:
        """Отрисовка массива точек в 2D"""
        return self.ax.scatter(xs, ys, **kwargs)

    def plot(self, xs, ys, zs=None, **kwargs) -> Any:
        """Отрисовка линии"""
        if zs is not None:
            return self.ax.plot(xs, ys, zs, **kwargs)
        return self.ax.plot(xs, ys, **kwargs)

    def text(self, x, y, z, label, **kwargs) -> Any:
        """Отрисовка текста"""
        return self.ax.text(x, y, z, label, **kwargs)

    def text_2d(self, x, y, label, **kwargs) -> Any:
        """Отрисовка текста в 2D"""
        return self.ax.text(x, y, label, **kwargs)

    def quiver(self, x, y, z, u, v, w, **kwargs) -> Any:
        """Отрисовка вектора в 3D"""
        return self.ax.quiver(x, y, z, u, v, w, **kwargs)

    def quiver_2d(self, x, y, u, v, **kwargs) -> Any:
        """Отрисовка вектора в 2D"""
        return self.ax.quiver(x, y, u, v, **kwargs)

    def add_collection_3d(self, collection) -> None:
        """Добавление 3D коллекции"""
        self.ax.add_collection3d(collection)

    def add_collection(self, collection) -> None:
        """Добавление 2D коллекции"""
        self.ax.add_collection(collection)

    def clear(self) -> None:
        """Очистка графика"""
        self.ax.clear()

    def set_limits(self, xlim, ylim, zlim=None) -> None:
        """Установка пределов осей"""
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        if zlim is not None:
            self.ax.set_zlim(zlim)

    def get_limits(self):
        """Получение пределов осей"""
        return self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim() if hasattr(self.ax, 'get_zlim') else None

    def set_labels(self, xlabel, ylabel, zlabel=None) -> None:
        """Установка подписей осей"""
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if zlabel is not None:
            self.ax.set_zlabel(zlabel)

    def set_title(self, title) -> None:
        """Установка заголовка"""
        self.ax.set_title(title)

    def legend(self, **kwargs) -> None:
        """Отрисовка легенды"""
        self.ax.legend(**kwargs)

    def grid(self, visible, **kwargs) -> None:
        """Установка сетки"""
        self.ax.grid(visible, **kwargs)

    def axhline(self, y, **kwargs) -> None:
        """Горизонтальная линия"""
        self.ax.axhline(y, **kwargs)

    def axvline(self, x, **kwargs) -> None:
        """Вертикальная линия"""
        self.ax.axvline(x, **kwargs)

    def set_aspect(self, aspect) -> None:
        """Установка соотношения сторон"""
        self.ax.set_aspect(aspect)


class Control:
    """Интерфейс для слайдера"""

    def __init__(self, fig, rect, label, valmin, valmax, valinit, valstep):
        ax_slider = plt.axes(rect)
        self.slider = Slider(ax_slider, label, valmin, valmax, valinit=valinit, valstep=valstep, color='lightblue')

    def on_changed(self, callback):
        """Привязка callback функции к изменению слайдера"""
        self.slider.on_changed(callback)

    def get_val(self):
        """Получение текущего значения слайдера"""
        return self.slider.val
