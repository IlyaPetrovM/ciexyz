import plotly.graph_objects as go
from typing import Any, Tuple, List
import numpy as np


class VisPlotlyInterface:
    """Интерфейс для инкапсуляции вызовов библиотеки Plotly с WebGL"""

    def __init__(self, fig: go.Figure, subplot_index: Tuple[int, int] = None, is_3d: bool = False):
        """
        Args:
            fig: Plotly Figure объект
            subplot_index: Индекс subplot (row, col), если None - используется основной график
            is_3d: Флаг 3D графика
        """
        self.fig = fig
        self.subplot_index = subplot_index
        self.is_3d = is_3d
        self.traces = []  # Хранение ссылок на traces для удаления
        self._xlim = None
        self._ylim = None
        self._zlim = None

    def _get_subplot_kwargs(self):
        """Получить параметры для subplot"""
        if self.subplot_index:
            return {'row': self.subplot_index[0], 'col': self.subplot_index[1]}
        return {}

    def scatter(self, x, y, z=None, **kwargs) -> Any:
        """Отрисовка точек"""
        # Преобразование matplotlib параметров в plotly
        size = kwargs.get('s', 30)
        color = kwargs.get('color', 'blue')
        marker_symbol = kwargs.get('marker', 'circle')

        # Преобразование matplotlib маркеров в plotly
        marker_map = {'o': 'circle', '.': 'circle', 's': 'square', '^': 'triangle-up'}
        marker_symbol = marker_map.get(marker_symbol, marker_symbol)

        if self.is_3d and z is not None:
            trace = go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(size=size/5, color=color, symbol=marker_symbol),
                showlegend=False,
                hoverinfo='text',
                hovertext=f'({x:.3f}, {y:.3f}, {z:.3f})'
            )
        else:
            trace = go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(size=size/2, color=color, symbol=marker_symbol),
                showlegend=False
            )

        self.fig.add_trace(trace, **self._get_subplot_kwargs())
        self.traces.append(len(self.fig.data) - 1)
        return trace

    def scatter_multiple(self, xs, ys, **kwargs) -> Any:
        """Отрисовка массива точек в 2D"""
        size = kwargs.get('s', 30)
        color = kwargs.get('color', 'blue')
        marker_symbol = kwargs.get('marker', 'circle')
        alpha = kwargs.get('alpha', 1.0)

        marker_map = {'o': 'circle', '.': 'circle', 's': 'square', '^': 'triangle-up'}
        marker_symbol = marker_map.get(marker_symbol, marker_symbol)

        trace = go.Scatter(
            x=xs, y=ys,
            mode='markers',
            marker=dict(size=size/2, color=color, symbol=marker_symbol, opacity=alpha),
            showlegend=False
        )

        self.fig.add_trace(trace, **self._get_subplot_kwargs())
        self.traces.append(len(self.fig.data) - 1)
        return trace

    def plot(self, xs, ys, zs=None, **kwargs) -> Any:
        """Отрисовка линии"""
        color = kwargs.get('color', 'blue')
        linewidth = kwargs.get('linewidth', 2)
        alpha = kwargs.get('alpha', 1.0)
        label = kwargs.get('label', None)
        linestyle = kwargs.get('linestyle', 'solid')

        # Преобразование matplotlib linestyle в plotly dash
        dash_map = {'-': 'solid', '--': 'dash', ':': 'dot', '-.': 'dashdot'}
        dash = dash_map.get(linestyle, 'solid')

        if self.is_3d and zs is not None:
            trace = go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(color=color, width=linewidth, dash=dash),
                opacity=alpha,
                name=label,
                showlegend=label is not None
            )
        else:
            trace = go.Scatter(
                x=xs, y=ys,
                mode='lines',
                line=dict(color=color, width=linewidth, dash=dash),
                opacity=alpha,
                name=label,
                showlegend=label is not None
            )

        self.fig.add_trace(trace, **self._get_subplot_kwargs())
        self.traces.append(len(self.fig.data) - 1)
        return [trace]  # Возвращаем список для совместимости

    def text(self, x, y, z, label, **kwargs) -> Any:
        """Отрисовка текста в 3D"""
        color = kwargs.get('color', 'black')

        # В Plotly текст добавляется как аннотация
        if self.is_3d:
            # Для 3D используем Scatter3d с текстом
            trace = go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='text',
                text=[label],
                textfont=dict(color=color, size=10),
                showlegend=False
            )
            self.fig.add_trace(trace, **self._get_subplot_kwargs())
            self.traces.append(len(self.fig.data) - 1)
            return trace
        return None

    def text_2d(self, x, y, label, **kwargs) -> Any:
        """Отрисовка текста в 2D"""
        color = kwargs.get('color', 'black')
        fontsize = kwargs.get('fontsize', 10)

        # Используем annotations для 2D текста
        # Для горизонтального layout (rows=1, cols=N) используем col index для обоих осей
        if self.subplot_index:
            col = self.subplot_index[1]
            xref = 'x' if col == 1 else f'x{col}'
            yref = 'y' if col == 1 else f'y{col}'
        else:
            xref = 'x'
            yref = 'y'

        annotation = dict(
            x=x, y=y,
            text=label,
            showarrow=False,
            font=dict(color=color, size=fontsize),
            xref=xref,
            yref=yref
        )
        self.fig.add_annotation(annotation)
        return annotation

    def quiver(self, x, y, z, u, v, w, **kwargs) -> Any:
        """Отрисовка вектора в 3D"""
        color = kwargs.get('color', 'blue')
        linewidth = kwargs.get('linewidth', 2)
        alpha = kwargs.get('alpha', 1.0)
        arrow_length = kwargs.get('arrow_length_ratio', 0.1)

        # Рисуем вектор как линию со стрелкой
        trace = go.Scatter3d(
            x=[x, x + u],
            y=[y, y + v],
            z=[z, z + w],
            mode='lines',
            line=dict(color=color, width=linewidth),
            opacity=alpha,
            showlegend=False
        )

        # Добавляем стрелку на конце (конус)
        arrow_trace = go.Cone(
            x=[x + u * (1 - arrow_length)],
            y=[y + v * (1 - arrow_length)],
            z=[z + w * (1 - arrow_length)],
            u=[u * arrow_length],
            v=[v * arrow_length],
            w=[w * arrow_length],
            colorscale=[[0, color], [1, color]],
            showscale=False,
            showlegend=False,
            sizemode='absolute',
            sizeref=0.1
        )

        self.fig.add_trace(trace, **self._get_subplot_kwargs())
        self.fig.add_trace(arrow_trace, **self._get_subplot_kwargs())
        self.traces.append(len(self.fig.data) - 2)
        self.traces.append(len(self.fig.data) - 1)
        return trace

    def quiver_2d(self, x, y, u, v, **kwargs) -> Any:
        """Отрисовка вектора в 2D"""
        color = kwargs.get('color', 'blue')
        alpha = kwargs.get('alpha', 1.0)
        width = kwargs.get('width', 0.005)

        # Определяем xref и yref для subplot
        # Для горизонтального layout (rows=1, cols=N) используем col index для обоих осей
        if self.subplot_index:
            col = self.subplot_index[1]
            xref = 'x' if col == 1 else f'x{col}'
            yref = 'y' if col == 1 else f'y{col}'
        else:
            xref = 'x'
            yref = 'y'

        # Рисуем стрелку
        # Plotly не имеет прямой поддержки quiver для 2D, используем annotations
        annotation = dict(
            x=x + u,
            y=y + v,
            ax=x,
            ay=y,
            xref=xref,
            yref=yref,
            axref=xref,
            ayref=yref,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=width * 1000,
            arrowcolor=color,
            opacity=alpha
        )
        self.fig.add_annotation(annotation)
        return annotation

    def add_collection_3d(self, collection) -> None:
        """Добавление 3D коллекции (для совместимости с matplotlib)"""
        # Matplotlib Poly3DCollection -> Plotly Mesh3d
        # Это сложное преобразование, упрощенная версия
        if hasattr(collection, '_vec'):
            verts = collection._vec
            if len(verts) > 0:
                # Берем первый полигон
                poly = verts[0]
                xs, ys, zs = zip(*poly)

                trace = go.Mesh3d(
                    x=xs, y=ys, z=zs,
                    opacity=0.25,
                    color='gray',
                    showlegend=False
                )
                self.fig.add_trace(trace, **self._get_subplot_kwargs())
                self.traces.append(len(self.fig.data) - 1)

    def add_collection(self, collection) -> None:
        """Добавление 2D коллекции"""
        # Matplotlib PolyCollection -> Plotly Shape
        # Упрощенная версия
        if hasattr(collection, 'get_paths'):
            paths = collection.get_paths()
            facecolors = collection.get_facecolors()
            alpha = facecolors[0][3] if len(facecolors) > 0 else 0.4
            color = f'rgba({int(facecolors[0][0]*255)},{int(facecolors[0][1]*255)},{int(facecolors[0][2]*255)},{alpha})'

            for path in paths:
                vertices = path.vertices
                xs, ys = vertices[:, 0], vertices[:, 1]

                trace = go.Scatter(
                    x=list(xs) + [xs[0]],
                    y=list(ys) + [ys[0]],
                    fill='toself',
                    fillcolor=color,
                    line=dict(color=color),
                    mode='lines',
                    showlegend=False
                )
                self.fig.add_trace(trace, **self._get_subplot_kwargs())
                self.traces.append(len(self.fig.data) - 1)

    def clear(self) -> None:
        """Очистка графика"""
        # Удаляем только traces, добавленные этим интерфейсом
        if hasattr(self, 'traces'):
            # Удаляем в обратном порядке, чтобы индексы не сбивались
            for idx in sorted(self.traces, reverse=True):
                if idx < len(self.fig.data):
                    self.fig.data = list(self.fig.data[:idx]) + list(self.fig.data[idx+1:])
            self.traces.clear()

    def set_limits(self, xlim, ylim, zlim=None) -> None:
        """Установка пределов осей"""
        self._xlim = xlim
        self._ylim = ylim
        self._zlim = zlim

        subplot_kwargs = self._get_subplot_kwargs()

        if self.is_3d and zlim is not None:
            self.fig.update_scenes(
                xaxis=dict(range=[xlim[0], xlim[1]]),
                yaxis=dict(range=[ylim[0], ylim[1]]),
                zaxis=dict(range=[zlim[0], zlim[1]]),
                **subplot_kwargs
            )
        else:
            self.fig.update_xaxes(range=[xlim[0], xlim[1]], **subplot_kwargs)
            self.fig.update_yaxes(range=[ylim[0], ylim[1]], **subplot_kwargs)

    def get_limits(self):
        """Получение пределов осей"""
        if self._xlim and self._ylim:
            return self._xlim, self._ylim, self._zlim
        return (0, 1), (0, 1), (0, 1) if self.is_3d else None

    def set_labels(self, xlabel, ylabel, zlabel=None) -> None:
        """Установка подписей осей"""
        subplot_kwargs = self._get_subplot_kwargs()

        if self.is_3d and zlabel is not None:
            self.fig.update_scenes(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=zlabel,
                **subplot_kwargs
            )
        else:
            self.fig.update_xaxes(title_text=xlabel, **subplot_kwargs)
            self.fig.update_yaxes(title_text=ylabel, **subplot_kwargs)

    def set_title(self, title) -> None:
        """Установка заголовка"""
        # Для subplot используем annotations
        if self.subplot_index:
            # Добавляем заголовок как аннотацию
            pass  # Упрощено для примера
        else:
            self.fig.update_layout(title=title)

    def legend(self, **kwargs) -> None:
        """Отрисовка легенды"""
        self.fig.update_layout(showlegend=True)

    def grid(self, visible, **kwargs) -> None:
        """Установка сетки"""
        subplot_kwargs = self._get_subplot_kwargs()

        if self.is_3d:
            self.fig.update_scenes(
                xaxis=dict(showgrid=visible),
                yaxis=dict(showgrid=visible),
                zaxis=dict(showgrid=visible),
                **subplot_kwargs
            )
        else:
            self.fig.update_xaxes(showgrid=visible, **subplot_kwargs)
            self.fig.update_yaxes(showgrid=visible, **subplot_kwargs)

    def axhline(self, y, **kwargs) -> None:
        """Горизонтальная линия"""
        alpha = kwargs.get('alpha', 1.0)
        color = kwargs.get('color', 'gray')
        linestyle = kwargs.get('linestyle', 'solid')

        # Преобразование matplotlib linestyle в plotly dash
        dash_map = {'-': 'solid', '--': 'dash', ':': 'dot', '-.': 'dashdot', 'solid': 'solid', 'dash': 'dash'}
        dash = dash_map.get(linestyle, 'solid')

        # Определяем xref и yref для subplot
        if self.subplot_index:
            col = self.subplot_index[1]
            xref = 'x' if col == 1 else f'x{col}'
            yref = 'y' if col == 1 else f'y{col}'
        else:
            xref = 'x'
            yref = 'y'

        # Используем add_shape вместо add_hline для корректной работы с mixed subplots
        self.fig.add_shape(
            type="line",
            x0=0, x1=1,
            y0=y, y1=y,
            xref=xref, yref=yref,
            xsizemode="scaled",
            line=dict(color=color, dash=dash, width=1),
            opacity=alpha
        )

    def axvline(self, x, **kwargs) -> None:
        """Вертикальная линия"""
        alpha = kwargs.get('alpha', 1.0)
        color = kwargs.get('color', 'gray')
        linestyle = kwargs.get('linestyle', 'solid')

        # Преобразование matplotlib linestyle в plotly dash
        dash_map = {'-': 'solid', '--': 'dash', ':': 'dot', '-.': 'dashdot', 'solid': 'solid', 'dash': 'dash'}
        dash = dash_map.get(linestyle, 'solid')

        # Определяем xref и yref для subplot
        if self.subplot_index:
            col = self.subplot_index[1]
            xref = 'x' if col == 1 else f'x{col}'
            yref = 'y' if col == 1 else f'y{col}'
        else:
            xref = 'x'
            yref = 'y'

        # Используем add_shape вместо add_vline для корректной работы с mixed subplots
        self.fig.add_shape(
            type="line",
            x0=x, x1=x,
            y0=0, y1=1,
            xref=xref, yref=yref,
            ysizemode="scaled",
            line=dict(color=color, dash=dash, width=1),
            opacity=alpha
        )

    def set_aspect(self, aspect) -> None:
        """Установка соотношения сторон"""
        if aspect == 'equal':
            self.fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
                **self._get_subplot_kwargs()
            )


class PlotlyControl:
    """Интерфейс для слайдера в Plotly"""

    def __init__(self, fig: go.Figure, label: str, valmin: int, valmax: int, valinit: int, valstep: int):
        """
        Args:
            fig: Plotly Figure объект
            label: Метка слайдера
            valmin: Минимальное значение
            valmax: Максимальное значение
            valinit: Начальное значение
            valstep: Шаг изменения
        """
        self.fig = fig
        self.label = label
        self.valmin = valmin
        self.valmax = valmax
        self.val = valinit
        self.valstep = valstep
        self.callback = None

        # Создаем слайдер через layout
        steps = []
        for i in range(valmin, valmax + 1, valstep):
            step = dict(
                method="skip",  # Мы будем обрабатывать через callback
                args=[],
                label=str(i),
                value=str(i)
            )
            steps.append(step)

        slider = dict(
            active=valinit,
            currentvalue={"prefix": f"{label}: ", "visible": True},
            steps=steps
        )

        self.fig.update_layout(
            sliders=[slider]
        )

    def on_changed(self, callback):
        """Привязка callback функции к изменению слайдера"""
        self.callback = callback
        # Примечание: в Plotly слайдеры работают иначе,
        # нужна интеграция через Dash или Jupyter widgets для полной интерактивности

    def get_val(self):
        """Получение текущего значения слайдера"""
        return self.val

    def set_val(self, val):
        """Установка значения слайдера"""
        self.val = val
        if self.callback:
            self.callback(val)
