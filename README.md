# ciexyz

## Как запустить интерактивную визуализацию
1. Установите зависимости: `pip install plotly dash numpy`
2. Запустите: `python main.py`
3. Откройте браузер по адресу: http://127.0.0.1:8050

## Manim анимации

### Установка Manim
```bash
pip install manim
```

### Доступные анимации

#### 1. CMF Plot (2D график функций цветового соответствия)
Визуализирует кривые отклика R, G, B в зависимости от длины волны.

**Файл:** `plot_cmf_manim.py`

**Запуск:**
```bash
# Низкое качество (быстро, для проверки)
python -m manim -pql plot_cmf_manim.py CMFPlot

# Высокое качество (1080p)
python -m manim -pqh plot_cmf_manim.py CMFPlot
```

#### 2. XYZ 3D Plot (3D пространство CIE XYZ)
Визуализирует спектральную кривую в 3D пространстве RGB с проекцией на плоскость и декомпозицией векторов.

**Файл:** `plot_xyz_3d_manim.py`

**Запуск:**
```bash
# Низкое качество (480p, 15 fps)
python -m manim -pql plot_xyz_3d_manim.py XYZ3DPlot

# Среднее качество (720p, 30 fps)
python -m manim -pqm plot_xyz_3d_manim.py XYZ3DPlot

# Высокое качество (1080p, 60 fps)
python -m manim -pqh plot_xyz_3d_manim.py XYZ3DPlot

# Ускорение рендера (отключение кеширования)
python -m manim -pql --disable_caching plot_xyz_3d_manim.py XYZ3DPlot
```

**Настройки в коде:**
- `initial_point = 39` - выбор точки спектра для визуализации (555 nm)
- `SHOW_PROJECTION = True` - показать плоскость RGB и проекции
- `SHOW_DECOMPOSITION = True` - показать декомпозицию векторов
- `scale = 3.5` - масштаб объектов
- Углы камеры: `phi=65°`, `theta=-50°`

**Результат:** Видео сохраняется в `media/videos/plot_xyz_3d_manim/`

## Screenshots
<img width="1903" height="734" alt="image" src="https://github.com/user-attachments/assets/2fa895a3-7916-466f-aa5a-e53ddafe466f" />

## data sources
https://www.cvrl.org/ 
 
