import matplotlib.pyplot as plt
import numpy as np

# Создаем полный диапазон длин волн
all_wavelengths = np.arange(380, 785, 5)

# Основные длины волн из эксперимента
main_wavelengths = [445, 555, 650]
intensities_1_values = [0.6, 0.8, 0.5]
intensities_2_values = [0.4, 0.3, 0.7]

# Создаем массивы интенсивностей для всех длин волн
intensities_1 = np.zeros(len(all_wavelengths))
intensities_2 = np.zeros(len(all_wavelengths))

# Заполняем значениями только для основных длин волн
for i, wl in enumerate(main_wavelengths):
    idx = np.where(all_wavelengths == wl)[0][0]
    intensities_1[idx] = intensities_1_values[i]
    intensities_2[idx] = intensities_2_values[i]

intensities_sum = intensities_1 + intensities_2
y_max = max(intensities_sum) * 2

# Функция для преобразования длины волны в RGB
def wavelength_to_rgb(wavelength):
    if 380 <= wavelength < 440:
        r, g, b = -(wavelength - 440) / (440 - 380), 0.0, 1.0
    elif 440 <= wavelength < 490:
        r, g, b = 0.0, (wavelength - 440) / (490 - 440), 1.0
    elif 490 <= wavelength < 510:
        r, g, b = 0.0, 1.0, -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        r, g, b = (wavelength - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= wavelength < 645:
        r, g, b = 1.0, -(wavelength - 645) / (645 - 580), 0.0
    elif 645 <= wavelength <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        r, g, b = 0.0, 0.0, 0.0
    return (r, g, b)

colors = [wavelength_to_rgb(wl) for wl in all_wavelengths]

# Параметры для левых графиков (комбинированные цвета)
# График 4: комбинация для intensities_2 (больше красного) - СУММА всех значений
combined_wl_4 = 610
combined_intensity_4 = sum(intensities_2_values)  # 0.4 + 0.3 + 0.7 = 1.4

# График 5: комбинация для intensities_1 (больше зеленого) - СУММА всех значений
combined_wl_5 = 505
combined_intensity_5 = sum(intensities_1_values)  # 0.6 + 0.8 + 0.5 = 1.9

# График 6: два столбика с теми же высотами
combined_wl_6_first = 610
combined_intensity_6_first = sum(intensities_2_values)  # 1.4
combined_wl_6_second = 505
combined_intensity_6_second = sum(intensities_1_values)  # 1.9

# Создание фигуры с сеткой 3x2
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# График 4 (левый верхний)
axes[0, 0].bar(
 [combined_wl_4], 
 [combined_intensity_4], 
 hatch='..',
 width=20, 
 color = wavelength_to_rgb(combined_wl_4),
 edgecolor='black', 
 linewidth=1.5, 
 alpha=0.7
)

axes[0, 0].set_ylabel('Интенсивность')
axes[0, 0].set_title('График 4')
axes[0, 0].set_xlim(375, 785)
axes[0, 0].set_ylim(0, y_max)
axes[0, 0].grid(axis='y', alpha=0.3)

# График 1 (правый верхний)
axes[0, 1].bar(all_wavelengths, intensities_2, width=20, color=colors, edgecolor='black', 
               hatch='..', linewidth=0.5, alpha=0.7)
axes[0, 1].set_ylabel('Интенсивность')
axes[0, 1].set_title('График 1')
axes[0, 1].set_xlim(375, 785)
axes[0, 1].set_ylim(0, y_max)
axes[0, 1].grid(axis='y', alpha=0.3)

# График 5 (левый средний)
axes[1, 0].bar(
 [combined_wl_5],
 [combined_intensity_5], 
 hatch='//',
 width=20, 
 color=wavelength_to_rgb(combined_wl_5),
 edgecolor='black', 
 linewidth=1.5, 
 alpha=0.7
)
axes[1, 0].set_ylabel('Интенсивность')
axes[1, 0].set_title('График 5')
axes[1, 0].set_xlim(375, 785)
axes[1, 0].set_ylim(0, y_max)
axes[1, 0].grid(axis='y', alpha=0.3)

# График 2 (правый средний)
axes[1, 1].bar(all_wavelengths, intensities_1, width=20, color=colors, edgecolor='black', 
               hatch='//', linewidth=0.5, alpha=0.7)
axes[1, 1].set_ylabel('Интенсивность')
axes[1, 1].set_title('График 2')
axes[1, 1].set_xlim(375, 785)
axes[1, 1].set_ylim(0, y_max)
axes[1, 1].grid(axis='y', alpha=0.3)

# График 6 (левый нижний) - ДВА СТОЛБИКА с полными высотами
axes[2, 0].bar(
    [combined_wl_6_first],
    [combined_intensity_6_first], 
    width=20, 
    color = wavelength_to_rgb(combined_wl_6_first),
    edgecolor='black', 
    hatch='..',
    linewidth=1.5, 
    alpha=0.7, 
    label='График 4'
)

axes[2, 0].bar(
    [combined_wl_6_second], 
    [combined_intensity_6_second], 
    width=20, 
    color=wavelength_to_rgb(combined_wl_6_second), 
    edgecolor='black', 
    hatch='//', linewidth=1.5, alpha=0.7, label='График 5'
)

axes[2, 0].set_xlabel('Длина волны (нм)')
axes[2, 0].set_ylabel('Интенсивность')
axes[2, 0].set_title('График 6 (Сумма)')
axes[2, 0].set_xlim(375, 785)
axes[2, 0].set_ylim(0, y_max)
axes[2, 0].legend()
axes[2, 0].grid(axis='y', alpha=0.3)

# График 3 (правый нижний)
axes[2, 1].bar(
    all_wavelengths, 
    intensities_1, 
    width=20, 
    color=colors, 
    edgecolor='black', 
    hatch='//', 
    linewidth=0.5, alpha=0.7, label='График 2')

axes[2, 1].bar(
    all_wavelengths, intensities_2, width=20, bottom=intensities_1, 
               color=colors, edgecolor='black', hatch='..', linewidth=0.5, alpha=0.7, label='График 1')
axes[2, 1].set_xlabel('Длина волны (нм)')
axes[2, 1].set_ylabel('Интенсивность')
axes[2, 1].set_title('График 3 (Сумма)')
axes[2, 1].set_xlim(375, 785)
axes[2, 1].set_ylim(0, y_max)
axes[2, 1].legend()
axes[2, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
