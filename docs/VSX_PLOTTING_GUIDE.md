# VSX Plotting Guide

Правила создания графиков для подачи переменных звёзд в VSX (AAVSO).

---

## Часть 1: Обычная кривая блеска (Raw Light Curve)

### 1) Ось X — BTJD

```
Ось X = BTJD = BJD − 2457000
```

**Почему:** так принято для TESS (цифры компактные, без "+2.4602e6").

**Правило:**
- Если в таблице есть BTJD — берём его
- Если есть только BJD — делаем `BTJD = BJD - 2457000`

### 2) Ось Y — T magnitude (inverted)

```
Ось Y = T magnitude (smaller = brighter)
```

**Правила:**
- Всегда подписывай "smaller = brighter"
- Ставь ylim в обратном порядке (вверх = ярче):
```python
ax.set_ylim(ymax, ymin)  # или ax.invert_yaxis()
```

### 3) Очистка данных (мягкая, без подгона)

#### (а) Вырезаем известные проблемные интервалы

Например, лунный scattered light в секторе 70:
```python
mask = (btjd < 3215) | (btjd > 3221)  # выкинуть 3215-3221
```

#### (б) Выкидываем худшие точки по ошибке

Если есть столбец `Tmag_err`:
```python
threshold = np.percentile(mag_err, 99)  # верхний 1%
mask = mag_err < threshold
```

**Важно:** НЕ делаем агрессивный sigma-clip по магнитудам — можно выкинуть реальные минимумы!

### 4) Binned median для читаемости

Поверх облака точек рисуем binned median:

```python
bin_width = 0.05  # дней (≈72 минуты для TESS)
bins = np.arange(btjd.min(), btjd.max(), bin_width)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_medians = [np.median(mag[(btjd >= bins[i]) & (btjd < bins[i+1])])
               for i in range(len(bins)-1)]
```

**Правила:**
- Медиана лучше среднего (устойчивее к выбросам)
- Бины с <5 точек — пропускаем
- Рисуем линией или крупными точками поверх scatter

### 5) Масштаб Y — чтобы видно было вариабельность

Проблема: выбросы "растягивают" auto-scale.

**Решение:**
```python
y_low = np.percentile(mag, 0.5)
y_high = np.percentile(mag, 99.5)
pad = (y_high - y_low) * 0.1
ax.set_ylim(y_high + pad, y_low - pad)  # inverted!
```

### 6) Убираем scientific notation

```python
from matplotlib.ticker import ScalarFormatter
formatter = ScalarFormatter(useOffset=False)
formatter.set_scientific(False)
ax.xaxis.set_major_formatter(formatter)
```

### 7) Легенда — не налезает на данные

```python
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
```

Все детали чистки — в Remarks VSX, не на графике.

### 8) Отметка проблемных участков

Если показываем данные с артефактами:
```python
ax.axvspan(3215, 3221, alpha=0.3, color='lightcoral', label='Lunar light removed')
```

---

## Часть 2: Фазовая кривая (Phase Plot)

### 1) Ось X — Phase (не "Orbital Phase"!)

"Orbital Phase" подразумевает бинарник. Для ROT/pulsating просто "Phase".

```python
phase = ((btjd - epoch) % period) / period
```

**Диапазоны:**
| Вариант | Когда использовать |
|---------|-------------------|
| 0 до 1 | Компактно, один цикл |
| **0 до 2** | Нагляднее, рекомендую |
| -0.5 до 0.5 | Как в учебниках |
| -1 до 1 | Два цикла, центр на нуле |

### 2) Детренд ПЕРЕД фазовым сворачиванием

**Когда нужен:** если есть видимый тренд на raw кривой (>1-2% за сектор)

**Методы:**

| Метод | Когда использовать |
|-------|-------------------|
| Линейный `polyfit(x, y, 1)` | Стандартный, безопасный |
| Квадратичный `polyfit(x, y, 2)` | Если тренд изогнутый |
| Savitzky-Golay | Агрессивнее, только если много циклов (>10) |

**Как применить:**
```python
coef = np.polyfit(btjd, flux_norm, 1)
trend = np.polyval(coef, btjd)
flux_detrend = flux_norm / trend
```

**Правило безопасности:**
```
Число циклов = длина_наблюдений / период

> 10 циклов → детренд безопасен
< 5 циклов → осторожно, можно повредить сигнал
```

### 3) Проверка периода: P vs 2P

**Обязательно перед подачей!**

```python
# Строим оба варианта
periods = [P, P * 2]
```

**Интерпретация:**

| На графике 2P | Значит | Правильный период |
|---------------|--------|-------------------|
| Две одинаковые волны | ROT (пятна) | P |
| Две разные волны (разная глубина) | ELL/EW | 2P |

### 4) Binning для фазовой

```python
n_bins = 100  # больше чем для raw
bins = np.linspace(0, 2, n_bins + 1)
centers = (bins[:-1] + bins[1:]) / 2
medians = [np.median(mag_ext[(phase_ext >= bins[i]) & (phase_ext < bins[i+1])])
           for i in range(n_bins)]
```

### 5) Info box

```python
info_text = f'''Epoch (max) = BJD {epoch:.2f}
Range: {mag_bright:.2f} - {mag_faint:.2f}
Amplitude: {amplitude:.2f} mag'''

ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
```

### 6) Заголовок — честный, без overclaiming

**Хорошо:**
```
TIC XXXXXXXXX — ROT: (P = 0.87 d)      # двоеточие = uncertain
TIC XXXXXXXXX — Periodic variable (P = 0.87 d)
```

**Плохо:**
```
TIC XXXXXXXXX — Rotating Variable      # слишком уверенно
```

---

## Часть 3: Параметры для формы VSX

### Период

- Указывай с реальной точностью (не выдумывай лишние знаки)
- Если ошибка ~1% → один знак после запятой: `0.87 d`
- Если ошибка ~0.1% → два знака: `0.867 d`

### Epoch

- VSX ожидает **HJD**, TESS даёт **BJD**
- Разница мала (~секунды), но честно пиши в Remarks:
  > "Epoch is BJD (TDB)"

### Range (диапазон магнитуд)

- Используй перцентили 5-95% (робастно)
- Не min-max (включает выбросы)

```python
mag_bright = np.percentile(mag, 5)   # максимум яркости
mag_faint = np.percentile(mag, 95)   # минимум яркости
```

### Type

- Если не 100% уверен → добавь двоеточие: `ROT:`
- Варианты для синусоиды:
  - `ROT:` — вращение (пятна)
  - `DSCT:` — пульсации δ Scuti
  - `GDOR:` — пульсации γ Doradus
  - `VAR` — просто переменная (если совсем не ясно)

---

## Часть 4: Чеклист перед подачей

- [ ] Raw кривая: виден ли сигнал? Отмечены ли артефакты?
- [ ] Фазовая: гладкая ли binned линия?
- [ ] Проверен P vs 2P?
- [ ] Детренд применён (если нужен)?
- [ ] Оси подписаны правильно?
- [ ] Заголовок не overclaiming?
- [ ] Период с реальной точностью?
- [ ] Range посчитан робастно (перцентили)?
- [ ] CSV с данными готов?

---

## Ссылки

- [VSX Portal](https://vsx.aavso.org/index.php)
- [AAVSO Period Determination](https://www.aavso.org/period-determination-variable-star)
- [Phase Diagrams Guide (PDF)](https://www.aavso.org/sites/default/files/Chapter12.pdf)
- [Rotating Variables](https://www.aavso.org/rotating-variables-mapping-surfaces-stars)
