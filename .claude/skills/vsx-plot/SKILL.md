---
name: vsx-plot
description: Generate VSX-compliant plots for variable star submission. Creates raw lightcurve and phase-folded plots following AAVSO standards. Use when preparing variable stars for VSX submission.
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
---

# VSX Plot Generator

Универсальный скилл для создания графиков переменных звёзд для подачи в VSX (AAVSO).

## Использование

```
/vsx-plot STAR_000088 period=0.87
/vsx-plot TIC_610538470 period=0.87 type=ROT
/vsx-plot STAR_001250 period=4.29 type=EA sector=70
```

## Параметры

| Параметр | Обязательный | Описание | Пример |
|----------|--------------|----------|--------|
| `star_id` или `tic_id` | Да | Идентификатор звезды | STAR_000088, TIC_610538470 |
| `period` | Да | Период в днях | 0.87 |
| `type` | Нет | Тип переменной (ROT, EA, EW, DSCT, GDOR, VAR) | ROT |
| `sector` | Нет | Номер сектора TESS (по умолчанию 70) | 70 |
| `camera` | Нет | Камера (по умолчанию 1) | 1 |
| `ccd` | Нет | CCD (по умолчанию 1) | 1 |

## Workflow

При вызове `/vsx-plot STAR_XXXXXX period=X`:

### 1. Определение параметров
```python
# Найти TIC ID из tic_ids.csv
tic_file = f'data/tess/sector_{sector:03d}/cam{camera}_ccd{ccd}/s{sector:04d}_{camera}-{ccd}_tic_ids.csv'
# Формат: star_id,tic_id,separation_arcsec

# Загрузить Tmag из TIC каталога (через astroquery если нет локально)
```

### 2. Загрузка данных
```python
photometry_file = f'data/tess/sector_{sector:03d}/cam{camera}_ccd{ccd}/s{sector:04d}_{camera}-{ccd}_photometry.csv'
df = pd.read_csv(photometry_file)
star = df[df['star_id'] == star_id]
```

### 3. Очистка данных
```python
# Artifact windows из config.py
from tess.config import ARTIFACT_WINDOWS

# Применить для текущего сектора
if sector in ARTIFACT_WINDOWS:
    for start, end, reason in ARTIFACT_WINDOWS[sector]:
        mask &= (btjd < start) | (btjd > end)
```

### 4. Определение типа (если не указан)
По форме кривой предложить тип:
- Синусоида → ROT или DSCT
- V-образные провалы → EA
- Непрерывная переменность с двумя минимумами → EW/EB

### 5. Создание графиков по правилам типа
См. файлы в `types/`:
- `types/EA.md` — затменные
- `types/ROT.md` — вращательные
- `types/DSCT.md` — пульсирующие

### 6. Вывод параметров для VSX формы

## Универсальные правила (все типы)

### Ось X для raw: BTJD
```python
ax.set_xlabel('BTJD (BJD − 2457000)')
```

### Ось Y: T magnitude (inverted)
```python
ax.set_ylabel('T magnitude (smaller = brighter)')
ax.invert_yaxis()
```

### Ось X для phase: просто "Phase"
```python
ax.set_xlabel('Phase')  # НЕ "Orbital Phase" (только для бинарников)
```

### Диапазон фазы
- `0 до 2` — стандарт, два цикла (рекомендуется)
- `0 до 1` — компактно
- `-0.5 до 0.5` — как в учебниках

### Range через перцентили
```python
mag_bright = np.percentile(mag, 5)   # максимум яркости
mag_faint = np.percentile(mag, 95)   # минимум яркости
```

### Заголовок — не overclaiming
```python
# Двоеточие = uncertain
ax.set_title(f'TIC {tic_id} — {var_type}: (P = {period} d)')
```

### Детренд
```python
# Линейный — безопасный стандарт
coef = np.polyfit(btjd, flux_norm, 1)
trend = np.polyval(coef, btjd)
flux_detrend = flux_norm / trend
```

Правило: безопасно если число циклов > 10

### Проверка P vs 2P (ОБЯЗАТЕЛЬНО для EW/ROT)
Построить оба варианта и сравнить форму.

## Структура выходных файлов

```
variable_stars/TIC_XXXXXXXXX/
├── VSX_rawlc.png              # Raw lightcurve (для подачи)
├── VSX_phase.png              # Phase-folded (для подачи)
├── TIC_XXXXXXXXX_photometry.csv  # Данные
├── period_check_2x.png        # Проверка P vs 2P (рабочий)
└── notes.txt                  # Заметки (опционально)
```

## Конфигурация артефактов

В `src/tess/config.py`:
```python
ARTIFACT_WINDOWS = {
    70: [(3215, 3221, 'lunar scattered light')],
    # Добавлять по мере обработки секторов:
    # 71: [(start, end, 'reason')],
}
```

## Чеклист перед подачей

- [ ] Raw кривая показывает сигнал?
- [ ] Артефакты отмечены?
- [ ] Проверен P vs 2P?
- [ ] Тип соответствует форме?
- [ ] Фазовая кривая гладкая?
- [ ] Epoch на правильной фазе (мин для EA, макс для ROT)?
- [ ] Period с реальной точностью?
- [ ] Range через перцентили?

## Примеры успешных подач

См. `examples/`:
- `examples/TIC_610495795.md` — EA, P=4.29d (первая подача)

## Ссылки

- [VSX Portal](https://vsx.aavso.org/)
- [Подробный гайд](docs/VSX_PLOTTING_GUIDE.md)
- [AAVSO Period Determination](https://www.aavso.org/period-determination-variable-star)
