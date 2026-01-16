# DSCT (δ Scuti) и GDOR (γ Doradus)

Пульсирующие переменные — звёзды которые физически расширяются/сжимаются.

## DSCT (δ Scuti)

### Характерные признаки
- **Короткие периоды**: 0.02 - 0.25 дня (0.5 - 6 часов)
- **Амплитуда**: 0.003 - 0.9 mag (чаще 0.01 - 0.1)
- **Спектральный класс**: A-F
- **Часто многопериодичность** — несколько мод пульсаций

### Фазовая кривая
```
mag ↑
      _
     / \
    /   \
___/     \___

  0   0.5   1.0
```
Может быть асимметричной (быстрый подъём, медленный спуск).

## GDOR (γ Doradus)

### Характерные признаки
- **Более длинные периоды**: 0.3 - 3 дня
- **Амплитуда**: < 0.1 mag обычно
- **Спектральный класс**: A-F (как DSCT)
- **Тоже многопериодичность**

### Как отличить от ROT?
Сложно! Но:
- GDOR чаще имеет несколько близких периодов (биения)
- ROT форма стабильнее во времени

## Правила для DSCT/GDOR

### Epoch = момент МАКСИМУМА яркости
```python
idx_max = np.argmin(binned_mags[:50])
epoch_bjd = btjd.min() + phase_centers[idx_max] * period + 2457000
```

### Проверка многопериодичности
```python
# Lomb-Scargle периодограмма
# Если несколько значимых пиков — возможно multi-periodic
from astropy.timeseries import LombScargle
ls = LombScargle(btjd, flux)
freq, power = ls.autopower()

# Найти несколько пиков
peaks = find_peaks(power, height=0.1 * power.max())
```

### Детренд
Обычно нужен — DSCT/GDOR часто слабые.

## Info box
```python
info_text = f'''Epoch (max) = BJD {epoch_bjd:.4f}
Range: {mag_bright:.2f} - {mag_faint:.2f}
Amplitude: {amplitude:.3f} mag
Type: DSCT or GDOR (uncertain)'''
```

## Заголовок
```python
# Период < 0.25 d → вероятно DSCT
# Период 0.3-3 d → вероятно GDOR или ROT
ax.set_title(f'TIC {tic_id} — DSCT: (P = {period:.4f} d)')
# или
ax.set_title(f'TIC {tic_id} — GDOR|ROT: (P = {period:.3f} d)')
```

## Remarks для VSX
```
Short-period pulsating variable.
Single dominant period detected; possible additional periods not resolved.
Epoch is BJD (TDB).
```

## Таблица: DSCT vs GDOR vs ROT

| Параметр | DSCT | GDOR | ROT |
|----------|------|------|-----|
| Период | < 0.25 d | 0.3-3 d | 0.2-50 d |
| Амплитуда | 0.01-0.5 mag | < 0.1 mag | 0.01-0.2 mag |
| Форма | Асимметричная возможна | Синусоида | Синусоида |
| Многопериодичность | Часто | Часто | Редко |
| Изменение формы | Нет | Нет | Да (пятна) |

## Когда использовать VAR

Если не можешь уверенно отличить DSCT/GDOR/ROT:
```python
ax.set_title(f'TIC {tic_id} — VAR (P = {period:.3f} d)')
```

В Remarks: "Classification uncertain between DSCT/GDOR/ROT based on period and light curve shape."
