# ROT (Rotating Variable)

Вращательные переменные — звёзды с пятнами на поверхности. При вращении пятно уходит/приходит, меняя яркость.

## Характерные признаки

- **Синусоидальная форма** (гладкая, без резких провалов)
- **Одна волна за период** (одно пятно) или две (два пятна)
- **Амплитуда обычно мала** (0.01 - 0.2 mag)
- **Форма может меняться** со временем (эволюция пятен)

## Фазовая кривая

```
mag ↑
         _____
        /     \
   ____/       \____

  0    0.25   0.5   0.75   1.0
       max         min
```

## Подтипы

| Подтип | Описание |
|--------|----------|
| BY | BY Dra — K-M карлики с пятнами |
| RS | RS CVn — активные двойные |
| SXARI | Магнитные Ap/Bp звёзды |

## Правила для ROT

### Epoch = момент МАКСИМУМА яркости
```python
# Найти фазу максимума яркости (минимума магнитуды)
idx_max = np.argmin(binned_mags[:50])  # первый цикл
epoch_phase = phase_centers[idx_max]
epoch_bjd = btjd.min() + epoch_phase * period + 2457000
```

### Проверка P vs 2P (КРИТИЧНО!)
```python
# Если на 2P появляются ДВЕ одинаковые волны — период = P
# Если на 2P появляются две РАЗНЫЕ волны — возможно ELL, период = 2P
```

### Детренд почти всегда нужен
ROT звёзды часто слабые, тренд заметен. Линейный детренд безопасен если > 10 циклов.

## Info box для ROT
```python
info_text = f'''Epoch (max) = BJD {epoch_bjd:.2f}
Range: {mag_bright:.2f} - {mag_faint:.2f}
Amplitude: {amplitude:.2f} mag'''
```

## Заголовок
```python
# Всегда с двоеточием (uncertain) — ROT сложно отличить от pulsating
ax.set_title(f'TIC {tic_id} — ROT: (P = {period:.2f} d)')
```

## Remarks для VSX
```
Single-wave sinusoidal variation consistent with rotational modulation.
Linear detrending applied to remove instrumental drift.
Period checked against 2P — no double-wave structure.
Epoch is BJD (TDB).
```

## Отличие от других типов

| Признак | ROT | DSCT | ELL |
|---------|-----|------|-----|
| Форма | Синусоида | Может быть острее | Двойная волна |
| Период | 0.2 - 50 d | < 0.3 d обычно | = Porb/2 |
| Амплитуда | 0.01-0.2 mag | 0.01-0.5 mag | < 0.1 mag |
| P vs 2P | Одинаковые волны | — | Разные глубины |

## Пример: TIC 610538470
- Period: 0.87 d
- Amplitude: 0.11 mag
- Форма: чистая синусоида
- P vs 2P: две одинаковые волны → ROT подтверждён
