# EA (Algol-type Eclipsing Binary)

Затменные двойные типа Алголя — две звезды на орбите, периодически затмевающие друг друга.

## Характерные признаки

- **V-образные или U-образные провалы** (затмения)
- **Два затмения за период**: primary (глубже) и secondary (мельче)
- **Плоское дно** между затмениями (в отличие от EW)
- Глубина primary ≠ глубина secondary

## Фазовая кривая

```
mag ↑
     ___________         ___________
    /           \       /           \
   /             \_____/             \
  primary        secondary           primary
  (глубокий)     (мелкий)

  0     0.25    0.5     0.75    1.0
```

## Правила для EA

### Epoch = момент PRIMARY минимума
```python
# Найти фазу минимума яркости (максимума магнитуды)
idx_primary = np.argmax(binned_mags[:50])  # первый цикл
epoch_phase = phase_centers[idx_primary]
epoch_bjd = btjd.min() + epoch_phase * period + 2457000
```

### Фаза 0 = primary eclipse
```python
# Сдвинуть фазу чтобы primary был на 0
phase = ((btjd - epoch_btjd) % period) / period
```

### Проверить P vs 2P
Если "primary" и "secondary" одинаковой глубины — возможно это EW и период = 2P.

## Info box для EA
```python
info_text = f'''Epoch (min I) = HJD {epoch_hjd:.4f}
Range: {mag_bright:.2f} - {mag_faint:.2f}
Pri. eclipse: {primary_depth:.2f} mag
Sec. eclipse: {secondary_depth:.2f} mag'''
```

## Заголовок
```python
ax.set_title(f'TIC {tic_id} — EA (P = {period:.4f} d)')
# или EA: если не 100% уверен
```

## Remarks для VSX
```
Eclipsing binary with well-defined primary minimum.
Secondary eclipse visible at phase ~0.5.
Epoch is BJD (TDB).
```

## Пример: TIC 610495795
- Period: 4.289694 d
- Primary: 0.18 mag
- Secondary: ~0.08 mag (phase 0.5)
- Тип подтверждён: чёткие V-образные затмения
