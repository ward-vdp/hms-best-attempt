# hms-best-attempt

Sandbox om het best haalbare resultaat te zien op
[HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
(6-klassen soft-label classificatie van EEG, metric = KL-divergentie). Niet bedoeld voor de cursusinzending.

- Plan, bronnen en ablatievolgorde: [docs/PLAN.md](docs/PLAN.md)
- Resultaten per experiment: `outputs/results.csv` (en `docs/RESULTS.md`)

## Pakket `hmsbest`
| module | rol |
|---|---|
| `prep` | parquet -> float16 `.npy` cache (EEG bipolair, log-spectrogram) |
| `data` | soft targets, patiënt-gegroepeerde gestratificeerde folds |
| `signal` | montage, band-pass per venster, spectrogram-crop (gedeeld door train en inferentie) |
| `dataset` | eeg_id-dataset, willekeurig venster per epoch |
| `specs` | STFT van EEG op GPU |
| `models` | `spec`, `eegspec` (timm), `eeg1d` (ResNet-GRU), `fusion` |
| `train` | k-fold training, twee stappen, EMA, OOF + log |
| `infer` | offline inferentie en blend voor Kaggle |
| `ensemble` | OOF-blendgewichten |

## Gebruik (op Kaggle)
```
python -m hmsbest.prep  --data_root /kaggle/input/hms-harmful-brain-activity-classification --cache_dir /kaggle/working/cache
python -m hmsbest.train --config configs/spec_b0.yaml
python -m hmsbest.infer --data_root ... --exp /kaggle/working/runs/spec_b0:1.0 --out submission.csv
```
Tests (CPU, synthetische data): `pytest -q`.
