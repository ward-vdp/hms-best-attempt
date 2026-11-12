# Resultaten (KL, lager is beter)

Validatie: per eeg_id het venster met de meeste annotators. `all` = alle eeg_id's, `>=6` = alleen vensters met >= 6 annotators
(dichter bij de topoplossingen). Fold 0 van 5-fold StratifiedGroupKFold(patient_id), seed 42, EMA-gewichten, laatste epoch.
Alle runs op Kaggle (T4). Schema, tenzij anders vermeld: 6 epochs (alle vensters, loss gewogen met annotators) + 3 epochs (>= 6 annotators).

| run | model | KL all | KL >=6 | tijd | opmerking |
|---|---|---|---|---|---|
| spec_b0_f0 | EfficientNet-B0 op Kaggle-spectrogram (4 regio's gestapeld, 300 rijen) | 0.714 | 0.415 | 15 min | eerste baseline |
