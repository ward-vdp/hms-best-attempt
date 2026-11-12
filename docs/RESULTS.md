# Resultaten (KL, lager is beter)

Validatie: per eeg_id het venster met de meeste annotators. `all` = alle eeg_id's, `>=6` = alleen vensters met >= 6 annotators
(dichter bij de topoplossingen). Fold 0 van 5-fold StratifiedGroupKFold(patient_id), seed 42, EMA-gewichten, laatste epoch.
Alle runs op Kaggle (T4). Schema, tenzij anders vermeld: 6 epochs (alle vensters, loss gewogen met annotators) + 3 epochs (>= 6 annotators).

| run | model | KL all | KL >=6 | tijd | opmerking |
|---|---|---|---|---|---|
| spec_b0_f0 | EfficientNet-B0 op Kaggle-spectrogram (4 regio's gestapeld, 300 rijen) | 0.714 | 0.415 | 15 min | eerste baseline |
| eeg1d_f0 | ruwe EEG: ResNet-1D + BiGRU | 0.837 | 0.436 | 20 min* | *2 runs parallel op 2 GPU's |
| eegspec_b0_f0 | EfficientNet-B0 op eigen STFT-spectrogram uit EEG (4 regio's, 0.5-40 Hz) | 0.699 | 0.365 | 20 min* | beter dan Kaggle-spectrogram op >=6 |
| fusion_spec_eeg1d_f0 | late fusie Kaggle-spec B0 + EEG-CNN | 0.660 | 0.374 | 22 min* | beste "alle" |
| fusion_eegspec_eeg1d_f0 | late fusie eigen-spec B0 + EEG-CNN | 0.671 | 0.339 | 21 min* | beste >=6 |

Observaties (fold 0):
- Trainverlies ~0.19-0.22 tegenover validatie ~0.34 (>=6): sterke overfitting -> regularisatie/augmentatie is de eerstvolgende hefboom.
- Stap 2 (alleen >=6 annotators) verbetert KL op >=6 met ~0.04 maar verslechtert "alle" met ~0.015. De testset heeft 3-20 annotators
  (competitiepagina), dus beide maten blijven staan.
- Alle fusies slaan alle losse modellen op "alle".
