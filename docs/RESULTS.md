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
| fusion_eegspec_eeg1d_f0 | late fusie eigen-spec B0 + EEG-CNN | 0.671 | 0.339 | 21 min* | beste >=6 (job 04) |

Job 05 (Kaggle Version 2, XY/SpecAugment-masking + mixup, extra kolom KL >=10):

| run | model | KL all | KL >=6 | KL >=10 | opmerking |
|---|---|---|---|---|---|
| eegspec_mask | B0 eigen-spec + masking (2 banden) | 0.755 | 0.367 | 0.357 | masking alleen: "all" slechter dan zonder (0.699), >=6 gelijk |
| eegspec_mask_mixup | idem + mixup 0.4 | 0.731 | 0.361 | 0.353 | mixup herstelt een stuk van "all" |
| fusion_eegspec_eeg1d_mask | late fusie eigen-spec B0 (masking) + EEG-CNN | 0.667 | **0.329** | **0.321** | beste tot nu toe op >=6 en >=10 |
| eegspec_b2_mask | EfficientNet-B2 eigen-spec + masking | 0.699 | 0.339 | 0.332 | B2 > B0 met dezelfde regularisatie (0.339 vs 0.367 op >=6) |

Observaties (fold 0):
- Trainverlies ~0.19-0.22 tegenover validatie ~0.34 (>=6): sterke overfitting -> regularisatie/augmentatie is de eerstvolgende hefboom.
- Stap 2 (alleen >=6 annotators) verbetert KL op >=6 met ~0.04 maar verslechtert "alle" met ~0.015. De testset heeft 3-20 annotators
  (competitiepagina), dus beide maten blijven staan.
- Alle fusies slaan alle losse modellen op "alle".
- Job 05: masking verlaagt de trainfout niet meer zover (train 0.30 vs 0.20) maar de validatie verbetert alleen mee bij de grotere backbone en de fusie;
  met 6+3 epochs onderfit het B0-model nu waarschijnlijk. Langer trainen en grotere backbone zijn dus logisch bij regularisatie.
- Bron voor de volgende stappen (raw EEG als beeld, label smoothing, time shift): zie docs/IDEAS.md.

Job 06 (Kaggle Version 4, ronde 1; tricks = label smoothing 0.02 vóór normalisatie + ±5 s time-shift + target blending 0.5):

| run | model | KL all | KL >=6 | KL >=10 | opmerking |
|---|---|---|---|---|---|
| eegspec_mask_tricks | B0 eigen-spec + masking + tricks | 0.725 | 0.355 | 0.346 | tricks helpen t.o.v. eegspec_mask (0.755 / 0.367 / 0.357) |
| eegimg_b0_tricks | B0 op raw EEG als beeld (0.5-20 Hz) + tricks | 0.734 | 0.361 | 0.351 | iets slechter dan eigen-spec bij B0 en 9 epochs; geen doorbraak, wel andere representatie voor ensemble |

Ronde 2 (Version 5; Version 4 liet die ronde door een startup-race in `progress()` over: de wachtlus zag de net gestarte runs nog niet):

| run | model | KL all | KL >=6 | KL >=10 | opmerking |
|---|---|---|---|---|---|
| eegimg_b0_hi40_tricks | eegimg, filter 0.5-40 Hz, tricks | 0.748 | 0.367 | 0.356 | 20 Hz-grens is iets beter dan 40 Hz |
| eegimg_b0_plain | eegimg, 0.5-20 Hz, zonder tricks | 0.736 | 0.361 | 0.351 | identiek aan mét tricks (0.361): de tricks doen niets voor eegimg |

Conclusie job 06/07: tricks helpen de spectrogram-variant (-0.012 op >=6) maar niet eegimg; eegimg B0 (0.361) is niet beter dan eegspec B0 (0.355).
Verschillen < ~0.005 zijn op één fold ruis. Sterkste hefboom blijft fusie van verschillende representaties (job 05: 0.329).

Job 08 (Kaggle Version 6, late fusie van verschillende representaties, fold 0; tricks aan, eeg_hi 40 Hz, 6+3 epochs):

| run | model | KL all | KL >=6 | KL >=10 | opmerking |
|---|---|---|---|---|---|
| fusion_spec_img_tricks | eigen-spec B0 (masking) + eegimg B0 | 0.629 | 0.298 | 0.292 | grote sprong t.o.v. beste vorige (0.667 / 0.329 / 0.321) |
| fusion_spec_eeg1d_img_tricks | eigen-spec B0 + EEG-CNN + eegimg B0 | **0.602** | 0.299 | 0.293 | beste "all"; >=6 gelijk aan 2-delige fusie, 33 min/run op T4 |

Conclusie: de ene representatie voegt weinig toe (eegimg B0 alleen 0.361), maar gefuseerd met de spectrogramtak geeft het -0.03 op >=6 en -0.04 à -0.07 op "all".
Diversiteit tussen representaties is de hefboom; de trainfout (0.19) is nog steeds veel lager dan de validatie (0.31).
