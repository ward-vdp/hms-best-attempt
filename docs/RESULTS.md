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

## Job 09 (Kaggle Version 7): 5-fold CV van de 3-delige fusie (eigen-spec B0 + EEG-CNN + eegimg B0, tricks, 6+3 epochs)

**5-fold OOF KL: 0.565 (alle 17 089 eeg_id's) / 0.278 (>= 6) / 0.275 (>= 10, 5 939 eeg_id's).**
Per fold op >= 10: 0.293, 0.252, 0.290, 0.270, 0.266 (spreiding ±0.017 -> verschillen < ~0.008 op één fold zijn ruis).

Foutanalyse (uit `oof_f3p.csv`; zelfde berekening als `hmsbest/analyze.py`):

| aantal stemmen | 1-2 | 3 | 4-5 | 6-9 | 10-12 | 13-15 | >= 16 |
|---|---|---|---|---|---|---|---|
| eeg_id's | 4 780 | 4 949 | 1 307 | 114 | 1 111 | 3 177 | 1 651 |
| KL | 0.467 | **1.010** | 0.562 | 0.416 | 0.310 | 0.271 | 0.260 |

1. **Twee annotator-populaties met verschillende labels, niet alleen meer ruis.** Gemiddeld label bij 3 stemmen: 29 % seizure,
   21 % GRDA, 27 % other; bij >= 10 stemmen: 5 % seizure, 9 % GRDA, 48 % other. Het model (na stap 2) voorspelt op de
   3-stem-rijen 11 % seizure en 50 % other. 74-91 % van de laag-stem-labels is unaniem, tegen 14 % bij >= 10.
   -> "KL alle" meet vooral die mismatch en zegt weinig over de test (>= 10 annotators); stuurcijfer = KL >= 10.
   -> Ondersteunt annotator-conditionering (idee 8) en teacher-labels (idee 9); verklaart waarom stap 2 zoveel oplevert.
2. **Kalibratie is al goed.** Beste temperatuur op >= 10 is T = 1.00-1.05 (KL 0.275 -> 0.275); per betrouwbaarheidsklasse klopt
   de voorspelde kans met de werkelijke stemfractie (0.50 vs 0.48, 0.70 vs 0.68, 0.88 vs 0.86, 0.97 vs 0.98).
   -> Temperatuur/entmax-achtige nabewerking levert hier niets op; winst moet uit beter onderscheid komen.
3. **Waar zit de fout (>= 10)?** Aandeel in de KL: LPD 24 %, LRDA 19 %, other 18 %, GPD 17 %, GRDA 14 %, seizure 8 %.
   KL als de klasse dominant is: LRDA 0.445 (305), LPD 0.340, seizure 0.322, GRDA 0.305, GPD 0.286, other 0.223.
   Verwarringen (gemiddelde voorspelde massa vs stemmen): LRDA -> other 0.36 (stemmen 0.25) en -> LPD 0.17 (0.11);
   GRDA -> other 0.39 (0.28). Links/rechts versus gegeneraliseerd wordt nauwelijks verward (LRDA -> GPD 0.01, LPD -> GPD 0.06).
   -> Het knelpunt is *ritmische delta versus achtergrond* en *ritmisch versus periodiek*, niet lateralisatie. Onze STFT
      (venster 0.64 s) heeft ~1.6 Hz resolutie en kan 1 Hz van 2.5 Hz niet scheiden: een tak met fijne delta-resolutie
      (scalogram `eegcwt`, of STFT met 2-4 s venster) is de meest gerichte verbetering. De siamese tak (idee 7) lost een
      probleem op dat klein blijkt.
4. **Moeilijkheid volgt de onenigheid tussen experts**: unanieme doelen KL 0.153, sterke onenigheid 0.34. 10 % van de
   voorspelde massa staat op klassen zonder enkele stem. Top-10 % rijen = 33 % van de KL (geen extreme uitschieters).

## Job 10 (Kaggle Version 8): eigen ideeën op fold 0 (referentie: 3-delige fusie 0.602 / 0.299 / 0.293)

| run | wat | KL all | KL >=6 | KL >=10 | oordeel |
|---|---|---|---|---|---|
| eegcwt_b0_tricks | leerbaar Morlet-scalogram B0, losse tak | 0.663 | 0.348 | 0.340 | **beste losse tak** (eegspec 0.725/0.355/0.346, eegimg 0.734/0.361/0.351) -> in de fusie opnemen |
| fusion3_prior_global | fusie + smoothing naar globale stemverdeling | 0.602 | 0.293 | 0.288 | -0.006 op >=10: klein, gratis -> houden |
| fusion3_votecond | fusie + annotator-conditionering | 0.661 | 0.296 | 0.288 | -0.005 op >=10, "all" slechter (voorspelt altijd in veel-stem-stijl); binnen ruis |
| fusion3_siamese | fusie met hemisfeer-symmetrische takken | 0.673 | 0.347 | 0.338 | **mislukt** (+0.045): past bij de foutanalyse (lateralisatie is niet het knelpunt) en de halve beelden verliezen context -> schrappen |

## Job 12 (Kaggle Version 10): scalogramtak in de fusie, fold 0, batch 32, prior global (referentie fusion3_prior_global 0.602 / 0.293 / 0.288)

| run | takken | KL all | KL >=6 | KL >=10 | oordeel |
|---|---|---|---|---|---|
| fusion4_cwt | STFT-spec + 1D + eegimg + scalogram | 0.610 | 0.290 | 0.283 | -0.004 op >=10: binnen ruis, 59 min/fold (1.7x duurder) |
| fusion3_cwt_noSpec | scalogram + 1D + eegimg | 0.610 | 0.295 | 0.287 | gelijk aan referentie: scalogram en STFT zijn uitwisselbaar in de fusie |

Conclusie: de scalogramtak is los de beste, maar in de fusie levert ze nauwelijks iets extra; LRDA/GRDA -> other blijft
(LRDA 0.35, GRDA 0.42 op other). Op B0 met 6+3 epochs zit fold 0 vast rond 0.283-0.293 ongeacht de representatie:
de volgende hefbomen zijn labels (teacher, job 11), capaciteit/trainingsduur (job 13) en het ensemble.

## Job 11 (Kaggle Version 9): teacher-verfijnde labels, lekvrij op fold 0 (referentie zelfde model 0.602 / 0.299 / 0.293)
Teachers: 4 modellen die fold 0 nooit zagen (elk op 60 % van de data). Geschatte waarde van de teacher:
**tau = 2.83 pseudo-annotators** (maximum likelihood), dekking 78 % van de rijen (alles behalve fold 0).

| run | teacher-labels in | KL all | KL >=6 | KL >=10 | oordeel |
|---|---|---|---|---|---|
| stud_s1 | stap 1; stap 2 zoals voorheen (ruwe labels, >= 6 stemmen) | 0.598 | 0.284 | **0.277** | **-0.016 op >=10**, 2x de foldruis -> overnemen |
| stud_both | stap 1 en stap 2 (stap 2 op alle vensters) | 0.561 | 0.299 | 0.294 | "all" beter, >=10 niet: stap 2 op laag-stem-rijen houdt het model in de laag-stem-stijl |

Verbetering bij stud_s1 zit vooral bij LPD (KL dominant 0.371 -> 0.334) en de 15+-stem-rijen (0.271 -> 0.250).
Conclusie: teacher-labels als betere voortraining (stap 1), daarna nog steeds specialiseren op veel-stem-rijen.

## Job 13 (Kaggle Version 11): capaciteit en trainingsduur, fold 0 (referentie fusion3_prior_global 0.602 / 0.293 / 0.288)

| run | wat | KL all | KL >=6 | KL >=10 | oordeel |
|---|---|---|---|---|---|
| fusion3_b2 | EfficientNet-B2-takken, batch 32, 6+3 epochs | 0.611 | 0.288 | 0.280 | -0.008 op >=10 (grens van de ruis), ~51 min/fold (1.5x) |
| fusion3_b0_long | B0-takken, 10+4 epochs | 0.611 | 0.297 | 0.291 | geen winst (+0.003): langer trainen overfit alleen meer (train 0.159) |

Conclusie: langer trainen schrappen. B2 is een kandidaat voor het tweede ensemble-model (andere backbone = diversiteit),
niet voor de teachers. Job 14 (Version 12, lekvrije 5-fold met teacher-labels, B0, 6+3) loopt.

## Job 14 (Kaggle Version 12): lekvrije 5-fold met teacher-verfijnde labels (B0-fusie, prior global, 6+3 epochs)

**5-fold OOF KL: 0.566 (alle) / 0.268 (>= 6) / 0.266 (>= 10)**; zonder teacher (job 09): 0.565 / 0.278 / 0.275.
Per fold op >= 10 (zonder -> met teacher): 0.293 -> 0.279, 0.252 -> 0.241, 0.290 -> 0.287, 0.270 -> 0.257, 0.266 -> 0.259:
**alle vijf de folds verbeteren**, gemiddeld -0.010. (Let op: job 09 had nog de uniforme prior; die was op fold 0 goed voor
~0.005, dus het zuivere teacher-effect is 0.005-0.010; op fold 0 lekvrij gemeten -0.016 met uniforme prior.)
tau per student: 2.83-3.36 pseudo-annotators. Gecross-fitte kalibratie: 0.2657 -> 0.2641 (klein, T = 1.04).
De 10 paar-teachers zelf (60 % data): 0.280-0.288 op >= 10.
Foutprofiel onveranderd: LRDA blijft het zwakst (KL dominant 0.402; 0.37 massa op other), GRDA -> other 0.40.

## Job 15 (Kaggle Version 13): ensemble van drie lekvrije 5-fold-modellen met teacher-labels (13 774 s ≈ 3.8 GPU-u)

| model | wat | KL all | KL >=6 | KL >=10 |
|---|---|---|---|---|
| A | B0-fusie (job 14) | 0.566 | 0.268 | 0.266 |
| A2 | B0-fusie, seed 7 (zelfde folds) | 0.570 | 0.270 | 0.267 |
| B | B2-fusie, batch 32 | 0.560 | 0.265 | 0.263 |
| gelijk gemiddelde | | 0.555 | 0.260 | 0.257 |
| geometrisch gemiddelde | | 0.557 | 0.260 | 0.257 |
| gecross-fit gewogen blend (≈ 0.28 A / 0.26 A2 / 0.47 B) | | 0.554 | 0.260 | **0.257** |
| + gecross-fitte kalibratie | | 0.575 | 0.258 | **0.256** |

Per fold (>= 10, blend): 0.270 / 0.235 / 0.275 / 0.250 / 0.249.
B2 is het beste losse model (-0.003 t.o.v. A, consistent met fold 0); een tweede seed van hetzelfde model voegt bijna evenveel
toe als een andere backbone: de ensemblewinst (-0.006 t.o.v. B) is vooral variantiereductie. Gewichten stabiel over folds.
Kalibratie helpt op >= 10 (-0.0013) maar schaadt "all" (scherpere uitvoer straft laag-stem-rijen); voor de test (>= 10-achtig) aanhouden.
Foutprofiel (blend, >= 10): LRDA zwakst (KL dominant 0.388; 0.37 massa naar other), GRDA -> other 0.39, LPD grootste aandeel (25 %).
Model is te onzeker: entropie voorspelling 0.89 vs doel 0.64; unaniem doel KL 0.145, sterke onenigheid 0.303.
Referentie: top-oplossingen CV ≈ 0.22-0.245 per model; ons ensemble 0.256 zit daar nog ~0.02-0.03 boven.
