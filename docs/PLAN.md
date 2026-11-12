# Plan en ablatievolgorde

Doel: het best haalbare KL (lager is beter) op HMS Harmful Brain Activity Classification, binnen
Kaggle-GPU-limieten. Dit is een sandbox, niet de cursusinzending.

## Gebruikte bronnen (alleen wat ik effectief las)
- Kaggle competitiepagina en dataset: <https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification>
- Samenvatting van de 2e-plaatsoplossing (blog van yuto over de oplossing van coolz):
  <https://dev.to/moyuto/kaggle-hms-2nd-solution-explaind-34jj>. Daaruit gebruik ik de ideeën: double-banana
  montage, EEG als 2D-beeld en als STFT-spectrogram, twee-staps training (stap 1 gewogen met aantal
  annotators, stap 2 alleen vensters met >= 6 annotators), links/rechts-spiegeling als augmentatie,
  validatie op vensters met >= 6 annotators. Opmerking: de blog bevat zelf inconsistenties (bijv. gewichten van
  het ensemble), dus ik neem enkel de ideeën over en niet de getallen.
- Cranjis-McB/HMS-Harmful-Brain-Activity-Classification (voorbeeld van een EfficientNet-aanpak met vote-gewogen
  KL-loss, same-class CutMix en twee-staps training): <https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification>
- Vergelijkende studie van deep-learning-aanpakken op HMS (arXiv 2412.07878): <https://arxiv.org/abs/2412.07878>
- Timm (modellen): Wightman, PyTorch Image Models, <https://github.com/huggingface/pytorch-image-models>

Alle externe ideeën worden in de README herhaald waar ze worden toegepast. Getallen uit die bronnen zijn niet
gecontroleerd en worden enkel als orde van grootte gebruikt (topoplossingen ~0.27 private, Kaggle-startnotebook ~0.55).

## Feiten over de data (zelf gecontroleerd op train.csv)
- 106 800 rijen, 17 089 eeg_id's, 11 138 spectrogram_id's, 1 950 patiënten. Mediaan 3 rijen per eeg_id, max 743.
- Aantal annotators per rij: vooral 3 (51 867 rijen) en 10-20; slechts ~35 % van de eeg_id's heeft een rij met >= 6.
- Spectrogramkolommen staan in de volgorde LL, RL, LP, RP (niet LL, LP, RP, RL). De code sorteert op naam.
- Spectrogram: 2 s per rij, gelabeld venster = middelste 10 s van 300 rijen; EEG: 200 Hz, label = middelste 10 s van 50 s.

## Ontwerpkeuzes
- Eenheid = eeg_id; elke epoch een willekeurig gelabeld venster per eeg_id (voorkomt dat lange opnames domineren).
- CV: StratifiedGroupKFold op patient_id, gestratificeerd op de dominante klasse van de som van de stemmen.
- Validatie: per eeg_id het venster met de meeste annotators; twee rapporten: alle en >= 6 annotators (dichter bij test, 3-20).
- Gefilterd per venster (niet per opname), zodat train- en testvensters dezelfde randeffecten hebben.
- Spiegelen (LL<->RL, LP<->RP) behoudt het label; gebruikt als augmentatie en als TTA.
- Loss: KL(target || pred); stap 1 gewogen met min(n,20)/20; stap 2 alleen vensters met >= 6 annotators.
- Evaluatie van de laatste epoch met EMA-gewichten (geen selectie op de validatieset -> geen optimistische OOF).

## Ablatievolgorde (elk experiment: 5-fold CV-KL in outputs/results.csv)
1. Basis: `spec` (Kaggle-spectrogram) EfficientNet-B0, 1 stap zonder gewichten.
2. + vote-gewogen loss, + stap 2 (>= 6 annotators).
3. Invoer: `eegspec` (eigen STFT uit EEG) en `eeg1d` (ruwe EEG CNN+GRU).
4. Fusie spectrogram + EEG (`fusion`).
5. Grotere backbone (B2/B3/B5), spec_rows 300 vs 120, augmentaties (mixup/cutmix zelfde klasse, SpecAugment).
6. Pseudo-labelling / label-smoothing van laagconsensus-vensters.
7. Ensemble (gewichten via OOF, `hmsbest.ensemble`) en TTA.
Gefaalde experimenten worden ook gelogd.

## Aanvullende bronbevindingen (2e-plaatsoplossing, via de blog van yuto; de blog is zelf inconsistent over ensemblegewichten)
- Ruwe EEG als 2D-beeld: 16 kanalen x 10 000 samples herschikt naar 160 x 1000, gefilterd 0.5-20 Hz, voorgetrainde
  EfficientNet-B5. Dit was daar het sterkste enkelvoudige model (zelf gerapporteerd private ~0.283). -> `eegimg` in `models.py`.
- Doelwaarden gemiddeld per eeg_id bij een willekeurig gekozen venster (label-ruis verlagen). -> `target_mean_w` in `dataset.py`.
- Stap 1: 15 epochs, loss-gewicht voters/20; stap 2: 5 epochs, lr 1e-4, alleen >= 6 annotators; 10-fold CV, validatie op >= 6.
- Ensemble van 6 modellen (spectrogram, ruwe EEG, gemengd) gaf slechts een kleine winst (~0.28 -> 0.27 private).
Eigen vaststellingen: de testset heeft 3-20 annotators (competitiepagina), dus naast >= 6 wordt ook "alle" gerapporteerd.
