# Ideeën uit top-oplossingen (HMS Harmful Brain Activity)

Bronnen (allemaal publiek, geraadpleegd 12-11-2026):
- [S1] 1e plaats Team Sony, Kaggle writeup: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/writeups/team-sony-1st-place-solution-team-sony
- [S2] 2e plaats (yuto), blog op dev.to / zenn (raw EEG als 160x1000 beeld, X3D-L).
- [S3] arXiv 2412.07878 (training-strategie, XY-masking, mixup, window shifting).
- [S4] Cranjis-McB repo (vote-gewogen KL, same-class CutMix, mosaic warm-up, globale normalisatie).
- [S5] Ryan Peters, https://ryanirl.com/blog/hbac (1D ResNet+RNN, multimodale attention, 10-fold).

## Wat de winnaars (S1) concreet deden
- **Validatie**: StratifiedGroupKFold 5-fold; labels per eeg_id opgeteld en genormaliseerd; midden 10 000 samples (50 s); CV enkel op één rij per eeg_id met **≥10 votes**. Die CV bleef gecorreleerd met de LB, ook bij overfit.
- **Raw EEG als één 2D-beeld** (yamash): 18 bipolaire kanalen, drie crops (2000/5000/10000 samples), geresized en samengevoegd → ConvNeXt-atto / InceptionNeXt-tiny. CV ≈ 0.23–0.245 met een heel simpele aanpak.
- **Wavelets** (suguuuuu, kfuji): CWT (Morlet resp. Paul, 0.5–40 Hz beter dan 0.5–20 Hz, 40 schalen) per kanaal, verticaal gestapeld, 512x512, MaxViT-base. STFT en Kaggle-spec werkten bij hen minder goed. Crop 50 s > 25 s > 10 s voor één model, maar kortere vensters droegen meer bij aan het ensemble.
- **Superlets + leerbare temporele filters** (Muku): 1D-conv met kernel = 200 (=1 s) als leerbare filterbank, feature maps gestapeld als beeld voor een timm-model; beste backbone swinv2_tiny_window16 (CV 0.2229). Kleine backbones deden het goed; diversiteit in het ensemble hielp omdat KL gevoelig is voor extreme fouten.
- **Training**: 2 stages (stage 1: >1 vote, stage 2: ≥10 votes), lr 1e-3 → 1e-4, Adan, cosine, per epoch één random eeg_sub_id met bijhorende offset.
- **Label smoothing vóór normalisatie**: +0.02 bij elke vote-count, zodat labels met weinig votes sterker gesmooth worden.
- **Augmentatie**: ±5 s random time-shift, random bandpass (highcut 30–40 Hz; hoogfrequente ruis → meer "other"), XYMasking, mixup.
- **Entmax (alpha≈1.03) i.p.v. softmax** op de outputs: ~0.004 winst op public én private LB (niet altijd in CV).
- **Ensemble**: niet-negatieve lineaire regressie per klasse op de modeluitvoer.

## Prioriteitenlijst voor onze pipeline
Gebaseerd op wat we al hebben (EEGSpec/ImageNet/EEG1D, 2-stage, EMA, mirror, mixup, masking).

1. **eegimg (raw EEG als beeld)**: al geïmplementeerd, nog niet gedraaid. Meest veelbelovend [S1, S2]. Varianten: filter 0.5–20 vs 0.5–40 Hz, multi-crop (2000/5000/10000) zoals yamash.
2. **Label smoothing +0.02 vóór normalisatie** [S1]: triviaal te implementeren in `dataset.py`/`data.py` en goedkoop te testen.
3. **Target blending per eeg_id** (`target_mean_w`) [S2, S1]: al geïmplementeerd; testen met 0.5 en 1.0.
4. **±5 s time-shift** (random offset binnen de 50 s) [S1, S3]: zinvol omdat het label het centrale 10 s-venster is; bij eegspec/eegimg goedkoop.
5. **Validatie op ≥10 votes** rapporteren naast "all" en "≥6" [S1]; stage 1 op alle windows, stage 2 enkel ≥10 testen i.p.v. ≥6.
6. **Entmax-output** (alpha≈1.03) [S1]: pas toevoegen aan het einde, kleine maar consistente winst op LB.
7. **Leerbare 1D-filterbank → 2D-backbone** (Muku) [S1]: vervanger voor onze zwakke EEG1D+BiGRU; EEGNet-achtige conv met kernel 200 gevolgd door timm-backbone.
8. **CWT/superlet-representatie** i.p.v. STFT [S1]: duurder, maar de winnaars zagen er duidelijk winst; kandidaat voor een tweede representatie in het ensemble.
9. **Same-class CutMix, mosaic, globale normalisatie** [S4]; **channel flips / zero 1–4 kanalen** [S5]: extra augmentaties als we blijven overfitten (train 0.19 vs val 0.34).
10. **Ensemble**: NNLS per klasse op OOF-voorspellingen (nu: Nelder-Mead op gewichten in `ensemble.py`); diversiteit (andere representatie + backbone) belangrijker dan grootte.

## Referentie voor onze CV
- Winnaars: CV ≈ 0.22–0.245 per model (≥10 votes), ensemble ≈ 0.22–0.23 en private LB ≈ 0.27.
- Cranjis-repo: OOF 0.414 ↔ public LB 0.34; baseline OOF 0.729 ↔ LB 0.45.
- Onze huidige fold-0 cijfers staan in `docs/RESULTS.md`.
