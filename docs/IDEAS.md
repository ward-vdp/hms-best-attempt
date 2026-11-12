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

## Na de competitie (2024-2026) en eigen ideeën

Nieuwe bron:
- [S6] VIPEEGNet, arXiv 2507.08874 (2025): één EfficientNetV2-B3 met een **leerbare EEG-naar-beeld conv-module** (drie groepen
  kernels) op 16-kanaal double banana, 0.5-45 Hz; 5-fold CV KL 0.225, Kaggle private 0.2725 (≈ 2e plaats) met ~49M parameters,
  2.8% van het winnende ensemble. Ablatie: de leerbare front-end en ImageNet-pretraining zijn de belangrijkste onderdelen.
  Extra augmentaties: kanaal-permutatie, inversie, tijdomkering, kanaal-swap.
- [S7] EEG-foundation-modellen (LaBraM, CBraMod, EEGPT, BIOT; benchmark EEG-FM-Bench, arXiv 2508.17742): voorgetraind op TUH;
  HMS zit niet in de benchmark en er is geen bewijs dat ze hier een gesuperviseerd CNN verslaan. Kandidaat als extra
  fusietak, niet als eerste keuze (zwaar, andere kanaalindeling).

Eigen ideeën (niet letterlijk uit een top-oplossing), in volgorde van verwachte winst per GPU-uur:
1. **Fusie van verschillende representaties in één netwerk** (al getest, job 08): eigen-spec + raw-EEG-beeld + 1D-CNN met één
   gedeelde kop i.p.v. losse modellen achteraf middelen. Fold 0: 0.329 -> 0.299 op >=6. De winnaars middelden losse modellen.
2. **Leerbaar Morlet-scalogram** (`kind: eegcwt`, `LearnableScalogram`): de CWT van de winnaars als conv-laag, geïnitialiseerd
   als complexe Morlet-wavelets (24 frequenties 0.75-40 Hz, log-spaced) en end-to-end mee getraind. Combineert de sterkste
   vaste representatie (CWT, 1e plaats) met de leerbare front-end (VIPEEGNet). Job 10.
3. **Dirichlet-prior label smoothing** (`label_prior: global`): de +0.02 per stem van Team Sony is een Dirichlet-posterior met
   uniforme prior; met de globale stemverdeling als prior krimpen labels met weinig stemmen naar wat annotatoren gemiddeld
   zeggen i.p.v. naar uniform. Job 10.
4. **Gecross-fitte kalibratie** (`hmsbest/calibrate.py`): temperatuur + bias per klasse, gefit op de andere folds en
   toegepast op de uitgehouden fold. Motivatie: getraind op sterk gesmoothte labels met weinig stemmen, geëvalueerd op
   scherpe labels met >=10 stemmen; KL straft verkeerde zekerheid zwaar. Dezelfde reden waarom entmax (1e plaats) hielp,
   maar eerlijk gemeten en goedkoper.
5. **Teacher-verfijnde labels** (nog niet gebouwd): voor vensters met < 10 stemmen het doel vervangen door een mix van
   stemmen en de OOF-voorspelling van een eerder model (Bayesiaans: de voorspelling als prior). Team Sony wilde pseudo-labeling
   doen maar had geen tijd.
6. Grotere/andere backbones per tak (EfficientNetV2-B3 zoals VIPEEGNet) en langer schema, pas na de 5-fold-baseline.

## Eigen ideeën, ronde 2 (12-11, gebouwd en getest in `tests/`, Kaggle-resultaten in RESULTS.md)

7. **Hemisfeer-symmetrisch netwerk** (`siamese: true` op een beeldtak). Vier van de zes klassen verschillen alleen in
   *gelateraliseerd* (LPD, LRDA) versus *gegeneraliseerd* (GPD, GRDA). Elk beeld bestaat uit 4 banden LL, LP, RP, RL; de
   linkerhelft [LL, LP] en de gespiegelde rechterhelft [RL, RP] gaan door dezelfde backbone, en de kop krijgt
   [fL + fR, |fL - fR|]. Gevolg: exact spiegelinvariant (test `test_siamese_is_mirror_invariant…`), een expliciet
   asymmetriekenmerk, en elke hemisfeer is een extra trainingsvoorbeeld voor dezelfde gewichten. De top-oplossingen
   gebruikten spiegelen alleen als augmentatie/TTA; hier zit het in de architectuur.
8. **Annotator-conditionering** (`vote_cond: true` op de fusie). Rijen met 3 stemmen (helft van de data) en rijen met
   10-20 stemmen zijn verschillende populaties: weinig stemmen = duidelijke, bijna one-hot gevallen; veel stemmen =
   moeilijke, gemengde gevallen, en de test bevat alleen de tweede soort. Iedereen lost dat op met een 2e trainingsstap op
   veel-stem-rijen (en gooit zo de helft van de data weg in de laatste epochs). Hier moduleert log(n) de gefuseerde
   features (FiLM, nul-geïnitialiseerd); bij inferentie wordt n vastgezet op 15: "geef het antwoord dat een groot panel
   zou geven". Het echte n wordt bij validatie niet gebruikt (test `…vote_cond`), dus geen lek.
9. **Teacher-verfijnde labels** (`hmsbest/teacher.py`, idee 5 uitgewerkt). Stemmen ~ Multinomiaal(n, q), teacher-
   voorspelling p als Dirichlet-prior: y' = (n·y + τ·p)/(n + τ). τ = "aantal pseudo-annotators dat de teacher waard is",
   niet met de hand gekozen maar de maximum-likelihood-concentratie van het Dirichlet-multinomiaal op stemmen die de
   teacher niet zag (`estimate_tau`). Rijen met 3 stemmen schuiven sterk, rijen met 15 nauwelijks. Daardoor kan stap 2 op
   ALLE vensters trainen i.p.v. alleen >= 6 stemmen.
   Lekregels: (1) teacher-voorspelling altijd out-of-fold (anders echoot ze het ruislabel); (2) voor een eerlijke
   validatie van student f mag de teacher fold f ook niet gezien hebben -> `exclude_folds`, teacher T(f,g) traint zonder
   f en g en voorspelt g. Job 11 test dit lekvrij op fold 0 (4 teachers op 60 % van de data).
   Voor een volledige 5-fold: 10 paar-teachers T{f,g} (elk bruikbaar voor student f én g).

Nog niet gebouwd, genoteerd:
- 10-minuten-context: de Kaggle-spectrogramtak ziet 10 min, de EEG-takken 50 s. Als vierde tak in de fusie (lage resolutie)
  geeft dat informatie over de evolutie (aanval vs. stationair patroon) die geen EEG-tak heeft.
- Consistentie tussen overlappende vensters van dezelfde eeg_id (zelfde stemmen, 2 s verschoven) als regularisatie.
