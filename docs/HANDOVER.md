# Overdrachtsrapport hms-best-attempt (sandbox, 13-14/11/2026)

Doel van dit rapport: een andere assistent helpt Ward naar een eigen, begrepen oplossing in zijn cursusrepo. Alles hieronder komt uit runs die echt gedraaid zijn (`docs/RESULTS.md`, Kaggle-logs, Kaggle-submissionpagina). **n.g. = niet gemeten.** KL: lager is beter. "≥10" = per eeg_id het venster met de meeste stemmen, enkel eeg_id's met ≥10 stemmen (stuurmaat); "≥6" idem met ≥6; "all" = alle eeg_id's.

## 1. Eindresultaat
- **Model:** 4 netwerken op alle trainingsdata (full-fit), 50/50 gemiddeld: late fusie van drie takken (eigen STFT-spectrogram + 1D-CNN/BiGRU + raw-EEG-beeld) in EfficientNet-B2 (2 seeds) en B0 (2 seeds), mirror-TTA, teacher-verfijnde labels in stap 1, geen kalibratie.
- **Late submission:** public **0.273757**, private **0.316473** (≈ plaats 34 op de private ranglijst, zelf afgelezen uit de top-49: 1e 0.2723, 10e 0.2867, 20e 0.3029; telt niet mee).
- **CV-KL:** voor dit full-fit-artefact zelf n.g. (geen uitgehouden data). Gemeten 5-fold-analoog (3 modellen A/A2/B, 80 % data): all/≥6/≥10 = 0.554 / 0.260 / 0.257 (+ kalibratie 0.575 / 0.258 / 0.256).
- **Runtijd/hardware:** training 7 394 s (≈2 h 03) op Kaggle 2×T4 (2 processen parallel); inferentie-notebook 102 s op de zichtbare test; duur op de verborgen test n.g. Het 5-fold-ensemble kostte 13 774 s (job 15), 5-fold + 10 paar-teachers 16 465 s (job 14).
- Twee oudere submissions op het account (0.892/0.792 en 0.775/0.717) zijn niet van mij; niet geanalyseerd.

## 2. Experimentenlog (fold 0, tenzij vermeld; CV = all / ≥6 / ≥10; public/private n.g. behalve de laatste rij)
| id | wat veranderde | CV | verdict |
|---|---|---|---|
| E1 | baseline B0 op Kaggle-spectrogram | .714/.415/– | referentie |
| E2 | alleen ruwe EEG, ResNet1D+BiGRU | .837/.436/– | hielp niet (zwakste tak) |
| E3 | eigen STFT uit EEG i.p.v. Kaggle-spec | .699/.365/– | hielp (≥6) |
| E4/E5 | late fusie spec+1D / eigen-spec+1D | .660/.374 ; .671/.339 | hielp |
| E6 | + masking op E3 | .755/.367/.357 | binnen ruis (≥6), "all" slechter |
| E7 | + mixup 0.4 | .731/.361/.353 | binnen ruis (herstelt deel "all") |
| E8 | fusie met masking | .667/.329/.321 | hielp |
| E9 | B2 i.p.v. B0 (spec, masking) | .699/.339/.332 | hielp (vs .367) |
| E10 | + smoothing 0.02 + ±5 s shift + target-blend (bundel) | .725/.355/.346 | hielp (−.012 ≥6; bundel niet ontleed) |
| E11-13 | raw EEG als beeld (20 Hz / 40 Hz / zonder tricks) | .734/.361/.351 ; .748/.367/.356 ; .736/.361/.351 | hielp niet t.o.v. E10; tricks doen niets voor eegimg |
| E14 | fusie eigen-spec + eegimg | .629/.298/.292 | **hielp (groot)** |
| E15 | + 1D-tak (3 takken) | .602/.299/.293 | hielp "all", ≥6 gelijk |
| E16 | 5-fold van E15 (OOF) | .565/.278/.275 | referentie, foldspreiding ±.017 |
| E17 | leerbaar Morlet-scalogram als losse tak | .663/.348/.340 | hielp licht (beste losse tak) |
| E18 | smoothing naar globale stemverdeling | .602/.293/.288 | binnen ruis, behouden |
| E19 | annotator-conditionering (FiLM) | .661/.296/.288 | binnen ruis, "all" slechter |
| E20 | hemisfeer-symmetrische takken | .673/.347/.338 | **hielp niet** (+.045) |
| E21/22 | scalogramtak in de fusie / i.p.v. STFT | .610/.290/.283 ; .610/.295/.287 | binnen ruis, 1.7× duurder |
| E23 | teacher-labels stap 1, lekvrij | .598/.284/.277 | **hielp** (−.016 ≥10) |
| E24 | teacher-labels ook in stap 2 | .561/.299/.294 | hielp niet op ≥10 |
| E25 | B2 in de fusie | .611/.288/.280 | grens van ruis (−.008) |
| E26 | 10+4 epochs i.p.v. 6+3 | .611/.297/.291 | hielp niet |
| E27 | 5-fold met teacher-labels (A) | .566/.268/.266 | hielp (5/5 folds, −.009; zie valkuil 4) |
| E28 | A2 (andere seed) / B (B2) 5-fold | .570/.270/.267 ; .560/.265/.263 | B licht beter |
| E29 | gelijk gemiddelde / gewogen blend / + kalibratie | .555/.260/.257 ; .554/.260/.257 ; .575/.258/.256 | hielp (−.006 t.o.v. B); kalibratie ≈ ruis |
| E30 | full-fit eindmodel | CV n.g.; **public .2738, private .3165** | gemeten op Kaggle |

## 3. De 5 grootste hefbomen
1. **Fusie van verschillende representaties.** (a) Spectrogram, raw-EEG-beeld en 1D-signaal in één netwerk met één kop. (b) Elke representatie verbergt andere fouten; de kop leert welke tak wanneer te geloven (diversiteit zoals bij een ensemble, maar gezamenlijk getraind). (c) ≥6: .355 (beste enkele tak toen) → .298; "all" .725 → .602. (d) Gemeten in een ablatie (E10-E15), 1 fold, 1 seed; grootte ver boven de ruis.
2. **Teacher-verfijnde labels.** (a) Labels met weinig stemmen worden gemengd met de voorspelling van een eerder model. (b) 3 annotatoren geven een ruwe schatting van de consensusverdeling; een Bayesiaanse prior (teacher, waarde ≈3 pseudo-annotatoren, ML-geschat) verlaagt de ruis. (c) −.016 (≥10, fold 0, lekvrij); 5-fold −.009. (d) Gemeten, E23/E27; 5-fold-vergelijking is vertroebeld (valkuil 4).
3. **Twee-staps training (stap 2 enkel ≥6 stemmen).** (a) Eerst alle data, dan finetunen op de betrouwbare vensters. (b) De test lijkt op de veel-stem-populatie; laag-stem-rijen hebben andere labels (74-91 % unaniem tegenover 14 %). (c) ≈ −.04 op ≥6, ≈ +.015 op "all". (d) Gemeten in de epoch-logs van dezelfde runs, geen aparte ablatie (n.g.).
4. **Ensemble van modellen.** (a) Gemiddelde van A, A2, B. (b) KL straft uitschieters; middelen vermindert variantie. (c) −.006 t.o.v. het beste enkele model (.263 → .257). (d) Gemeten, 5-fold OOF; seed-ensemble en andere backbone leveren ongeveer hetzelfde.
5. **Label-tricks (smoothing vóór normalisatie, ±5 s shift, target-blend).** (a) Zachtere en robuustere doelen. (b) Weinig-stem-labels zijn bijna one-hot en dus overconfident; smoothing krimpt ze. (c) −.012 op ≥6 (E10). (d) Gemeten als bundel op één tak; welke van de drie het doet n.g.; geen effect op eegimg.
Genoemd maar kleiner: B2 (−.008, grens ruis), globale prior (−.005, ruis), scalogram (−.007 los, ≈0 in de fusie).

## 4. Datakeuzes
- **Venster:** EEG 200 Hz, 50 s met het label in de middelste 10 s; training ±5 s random shift. Kaggle-spectrogram: 10 min, 300 rijen van 2 s, label in het midden; test = offset 0. Het eigen spectrogram: STFT (n_fft 512, venster 128, hop 50, 0.5-40 Hz), gemiddeld per keten.
- **Labels:** per rij (venster) zachte verdeling = stemmen/n. Per epoch één willekeurig venster per eeg_id. Gemengd met het eeg_id-gemiddelde (gewicht 0.5, enkel train). Stap 1 loss gewogen met min(n,20)/20, stap 2 niet gewogen en enkel ≥6 stemmen. Smoothing: 0.02·6 pseudo-stemmen naar een prior vóór normalisatie.
- **Validatie:** 5-fold StratifiedGroupKFold per patiënt (gestratificeerd op dominante klasse), seed 42; per eeg_id één venster (meeste stemmen), EMA-gewichten van de laatste epoch, geen selectie op de validatieset.
- **Preprocessing:** 16 bipolaire kanalen (double banana: LL, LP, RP, RL), per venster zero-phase Butterworth 0.5-40 Hz (eegimg testte 20 Hz), clip ±500 µV, delen door 50; spectrogram log + per-gebied standaardisatie; spiegelen (LL↔RL, LP↔RP) als augmentatie en TTA.

## 5. Valkuilen
1. **float16-cache met ±inf** maakte gefilterde vensters NaN (commit-bericht van de fix). Opgelost door clippen en een test; hoe precies ontdekt n.g.
2. **EMA zonder warm-up** (fix in commit "EMA warm-up": verval groeit van ~0.1 naar 0.998 zodat korte runs niet bij de beginwaarden blijven); wanneer en hoe ontdekt n.g.
3. **Startup-race** in de voortgangslus liet een ronde stil over (Version 4). Gezien omdat resultaten ontbraken; herstart.
4. **Lekkage bij teacher-labels:** een teacher die de rij of de validatiefold zag echoot het label. Opgelost met out-of-fold-voorspellingen en geneste paar-teachers; ontwerpfout vooraf bedacht, getest in `tests/`. Vertroebeling: E16 gebruikte nog de uniforme prior, E27 de globale; winst deels (~.005) van de prior (n.g. exact).
5. **Verkeerde maat:** "KL all" wordt gedomineerd door laag-stem-rijen die niet op de test lijken (foutanalyse op E16: 3-stem-rijen KL 1.01 tegenover ~0.27 bij ≥13 stemmen). ≥6 en ≥10 zijn daarom vanaf job 05 gerapporteerd; ≥10 is de stuurmaat.
6. **Enkele fold als beslissing:** foldspreiding ±.017, dus winsten < ~.008 (E18, E19, E21, E25) zijn ruis; deels toch gebruikt voor keuzes.
7. **CV→private:** CV .256 → private .317 (verschil .06). Mogelijke oorzaken (andere CV-definitie dan de winnaars, overfitting op CV, verschuiving private set) n.g.
8. **Full-fit:** de ingediende modellen hebben geen onafhankelijke score; de check op 400 trainvensters (KL .164) is in-sample en bewijst enkel dat train- en inferentiecode gelijk zijn (test in `tests/`).

## 6. Bronnen (wat overgenomen)
- 1e plaats Team Sony: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/writeups/team-sony-1st-place-solution-team-sony — validatie op ≥10 stemmen, smoothing vóór normalisatie, ±5 s shift, twee stappen, ensemblegewichten, idee scalogram/learnable filters.
- 2e plaats (yuto): https://dev.to/moyuto/kaggle-hms-2nd-solution-explaind-34jj — double banana, raw EEG als beeld, twee stappen, spiegelen.
- arXiv 2412.07878 — masking, mixup, window shifting.
- Cranjis-McB: https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification — vote-gewogen KL, twee stappen.
- Ryan Peters: https://ryanirl.com/blog/hbac — gelezen (1D ResNet+RNN, channel flips); effectief gebruik n.g.
- VIPEEGNet arXiv 2507.08874 — idee leerbare EEG-naar-beeld-front-end (basis voor mijn scalogram).
- EEG-FM-Bench arXiv 2508.17742 — gelezen, niet gebruikt.
- timm (EfficientNet): https://github.com/huggingface/pytorch-image-models.
- Eigen: teacher/Dirichlet-labels, leerbaar Morlet-scalogram, globale-prior smoothing, cross-fitted blend/kalibratie, FiLM-conditionering, siamese takken.

## 7. Advies voor 2 weken (opinie, niet gemeten)
1. **Eén tak + eerlijke validatie + foutanalyse** (patiënt-gegroepeerde folds, KL, maat ≥10, KL per aantal stemmen). Goedkoop en leert de dataset en de metriek (valkuil 5).
2. **Tweede representatie fuseren** (spectrogram + raw EEG): de grootste gemeten hefboom (E10-E15).
3. **Twee-staps training en een 3-seed-ensemble** als ablatie (hefbomen 3 en 4), en teacher-labels enkel als er tijd is (complexer door de lekregels).

## 8. Bestanden
- `hmsbest/constants.py` klassen, montage en spiegelpermutaties. `data.py` soft targets en patiëntfolds. `prep.py` cache van parquet naar float16-npy. `signal.py` montage, filter en vensters (gedeeld door training en inferentie). `specs.py` STFT op GPU en masking. `dataset.py` datasets met labelbouw (smoothing, shift, teacher). `models.py` takken, scalogram, fusie. `train.py` training, EMA, OOF. `teacher.py` teacher-labels en τ-schatting. `calibrate.py` kalibratie. `blend.py` cross-fitted ensemblegewichten. `ensemble.py` oudere blendtool. `analyze.py` foutanalyse. `infer.py` offline inferentie met `--check`. `metrics.py` KL.
- `configs/spec_b0.yaml` basisconfig. `jobs/job11-16.sh`, `current.sh` Kaggle-jobs. `notebooks/runner.ipynb` start een job, `submit.ipynb` offline inferentie. `tests/` 19 tests. `outputs/results.csv` runlog. `docs/PLAN.md`, `IDEAS.md`, `RESULTS.md` plan, bronnen, resultaten. `requirements.txt` afhankelijkheden.
