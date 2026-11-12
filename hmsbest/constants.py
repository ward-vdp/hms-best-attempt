"""Shared constants: classes, EEG montage and mirror symmetry."""

CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
VOTE_COLS = [f"{c}_vote" for c in CLASSES]
TARGET_COLS = [f"p_{c}" for c in CLASSES]

EEG_CHANNELS = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
                "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]
FS = 200            # EEG sampling rate (Hz)
EEG_LEN = 50 * FS   # a labelled EEG window is 50 s; the label covers the central 10 s

# Double-banana bipolar montage, 4 chains x 4 pairs, ordered LL, LP, RP, RL.
# Region order is kept identical for the EEG channels and for the Kaggle spectrograms
# so that left/right mirroring is the same permutation for both inputs.
REGIONS = ["LL", "LP", "RP", "RL"]
BIPOLAR = [
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),   # LL
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),   # LP
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),   # RP
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),   # RL
]
# Mirroring the head (left <-> right) keeps the class label: LL<->RL, LP<->RP.
MIRROR_CHANNELS = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
MIRROR_REGIONS = [3, 2, 1, 0]
