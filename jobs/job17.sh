# Job 17 (background, ~2h15): 4th fusion branch = the Kaggle 10-minute spectrogram, i.e. information the 50 s EEG branches
# do not see (how the pattern evolves). Model D = fusion [eegspec B0, eeg1d, eegimg B0, spec B0 (10 min)], 5-fold, teacher
# labels in stage 1 (pair teachers from the notebook-output input), same folds as A/A2/B (seed 42). The fold*.pt weights are
# KEPT, so the 5 models can go straight into a submission. Then blend with whichever oof_A/A2/B files the mounted input has.
export PYTHONPATH=/kaggle/working/repo
D=$(dirname "$(find /kaggle/input -maxdepth 4 -name train.csv | head -1)")
C=/kaggle/temp/cache
mkdir -p /kaggle/working/logs
python -m hmsbest.prep --data_root $D --cache_dir $C --workers 4 2>&1 | grep -v -i warn | tail -n 2
IN=$(dirname "$(find /kaggle/input -name oof_teacher5.csv | head -1)")
RP=$IN/runs
echo "teacher input: $IN  ($(ls $RP/pair_*/rows_for*.csv 2>/dev/null | wc -l) row files)"
find /kaggle/input -name 'oof_*.csv' | head
[ "$(ls $RP/pair_*/rows_for*.csv 2>/dev/null | wc -l)" = "20" ] || { echo "FOUT: paar-teacher-bestanden niet gevonden"; exit 1; }
T='label_smooth=0.02 shift_s=5.0 target_mean_w=0.5 label_prior=global'
SP='{kind: eegspec, backbone: efficientnet_b0, pretrained: true, mask: 2}'
IM='{kind: eegimg, backbone: efficientnet_b0, pretrained: true}'
KS='{kind: spec, backbone: efficientnet_b0, pretrained: true, mask: 2}'
MD="model={kind: fusion, parts: [$SP, {kind: eeg1d}, $IM, $KS]}"
STT='stages=[{epochs: 6, lr: 1.0e-3, min_votes: 0, weight: votes}, {epochs: 3, lr: 1.0e-4, min_votes: 6, weight: none, teacher_tau: 0}]'
progress() { sleep 45; while pgrep -f hmsbest.train > /dev/null; do sleep 180; echo "--- $(date +%H:%M)"; for f in /kaggle/working/logs/*.txt; do echo "$(basename $f) $(grep -h '^\[f' $f | tail -n 1)"; done; done; }
stud() { # gpu name fold model extra...
  local gpu=$1 name=$2 f=$3 m=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=${name}_f$f data_root=$D cache_dir=$C "run_folds=[$f]" num_workers=2 "$m" eeg_hi=40.0 $T "teacher_csv=$RP/pair_*/rows_for${f}_*.csv" "$STT" "$@" > /kaggle/working/logs/${name}_f$f.txt 2>&1
}
( stud 0 D 0 "$MD"; stud 0 D 2 "$MD"; stud 0 D 4 "$MD" ) &
( stud 1 D 1 "$MD"; stud 1 D 3 "$MD" ) &
progress; wait
cut -c1-130 /kaggle/working/results.csv
K=/kaggle/working/keep; mkdir -p $K
python - <<'PY'
import pandas as pd
pd.concat([pd.read_csv(f"/kaggle/working/runs/D_f{f}/oof.csv") for f in range(5)], ignore_index=True).to_csv("/kaggle/working/keep/oof_D.csv", index=False)
PY
# the 5 fold models in ONE directory (infer.py averages every fold*.pt of an experiment directory)
mkdir -p /kaggle/working/runs/D; for f in 0 1 2 3 4; do cp /kaggle/working/runs/D_f$f/fold$f.pt /kaggle/working/runs/D/; done
cp /kaggle/working/runs/D_f0/config.yaml /kaggle/working/runs/D/config.yaml
# blend with whatever is available (A = oof_teacher5.csv, A2/B from the version-13 output if that is what is mounted)
ARGS=""
PA=$(find /kaggle/input -name oof_A.csv | head -1); [ -z "$PA" ] && PA=$(find /kaggle/input -name oof_teacher5.csv | head -1)   # same file (A)
ARGS="$PA"
for n in oof_A2.csv oof_B.csv; do p=$(find /kaggle/input -name $n | head -1); [ -n "$p" ] && ARGS="$ARGS $p"; done
echo "blend inputs:$ARGS $K/oof_D.csv"
python -m hmsbest.blend $ARGS $K/oof_D.csv --out $K/oof_blend.csv
python -m hmsbest.analyze $K/oof_blend.csv
# carry the teacher predictions forward so later versions can keep using this notebook's output as input
cp $IN/oof_teacher5.csv /kaggle/working/oof_teacher5.csv
for d in $RP/pair_*/; do mkdir -p /kaggle/working/runs/$(basename $d); cp $d/rows_for*.csv /kaggle/working/runs/$(basename $d)/; done
for f in /kaggle/working/logs/*.txt; do echo "== $f"; tail -n 2 $f | cut -c1-200; done
rm -rf /kaggle/working/runs/D_f*/fold*.pt
du -sh /kaggle/working/*
