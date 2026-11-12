# Job 14 (background, ~4h30): lekvrije 5-fold met teacher-verfijnde labels.
# Ronde 1: 10 paar-teachers T{S,G} (getraind zonder folds S en G), elk schrijft out-of-fold-voorspellingen voor fold G
#          (bruikbaar voor student S) en fold S (bruikbaar voor student G). Ronde 2: 5 studenten, teacher-labels in stap 1.
export PYTHONPATH=/kaggle/working/repo
D=$(dirname "$(find /kaggle/input -maxdepth 4 -name train.csv | head -1)")
C=/kaggle/temp/cache
mkdir -p /kaggle/working/logs
python -m hmsbest.prep --data_root $D --cache_dir $C --workers 4 2>&1 | grep -v -i warn | tail -n 2
ST='stages=[{epochs: 6, lr: 1.0e-3, min_votes: 0, weight: votes}, {epochs: 3, lr: 1.0e-4, min_votes: 6, weight: none}]'
run() { # name gpu overrides...
  local name=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=$name data_root=$D cache_dir=$C 'run_folds=[0]' num_workers=2 "$ST" "$@" > /kaggle/working/logs/$name.txt 2>&1 &
}
progress() { sleep 45; while pgrep -f hmsbest.train > /dev/null; do sleep 180; echo "--- $(date +%H:%M)"; for f in /kaggle/working/logs/*.txt; do echo "$(basename $f) $(grep -h '^\[f0' $f | tail -n 1)"; done; done; }
T='label_smooth=0.02 shift_s=5.0 target_mean_w=0.5 label_prior=global'
SP='{kind: eegspec, backbone: efficientnet_b0, pretrained: true, mask: 2}'
IM='{kind: eegimg, backbone: efficientnet_b0, pretrained: true}'
M="model={kind: fusion, parts: [$SP, {kind: eeg1d}, $IM]}"
R=/kaggle/working/runs
pair() { # gpu excluded_fold folds
  CUDA_VISIBLE_DEVICES=$1 python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=pair_e$2 data_root=$D cache_dir=$C "run_folds=$3" "exclude_folds=[$2]" save_row_preds=true num_workers=2 "$ST" "$M" eeg_hi=40.0 $T > /kaggle/working/logs/pair_e$2.txt 2>&1
}
( pair 0 0 '[1,2,3,4]'; pair 0 3 '[4]' ) &
( pair 1 1 '[2,3,4]'; pair 1 2 '[3,4]' ) &
progress; wait
ls $R/pair_*/rows_for*.csv | wc -l
stud() { # gpu fold
  CUDA_VISIBLE_DEVICES=$1 python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=stud_f$2 data_root=$D cache_dir=$C "run_folds=[$2]" num_workers=2 "$M" eeg_hi=40.0 $T "teacher_csv=$R/pair_*/rows_for$2_*.csv" 'stages=[{epochs: 6, lr: 1.0e-3, min_votes: 0, weight: votes}, {epochs: 3, lr: 1.0e-4, min_votes: 6, weight: none, teacher_tau: 0}]' > /kaggle/working/logs/stud_f$2.txt 2>&1
}
( stud 0 0; stud 0 2; stud 0 4 ) &
( stud 1 1; stud 1 3 ) &
progress; wait
cut -c1-130 /kaggle/working/results.csv
grep -h "teacher:" /kaggle/working/logs/stud_f*.txt
python - <<'PY'
import pandas as pd
pd.concat([pd.read_csv(f"/kaggle/working/runs/stud_f{f}/oof.csv") for f in range(5)], ignore_index=True).to_csv("/kaggle/working/oof_teacher5.csv", index=False)
PY
python -m hmsbest.analyze /kaggle/working/oof_teacher5.csv
for f in /kaggle/working/logs/stud_f*.txt; do echo "== $f"; tail -n 3 $f | cut -c1-200; done
rm -rf /kaggle/working/runs/*/fold*.pt
