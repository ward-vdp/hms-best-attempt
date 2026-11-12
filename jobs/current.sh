# Job 11 (background, ~1h50): teacher-refined labels, leak-free test on fold 0.
# Round 1: 4 teachers that never see fold 0 (exclude_folds=[0]); teacher g is trained on the 3 folds != 0,g and writes
#          out-of-fold predictions for ALL windows of fold g. Round 2: fold-0 students on labels refined with those teachers.
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
T='label_smooth=0.02 shift_s=5.0 target_mean_w=0.5'
SP='{kind: eegspec, backbone: efficientnet_b0, pretrained: true, mask: 2}'
IM='{kind: eegimg, backbone: efficientnet_b0, pretrained: true}'
M="model={kind: fusion, parts: [$SP, {kind: eeg1d}, $IM]}"
teach() { # name gpu folds
  CUDA_VISIBLE_DEVICES=$2 python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=$1 data_root=$D cache_dir=$C "run_folds=$3" 'exclude_folds=[0]' save_row_preds=true num_workers=2 "$ST" "$M" eeg_hi=40.0 $T > /kaggle/working/logs/$1.txt 2>&1 &
}
teach teach_a 0 '[1,2]'
teach teach_b 1 '[3,4]'
progress
ls -la /kaggle/working/runs/teach_*/rows_fold*.csv
TC='teacher_csv=/kaggle/working/runs/teach_*/rows_fold*.csv'
# A: refined labels in both stages, stage 2 on ALL windows (refined) instead of only >=6 votes
run stud_both 0 "$M" eeg_hi=40.0 $T "$TC" 'stages=[{epochs: 6, lr: 1.0e-3, min_votes: 0, weight: votes}, {epochs: 3, lr: 1.0e-4, min_votes: 0, weight: votes}]'
# B: refined labels only in stage 1, stage 2 as baseline (raw labels, >=6 votes)
run stud_s1 1 "$M" eeg_hi=40.0 $T "$TC" 'stages=[{epochs: 6, lr: 1.0e-3, min_votes: 0, weight: votes}, {epochs: 3, lr: 1.0e-4, min_votes: 6, weight: none, teacher_tau: 0}]'
progress
cut -c1-130 /kaggle/working/results.csv
grep -h "teacher:" /kaggle/working/logs/stud_*.txt
for n in stud_both stud_s1; do python -m hmsbest.analyze /kaggle/working/runs/$n/oof.csv; done
for f in /kaggle/working/logs/*.txt; do echo "== $f"; tail -n 4 $f | cut -c1-200; done
mkdir -p /kaggle/working/keep && for d in /kaggle/working/runs/*/; do cp $d/oof.csv /kaggle/working/keep/$(basename $d)_oof.csv 2>/dev/null; done
rm -rf /kaggle/working/runs/*/fold*.pt
