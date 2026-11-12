# Job 07 (background): ronde 2 van job 06 opnieuw (filter-ablatie en tricks-ablatie op eegimg).
# Job 06 (background, ~1h): raw EEG as image (eegimg) + Team-Sony tricks (label smoothing, time shift, target blending) on fold 0.
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
# round 2: ablate the filter cut-off and the tricks on eegimg
run eegimg_b0_hi40_tricks 0 model.kind=eegimg model.backbone=efficientnet_b0 eeg_hi=40.0 $T
run eegimg_b0_plain 1 model.kind=eegimg model.backbone=efficientnet_b0 eeg_hi=20.0
progress
cut -c1-130 /kaggle/working/results.csv
for f in /kaggle/working/logs/*.txt; do echo "== $f"; tail -n 6 $f | cut -c1-200; done
rm -rf /kaggle/working/runs/*/fold*.pt
