# Job 02: full cache + first real fold (spec_b0, fold 0) to measure speed and a first CV number.
set -e
export PYTHONPATH=/kaggle/working/repo
D=$(dirname "$(find /kaggle/input -maxdepth 4 -name train.csv | head -1)")
echo "DATA_ROOT=$D"; nvidia-smi -L
time python -m hmsbest.prep --data_root $D --cache_dir /kaggle/working/cache --workers 4 | tail -3
df -h /kaggle/working | tail -1
time python -m hmsbest.train --config configs/spec_b0.yaml --set name=spec_b0_f0 data_root=$D 'run_folds=[0]' \
  'stages=[{epochs: 6, lr: 1.0e-3, min_votes: 0, weight: votes}, {epochs: 3, lr: 1.0e-4, min_votes: 6, weight: none}]' 2>&1 | grep -v Warning
