# Job executed by the Kaggle runner notebook. Edited and pushed from the repo; the notebook only does `git pull; bash jobs/current.sh`.
set -e
export PYTHONPATH=/kaggle/working/repo
D=$(dirname "$(find /kaggle/input -maxdepth 4 -name train.csv | head -1)")
echo "DATA_ROOT=$D"
nvidia-smi -L
python -c "import torch,timm;print('torch',torch.__version__,'timm',timm.__version__,'gpus',torch.cuda.device_count())"
ls $D | head
# smoke test on real data: cache the first 300 eeg_ids, train 1 short fold on GPU
time python -m hmsbest.prep --data_root $D --cache_dir /kaggle/working/cache_small --limit 300 --workers 4
time python -m hmsbest.train --config configs/spec_b0.yaml --set name=smoke cache_dir=/kaggle/working/cache_small \
  data_root=$D limit_eegs=300 folds=3 'run_folds=[0]' 'stages=[{epochs: 2, lr: 1.0e-3, min_votes: 0, weight: votes}]' batch_size=32
