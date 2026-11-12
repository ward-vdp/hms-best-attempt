# Job 03: input ablation on fold 0, two GPUs in parallel (same schedule as spec_b0_f0 for comparability).
export PYTHONPATH=/kaggle/working/repo
D=$(dirname "$(find /kaggle/input -maxdepth 4 -name train.csv | head -1)")
ST='stages=[{epochs: 6, lr: 1.0e-3, min_votes: 0, weight: votes}, {epochs: 3, lr: 1.0e-4, min_votes: 6, weight: none}]'
CUDA_VISIBLE_DEVICES=0 python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=eegspec_b0_f0 data_root=$D 'run_folds=[0]' model.kind=eegspec num_workers=2 "$ST" > /kaggle/working/l_eegspec.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=eeg1d_f0 data_root=$D 'run_folds=[0]' 'model={kind: eeg1d}' num_workers=2 "$ST" > /kaggle/working/l_eeg1d.txt 2>&1 &
while pgrep -f hmsbest.train > /dev/null; do sleep 150; echo "--- $(date +%H:%M)"; grep -h "^\[f0" /kaggle/working/l_eegspec.txt | tail -n 1 | sed 's/^/eegspec /'; grep -h "^\[f0" /kaggle/working/l_eeg1d.txt | tail -n 1 | sed 's/^/eeg1d   /'; done
tail -n 3 /kaggle/working/l_eegspec.txt /kaggle/working/l_eeg1d.txt
