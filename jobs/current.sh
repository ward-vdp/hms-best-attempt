# Job 04 (background commit run, ~1h): cache -> /kaggle/temp (not saved as output), then two parallel ablation rounds on fold 0.
export PYTHONPATH=/kaggle/working/repo
D=$(dirname "$(find /kaggle/input -maxdepth 4 -name train.csv | head -1)")
C=/kaggle/temp/cache
mkdir -p /kaggle/working/logs
python -m hmsbest.prep --data_root $D --cache_dir $C --workers 4 2>&1 | grep -v -i warn | tail -n 2
ST='stages=[{epochs: 6, lr: 1.0e-3, min_votes: 0, weight: votes}, {epochs: 3, lr: 1.0e-4, min_votes: 6, weight: none}]'
run() { # name gpu model-override
  CUDA_VISIBLE_DEVICES=$2 python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=$1 data_root=$D cache_dir=$C 'run_folds=[0]' num_workers=2 "$3" "$ST" > /kaggle/working/logs/$1.txt 2>&1 &
}
progress() { while pgrep -f hmsbest.train > /dev/null; do sleep 180; echo "--- $(date +%H:%M)"; for f in /kaggle/working/logs/*.txt; do echo "$(basename $f) $(grep -h '^\[f0' $f | tail -n 1)"; done; done; }
# round 1: input ablation
run eegspec_b0_f0 0 model.kind=eegspec
run eeg1d_f0 1 'model={kind: eeg1d}'
progress
# round 2: late fusion of Kaggle spectrogram + raw EEG CNN, and eeg-spectrogram + raw EEG CNN
run fusion_spec_eeg1d_f0 0 'model={kind: fusion, parts: [{kind: spec, backbone: efficientnet_b0, pretrained: true}, {kind: eeg1d}]}'
run fusion_eegspec_eeg1d_f0 1 'model={kind: fusion, parts: [{kind: eegspec, backbone: efficientnet_b0, pretrained: true}, {kind: eeg1d}]}'
progress
cat /kaggle/working/results.csv | cut -c1-120
for f in /kaggle/working/logs/*.txt; do echo "== $f"; grep -h "^\[f0\|^{" $f | tail -n 4; done
rm -rf /kaggle/working/runs/*/fold*.pt   # keep the output small; OOF csv + logs are what matters here
