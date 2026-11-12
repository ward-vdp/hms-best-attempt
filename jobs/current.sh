# Job 05 (background, ~1h): regularisation ablation on fold 0 (SpecAugment-style masking, mixup) and a bigger backbone.
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
progress() { while pgrep -f hmsbest.train > /dev/null; do sleep 180; echo "--- $(date +%H:%M)"; for f in /kaggle/working/logs/*.txt; do echo "$(basename $f) $(grep -h '^\[f0' $f | tail -n 1)"; done; done; }
run eegspec_mask 0 model.kind=eegspec model.mask=2
run eegspec_mask_mixup 1 model.kind=eegspec model.mask=2 aug.mixup=0.4
progress
run fusion_eegspec_eeg1d_mask 0 'model={kind: fusion, parts: [{kind: eegspec, backbone: efficientnet_b0, pretrained: true, mask: 2}, {kind: eeg1d}]}'
run eegspec_b2_mask 1 model.kind=eegspec model.backbone=efficientnet_b2 model.mask=2
progress
cut -c1-130 /kaggle/working/results.csv
for f in /kaggle/working/logs/*.txt; do echo "== $f"; grep -h "^\[f0\|^{" $f | tail -n 4; done
rm -rf /kaggle/working/runs/*/fold*.pt
