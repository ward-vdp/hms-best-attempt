# Job 12 (background, ~1h): scalogramtak in de fusie, fold 0, batch 32 (geheugen): (a) 4 takken spec+1d+img+cwt, (b) cwt i.p.v. STFT-spec.
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
T='label_smooth=0.02 shift_s=5.0 target_mean_w=0.5 label_prior=global batch_size=32'
SP='{kind: eegspec, backbone: efficientnet_b0, pretrained: true, mask: 2}'
IM='{kind: eegimg, backbone: efficientnet_b0, pretrained: true}'
CW='{kind: eegcwt, backbone: efficientnet_b0, pretrained: true, mask: 2}'
run fusion4_cwt 0 "model={kind: fusion, parts: [$SP, {kind: eeg1d}, $IM, $CW]}" eeg_hi=40.0 $T
run fusion3_cwt_noSpec 1 "model={kind: fusion, parts: [$CW, {kind: eeg1d}, $IM]}" eeg_hi=40.0 $T
progress
cut -c1-130 /kaggle/working/results.csv
for n in fusion4_cwt fusion3_cwt_noSpec; do python -m hmsbest.analyze /kaggle/working/runs/$n/oof.csv; done
for f in /kaggle/working/logs/*.txt; do echo "== $f"; tail -n 5 $f | cut -c1-200; done
mkdir -p /kaggle/working/keep && for d in /kaggle/working/runs/*/; do cp $d/oof.csv /kaggle/working/keep/$(basename $d)_oof.csv 2>/dev/null; done
rm -rf /kaggle/working/runs/*/fold*.pt
