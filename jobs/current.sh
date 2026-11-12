# Job 09 (background, ~1h45): 5-fold CV van de beste fusie (eigen-spec B0 + EEG-CNN + eegimg B0), folds verdeeld over 2 GPU's.
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
run5() { # name gpu folds
  local name=$1 gpu=$2 folds=$3
  CUDA_VISIBLE_DEVICES=$gpu python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=$name data_root=$D cache_dir=$C "run_folds=$folds" num_workers=2 "$ST" "$M" eeg_hi=40.0 $T > /kaggle/working/logs/$name.txt 2>&1 &
}
run5 f3p_a 0 '[0,2,4]'
run5 f3p_b 1 '[1,3]'
progress
cut -c1-130 /kaggle/working/results.csv
python - <<'PY'
import pandas as pd
from hmsbest.constants import CLASSES, TARGET_COLS
from hmsbest.metrics import kl_divergence
o = pd.concat([pd.read_csv(f"/kaggle/working/runs/{n}/oof.csv") for n in ("f3p_a", "f3p_b")], ignore_index=True)
o.to_csv("/kaggle/working/oof_f3p.csv", index=False)
pc = [f"pred_{c}" for c in CLASSES]
for name, m in (("all", o.n_votes >= 0), (">=6", o.n_votes >= 6), (">=10", o.n_votes >= 10)):
    print("5-fold OOF", name, len(o[m]), round(kl_divergence(o.loc[m, TARGET_COLS], o.loc[m, pc]), 4))
print(o.groupby("fold").apply(lambda g: round(kl_divergence(g.loc[g.n_votes >= 6, TARGET_COLS], g.loc[g.n_votes >= 6, pc]), 4)))
PY
for f in /kaggle/working/logs/*.txt; do echo "== $f"; tail -n 4 $f | cut -c1-200; done
rm -rf /kaggle/working/runs/*/fold*.pt
