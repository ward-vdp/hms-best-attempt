# Job 15 (background, ~3h50): twee extra modellen voor het ensemble, 5-fold, met de paar-teacher-voorspellingen van job 14
# (output van Version 12, als input aan het notebook gekoppeld). B = fusie met EfficientNet-B2-takken; A2 = B0-fusie, andere seed.
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
IN=$(dirname "$(find /kaggle/input -name oof_teacher5.csv | head -1)")
RP=$IN/runs
echo "teacher input: $IN  ($(ls $RP/pair_*/rows_for*.csv 2>/dev/null | wc -l) row files)"
[ "$(ls $RP/pair_*/rows_for*.csv 2>/dev/null | wc -l)" = "20" ] || { echo "FOUT: paar-teacher-bestanden niet gevonden"; exit 1; }
T='label_smooth=0.02 shift_s=5.0 target_mean_w=0.5 label_prior=global'
SP='{kind: eegspec, backbone: efficientnet_b0, pretrained: true, mask: 2}'
IM='{kind: eegimg, backbone: efficientnet_b0, pretrained: true}'
SP2='{kind: eegspec, backbone: efficientnet_b2, pretrained: true, mask: 2}'
IM2='{kind: eegimg, backbone: efficientnet_b2, pretrained: true}'
MA="model={kind: fusion, parts: [$SP, {kind: eeg1d}, $IM]}"
MB="model={kind: fusion, parts: [$SP2, {kind: eeg1d}, $IM2]}"
STT='stages=[{epochs: 6, lr: 1.0e-3, min_votes: 0, weight: votes}, {epochs: 3, lr: 1.0e-4, min_votes: 6, weight: none, teacher_tau: 0}]'
R=/kaggle/working/runs
stud() { # gpu name fold model extra...
  local gpu=$1 name=$2 f=$3 m=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=${name}_f$f data_root=$D cache_dir=$C "run_folds=[$f]" num_workers=2 "$m" eeg_hi=40.0 $T "teacher_csv=$RP/pair_*/rows_for${f}_*.csv" "$STT" "$@" > /kaggle/working/logs/${name}_f$f.txt 2>&1
}
( stud 0 B 0 "$MB" batch_size=32; stud 0 B 2 "$MB" batch_size=32; stud 0 B 4 "$MB" batch_size=32; stud 0 A2 1 "$MA" seed=7 fold_seed=42; stud 0 A2 3 "$MA" seed=7 fold_seed=42 ) &
( stud 1 B 1 "$MB" batch_size=32; stud 1 B 3 "$MB" batch_size=32; stud 1 A2 0 "$MA" seed=7 fold_seed=42; stud 1 A2 2 "$MA" seed=7 fold_seed=42; stud 1 A2 4 "$MA" seed=7 fold_seed=42 ) &
progress; wait
cut -c1-130 /kaggle/working/results.csv
K=/kaggle/working/keep; mkdir -p $K
python - <<'PY'
import pandas as pd
for n in ("B", "A2"):
    pd.concat([pd.read_csv(f"/kaggle/working/runs/{n}_f{f}/oof.csv") for f in range(5)], ignore_index=True).to_csv(f"/kaggle/working/keep/oof_{n}.csv", index=False)
PY
cp $IN/oof_teacher5.csv $K/oof_A.csv
python -m hmsbest.blend $K/oof_A.csv $K/oof_A2.csv $K/oof_B.csv --out $K/oof_blend.csv
python -m hmsbest.analyze $K/oof_blend.csv
# carry the teacher predictions forward so later versions can keep using this notebook's output as input
cp $IN/oof_teacher5.csv /kaggle/working/oof_teacher5.csv
for d in $RP/pair_*/; do mkdir -p $R/$(basename $d); cp $d/rows_for*.csv $R/$(basename $d)/; done
for f in /kaggle/working/logs/*.txt; do echo "== $f"; tail -n 2 $f | cut -c1-200; done
rm -rf /kaggle/working/runs/*/fold*.pt
