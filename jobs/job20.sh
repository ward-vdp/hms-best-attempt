# Job 20 (separate notebook "CLAUDE hms-best-attempt train 2", ~3 h): D' = model D (4-branch fusion incl. 10-min Kaggle
# spectrogram) with the broken eegimg branch replaced by the fixed polyphase eegimg2. Same folds/teacher labels/schedule as D,
# so D' vs D is a clean ablation of the fix. Weights kept in runs/Dp.
export PYTHONPATH=/kaggle/working/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$(dirname "$(find /kaggle/input -maxdepth 4 -name train.csv | head -1)")
C=/kaggle/temp/cache
mkdir -p /kaggle/working/logs /kaggle/working/runs /kaggle/working/keep
IN=$(dirname "$(find /kaggle/input -name oof_teacher5.csv | head -1)")
RP=$IN/runs
echo "teacher input: $IN ($(ls $RP/pair_*/rows_for*.csv 2>/dev/null | wc -l) row files)"
[ "$(ls $RP/pair_*/rows_for*.csv 2>/dev/null | wc -l)" = "20" ] || { echo "FOUT: paar-teacher-bestanden niet gevonden"; exit 1; }
python -m hmsbest.prep --data_root $D --cache_dir $C --workers 4 2>&1 | grep -v -i warn | tail -n 2
T='label_smooth=0.02 shift_s=5.0 target_mean_w=0.5 label_prior=global batch_size=32'
SP='{kind: eegspec, backbone: efficientnet_b0, pretrained: true, mask: 2}'
IM='{kind: eegimg2, backbone: efficientnet_b0, pretrained: true}'
KS='{kind: spec, backbone: efficientnet_b0, pretrained: true, mask: 2}'
MD="model={kind: fusion, parts: [$SP, {kind: eeg1d}, $IM, $KS]}"
STT='stages=[{epochs: 6, lr: 1.0e-3, min_votes: 0, weight: votes}, {epochs: 3, lr: 1.0e-4, min_votes: 6, weight: none, teacher_tau: 0}]'
progress() { sleep 45; while pgrep -f hmsbest.train > /dev/null; do sleep 300; echo "--- $(date +%H:%M)"; for f in /kaggle/working/logs/*.txt; do echo "$(basename $f) $(grep -h '^\[f' $f | tail -n 1)"; done; done; }
stud() { CUDA_VISIBLE_DEVICES=$1 python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=Dp_f$2 data_root=$D cache_dir=$C "run_folds=[$2]" num_workers=2 "$MD" eeg_hi=40.0 $T "teacher_csv=$RP/pair_*/rows_for$2_*.csv" "$STT" > /kaggle/working/logs/Dp_f$2.txt 2>&1; }
( stud 0 0; stud 0 2; stud 0 4 ) &
( stud 1 1; stud 1 3 ) &
progress; wait
N=$(ls /kaggle/working/runs/Dp_f*/oof.csv 2>/dev/null | wc -l)
[ "$N" = "5" ] || { echo "FOUT: slechts $N van 5 folds klaar"; for f in /kaggle/working/logs/Dp_f*.txt; do echo "== $f"; tail -n 4 $f | cut -c1-300; done; exit 1; }
cut -c1-130 /kaggle/working/results.csv
python - <<'PY'
import pandas as pd
pd.concat([pd.read_csv(f"/kaggle/working/runs/Dp_f{f}/oof.csv") for f in range(5)], ignore_index=True).to_csv("/kaggle/working/keep/oof_Dp.csv", index=False)
PY
mkdir -p /kaggle/working/runs/Dp; for f in 0 1 2 3 4; do cp /kaggle/working/runs/Dp_f$f/fold$f.pt /kaggle/working/runs/Dp/; done
cp /kaggle/working/runs/Dp_f0/config.yaml /kaggle/working/runs/Dp/config.yaml
rm -rf /kaggle/working/runs/Dp_f*/fold*.pt
python -m hmsbest.analyze /kaggle/working/keep/oof_Dp.csv
for f in /kaggle/working/logs/*.txt; do echo "== $f"; tail -n 2 $f | cut -c1-200; done
du -sh /kaggle/working/*
