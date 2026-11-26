# Job 19 (background, ~5-7 h): model E = the raw-EEG image model of the 2nd-place HMS solution, done right this time:
# polyphase 160x1000 image (kind eegimg2), 0.5-20 Hz, clip +-1024 uV (scaled /100), EfficientNet-B5, head dropout 0.5,
# 15 epochs lr 1e-3 (vote-weighted) + 5 epochs lr 1e-4 on >= 6 votes, labels averaged per eeg_id (target_mean_w 1).
# 2nd place reported this single model at public 0.231 / private 0.283. Our additions: leak-free teacher labels in
# stage 1, +-5 s shift, mirror/polarity aug. 5-fold on the same folds as A/A2/B/D (seed 42), fold weights kept in runs/E.
# FA/FB and the teacher files are copied forward, because the submit notebook reads this notebook's LATEST version.
export PYTHONPATH=/kaggle/working/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$(dirname "$(find /kaggle/input -maxdepth 4 -name train.csv | head -1)")
C=/kaggle/temp/cache
mkdir -p /kaggle/working/logs /kaggle/working/runs /kaggle/working/keep
# 1. carry forward what later notebooks depend on
FA=$(dirname "$(find /kaggle/input -path '*runs/FA/config.yaml' | head -1)")
FB=$(dirname "$(find /kaggle/input -path '*runs/FB/config.yaml' | head -1)")
cp -r $FA /kaggle/working/runs/FA; cp -r $FB /kaggle/working/runs/FB
IN=$(dirname "$(find /kaggle/input -name oof_teacher5.csv | head -1)")
RP=$IN/runs
cp $IN/oof_teacher5.csv /kaggle/working/
for d in $RP/pair_*/; do mkdir -p /kaggle/working/runs/$(basename $d); cp $d/rows_for*.csv /kaggle/working/runs/$(basename $d)/; done
echo "FA $(ls /kaggle/working/runs/FA/fold*.pt | wc -l)  FB $(ls /kaggle/working/runs/FB/fold*.pt | wc -l)  teacher files $(ls $RP/pair_*/rows_for*.csv | wc -l)"
[ "$(ls /kaggle/working/runs/FA/fold*.pt /kaggle/working/runs/FB/fold*.pt | wc -l)" = "4" ] || { echo "FOUT: FA/FB niet gevonden"; exit 1; }
[ "$(ls $RP/pair_*/rows_for*.csv 2>/dev/null | wc -l)" = "20" ] || { echo "FOUT: paar-teacher-bestanden niet gevonden"; exit 1; }
python -m hmsbest.prep --data_root $D --cache_dir $C --workers 4 2>&1 | grep -v -i warn | tail -n 2
# 2. model E, 5 folds over 2 GPUs
ME='model={kind: eegimg2, backbone: efficientnet_b5, pretrained: true, drop: 0.5, drop_path: 0.1}'
T='label_smooth=0.02 shift_s=5.0 target_mean_w=1.0 label_prior=global eeg_hi=20.0 eeg_clip=1024.0 eeg_scale=100.0 batch_size=16'
STT='stages=[{epochs: 15, lr: 1.0e-3, min_votes: 0, weight: votes}, {epochs: 5, lr: 1.0e-4, min_votes: 6, weight: none, teacher_tau: 0}]'
progress() { sleep 45; while pgrep -f hmsbest.train > /dev/null; do sleep 300; echo "--- $(date +%H:%M)"; for f in /kaggle/working/logs/*.txt; do echo "$(basename $f) $(grep -h '^\[f' $f | tail -n 1)"; done; done; }
stud() { # gpu fold
  CUDA_VISIBLE_DEVICES=$1 python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=E_f$2 data_root=$D cache_dir=$C "run_folds=[$2]" num_workers=2 "$ME" $T "teacher_csv=$RP/pair_*/rows_for$2_*.csv" "$STT" > /kaggle/working/logs/E_f$2.txt 2>&1
}
( stud 0 0; stud 0 2; stud 0 4 ) &
( stud 1 1; stud 1 3 ) &
progress; wait
N=$(ls /kaggle/working/runs/E_f*/oof.csv 2>/dev/null | wc -l)
[ "$N" = "5" ] || { echo "FOUT: slechts $N van 5 folds klaar"; for f in /kaggle/working/logs/E_f*.txt; do echo "== $f"; tail -n 4 $f | cut -c1-300; done; exit 1; }
cut -c1-130 /kaggle/working/results.csv
python - <<'PY'
import pandas as pd
pd.concat([pd.read_csv(f"/kaggle/working/runs/E_f{f}/oof.csv") for f in range(5)], ignore_index=True).to_csv("/kaggle/working/keep/oof_E.csv", index=False)
PY
mkdir -p /kaggle/working/runs/E; for f in 0 1 2 3 4; do cp /kaggle/working/runs/E_f$f/fold$f.pt /kaggle/working/runs/E/; done
cp /kaggle/working/runs/E_f0/config.yaml /kaggle/working/runs/E/config.yaml
rm -rf /kaggle/working/runs/E_f*/fold*.pt
python -m hmsbest.blend /kaggle/working/oof_teacher5.csv /kaggle/working/keep/oof_E.csv --out /kaggle/working/keep/oof_blend_AE.csv
python -m hmsbest.analyze /kaggle/working/keep/oof_E.csv
for f in /kaggle/working/logs/*.txt; do echo "== $f"; tail -n 2 $f | cut -c1-200; done
python -m hmsbest.infer --data_root $D --exp /kaggle/working/runs/E:1 --check 200
du -sh /kaggle/working/*
