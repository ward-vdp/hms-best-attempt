# Job 16 (background, ~2h): final models for a late submission. Full fit (100 % of the data) of the two architectures of
# the job-15 ensemble, 2 seeds each, with teacher-refined labels in stage 1 (all 20 pair-teacher row files: every row is
# predicted by 4 teachers that never saw its patient). Weights are KEPT in /kaggle/working/runs/{FA,FB}/fold*.pt.
export PYTHONPATH=/kaggle/working/repo
D=$(dirname "$(find /kaggle/input -maxdepth 4 -name train.csv | head -1)")
C=/kaggle/temp/cache
mkdir -p /kaggle/working/logs
python -m hmsbest.prep --data_root $D --cache_dir $C --workers 4 2>&1 | grep -v -i warn | tail -n 2
IN=$(dirname "$(find /kaggle/input -name oof_teacher5.csv | head -1)")
RP=$IN/runs
echo "teacher input: $IN  ($(ls $RP/pair_*/rows_for*.csv 2>/dev/null | wc -l) row files)"
[ "$(ls $RP/pair_*/rows_for*.csv 2>/dev/null | wc -l)" = "20" ] || { echo "FOUT: paar-teacher-bestanden niet gevonden"; exit 1; }
T='label_smooth=0.02 shift_s=5.0 target_mean_w=0.5 label_prior=global full_fit=true'
SP='{kind: eegspec, backbone: efficientnet_b0, pretrained: true, mask: 2}'
IM='{kind: eegimg, backbone: efficientnet_b0, pretrained: true}'
SP2='{kind: eegspec, backbone: efficientnet_b2, pretrained: true, mask: 2}'
IM2='{kind: eegimg, backbone: efficientnet_b2, pretrained: true}'
MA="model={kind: fusion, parts: [$SP, {kind: eeg1d}, $IM]}"
MB="model={kind: fusion, parts: [$SP2, {kind: eeg1d}, $IM2]}"
STT='stages=[{epochs: 6, lr: 1.0e-3, min_votes: 0, weight: votes}, {epochs: 3, lr: 1.0e-4, min_votes: 6, weight: none, teacher_tau: 0}]'
progress() { sleep 45; while pgrep -f hmsbest.train > /dev/null; do sleep 180; echo "--- $(date +%H:%M)"; for f in /kaggle/working/logs/*.txt; do echo "$(basename $f) $(grep -h '^\[f' $f | tail -n 1)"; done; done; }
# fold k only sets the init seed (seed + k) and an in-sample monitor; both runs of an architecture share one directory
fit() { # gpu name seedfold model extra...
  local gpu=$1 name=$2 k=$3 m=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu python -u -m hmsbest.train --config configs/spec_b0.yaml --set name=$name data_root=$D cache_dir=$C "run_folds=[$k]" num_workers=2 "$m" eeg_hi=40.0 $T "teacher_csv=$RP/pair_*/rows_for*.csv" "$STT" results_csv=/kaggle/working/results_$name$k.csv "$@" > /kaggle/working/logs/${name}_$k.txt 2>&1
}
( fit 0 FB 0 "$MB" batch_size=32; fit 0 FA 0 "$MA" ) &
( fit 1 FB 1 "$MB" batch_size=32; fit 1 FA 1 "$MA" ) &
progress; wait
R=/kaggle/working/runs
ls -la $R/FA $R/FB
for f in /kaggle/working/logs/F*.txt; do echo "== $f"; grep -h "teacher:" $f; tail -n 2 $f | cut -c1-200; done
# guards: (1) same code path as the submission on 400 train windows with >= 10 votes (in-sample, so optimistic);
# (2) the visible test.csv end to end
python -m hmsbest.infer --data_root $D --exp $R/FB:0.5 --exp $R/FA:0.5 --check 400
python -m hmsbest.infer --data_root $D --exp $R/FB:0.5 --exp $R/FA:0.5 --out /kaggle/working/submission.csv
# carry the teacher predictions forward
cp $IN/oof_teacher5.csv /kaggle/working/oof_teacher5.csv
for d in $RP/pair_*/; do mkdir -p $R/$(basename $d); cp $d/rows_for*.csv $R/$(basename $d)/; done
du -sh /kaggle/working/*
