# Job 18: make the LATEST version of the runner hold FA and FB (full-fit, job 16), because a notebook-output input always
# resolves to the latest version of that notebook (Version 2 of the submit notebook failed: the runner input was Version 16,
# which has D but not FA/FB). D keeps coming from the "weights + cv-check" notebook (its Version 1).
# Copy FA/FB if the mounted input has them (minutes); otherwise retrain them exactly as in job 16 (~2 h).
export PYTHONPATH=/kaggle/working/repo
mkdir -p /kaggle/working/runs
FA=$(dirname "$(find /kaggle/input -path '*runs/FA/config.yaml' | head -1)")
FB=$(dirname "$(find /kaggle/input -path '*runs/FB/config.yaml' | head -1)")
echo "found FA: $FA ($(ls $FA/fold*.pt 2>/dev/null | wc -l) models)  FB: $FB ($(ls $FB/fold*.pt 2>/dev/null | wc -l) models)"
if [ "$(ls $FA/fold*.pt 2>/dev/null | wc -l)" = "2" ] && [ "$(ls $FB/fold*.pt 2>/dev/null | wc -l)" = "2" ]; then
  cp -r $FA /kaggle/working/runs/FA; cp -r $FB /kaggle/working/runs/FB
  IN=$(dirname "$(find /kaggle/input -name oof_teacher5.csv | head -1)")
  cp $IN/oof_teacher5.csv /kaggle/working/ 2>/dev/null
  for d in $IN/runs/pair_*/; do mkdir -p /kaggle/working/runs/$(basename $d); cp $d/rows_for*.csv /kaggle/working/runs/$(basename $d)/; done
  echo "copied FA/FB from the input"
else
  echo "FA/FB not in the input: retraining them (job 16)"
  bash jobs/job16.sh
fi
ls -la /kaggle/working/runs/FA /kaggle/working/runs/FB
N=$(ls /kaggle/working/runs/FA/fold*.pt /kaggle/working/runs/FB/fold*.pt 2>/dev/null | wc -l)
[ "$N" = "4" ] || { echo "FOUT: $N van 4 FA/FB-modellen"; exit 1; }
du -sh /kaggle/working/*
