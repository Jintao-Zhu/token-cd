#!/usr/bin/env bash
# Install scikit-learn into an isolated overlay so the pi05 environment itself is untouched.
# --no-deps is deliberate: numpy/scipy/jax must keep the exact versions the pi0-FAST
# checkpoint was validated against. joblib and threadpoolctl are sklearn runtime deps that
# pi05 lacks, so they are installed into the same overlay.
set -u
PY=/home/leju-suzhou/miniconda3/envs/pi05/bin/python
OV=/home/leju-suzhou/zjt_ws/pi05_sklearn_overlay
mkdir -p "$OV"
echo "=== freeze BEFORE ==="
"$PY" -m pip freeze > /tmp/pi05_freeze_before.txt 2>/dev/null
wc -l /tmp/pi05_freeze_before.txt

echo "=== installing scikit-learn==1.7.2 (matching the OpenVLA env) ==="
"$PY" -m pip install --target "$OV" --no-deps \
  "scikit-learn==1.7.2" joblib threadpoolctl
echo "=== install rc=$? ==="

echo "=== freeze AFTER ==="
"$PY" -m pip freeze > /tmp/pi05_freeze_after.txt 2>/dev/null
echo "--- diff (must be empty) ---"
diff /tmp/pi05_freeze_before.txt /tmp/pi05_freeze_after.txt && echo "pi05 UNCHANGED"

echo "=== import check via overlay ==="
PYTHONPATH="$OV" "$PY" -c "
import sklearn, numpy, scipy, jax
print('sklearn', sklearn.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| jax', jax.__version__)
"
echo "=== DONE ==="
