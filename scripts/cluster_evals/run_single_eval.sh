#!/bin/bash
# Single-job eval runner. Invoked by launch_evals.py via sbatch.
# SLURM resource directives are passed on the sbatch command line by the launcher.
#
# Required args: --config --task_id --perturbation_id --port --experiment_name --run_id
# Optional args: --debug

set -euo pipefail

DEBUG=false
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --config)           CONFIG="$2";           shift 2 ;;
    --task_id)          TASK_ID="$2";          shift 2 ;;
    --perturbation_id)  PERTURBATION_ID="$2";  shift 2 ;;
    --port)             PORT="$2";             shift 2 ;;
    --experiment_name)  EXPERIMENT_NAME="$2";  shift 2 ;;
    --run_id)           RUN_ID="$2";           shift 2 ;;
    --debug)            DEBUG=true;            shift 1 ;;
    *)                  shift ;;
  esac
done

_yaml_get() {
  python3 - "$CONFIG" "$1" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
keys = sys.argv[2].split(".")
val = cfg
for k in keys:
    val = val[k]
print("" if val is None else val)
PYEOF
}

POLICY_CONFIG=$(_yaml_get "policy.config")
CHECKPOINT_PATH=$(_yaml_get "policy.checkpoint_path")
POLICY_RUN_DIR=$(_yaml_get "policy.run_dir")
EXTRA_POLICY_ARGS=$(_yaml_get "policy.extra_args")

REPEATS=$(_yaml_get "eval.repeats")
MAX_STEPS=$(_yaml_get "eval.max_steps")
RECORD_VIDEO=$(_yaml_get "record_video")

# ---------------------------------------------------------------------------

REALM_ROOT=$(pwd)

export HF_HOME=$REALM_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$REALM_ROOT/hf_cache
[[ -d "$HF_HOME" ]] || mkdir -p "$HF_HOME"

export XDG_CACHE_HOME=$REALM_ROOT/python_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

POLICY_SCRIPT="${POLICY_RUN_DIR}/scripts/serve_policy.py"

if [ "$DEBUG" = "false" ]; then
  cd "$POLICY_RUN_DIR"
  # shellcheck disable=SC2086  # EXTRA_POLICY_ARGS is intentionally word-split
  uv run "$POLICY_SCRIPT" \
    --port="$PORT" \
    $EXTRA_POLICY_ARGS \
    policy:checkpoint \
    --policy.config="$POLICY_CONFIG" \
    --policy.dir="$CHECKPOINT_PATH" & SERVER_PID=$!
  sleep 60
fi

# ---------------------------------------------------------------------------

cd "$REALM_ROOT"
mkdir -p "$REALM_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/mamba_cache/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/pip_cache/$SLURM_JOB_ID"

if [ "$DEBUG" = "true" ]; then
  MODEL_NAME="debug"
else
  CLEAN_PATH="${CHECKPOINT_PATH%/}"
  MODEL_NAME=$(basename "$(dirname "${CLEAN_PATH%/}")")_$(basename "${CLEAN_PATH%/}")
fi

RECORD_VIDEO="${RECORD_VIDEO:-true}"

apptainer exec \
  --userns \
  --nv \
  --writable-tmpfs \
  --bind "$(pwd):/app" \
  --bind "$REALM_DATA_PATH/datasets:/data" \
  --bind "$REALM_DATA_PATH/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit" \
  --bind "$REALM_DATA_PATH/isaac-sim/cache/ov:/root/.cache/ov" \
  --bind "$REALM_DATA_PATH/isaac-sim/cache/pip:/root/.cache/pip" \
  --bind "$REALM_DATA_PATH/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache" \
  --bind "$REALM_DATA_PATH/isaac-sim/cache/computecache:/root/.nv/ComputeCache" \
  --bind "$REALM_DATA_PATH/isaac-sim/logs:/root/.nvidia-omniverse/logs" \
  --bind "$REALM_DATA_PATH/isaac-sim/config:/root/.nvidia-omniverse/config" \
  --bind "$REALM_DATA_PATH/isaac-sim/data:/root/.local/share/ov/data" \
  --bind "$REALM_DATA_PATH/isaac-sim/documents:/root/Documents" \
  --bind "$REALM_ROOT/tmp/$SLURM_JOB_ID:/tmp" \
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env "MAMBA_CACHE_DIR=$REALM_ROOT/mamba_cache/$SLURM_JOB_ID" \
  --env "PIP_CACHE_DIR=$REALM_ROOT/pip_cache/$SLURM_JOB_ID" \
  "$REALM_SIF" \
  micromamba run -n omnigibson python examples/02_eval_dynamic_scenes.py \
  --perturbation_id "$PERTURBATION_ID" \
  --task_id "$TASK_ID" \
  --repeats "$REPEATS" \
  --max_steps "$MAX_STEPS" \
  --model "$MODEL_NAME" \
  --port "$PORT" \
  --run_id "$RUN_ID" \
  --experiment_name "$EXPERIMENT_NAME" \
  --record_video "$RECORD_VIDEO"
