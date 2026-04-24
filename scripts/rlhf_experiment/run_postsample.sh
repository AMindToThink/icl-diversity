#!/usr/bin/env bash
# Run the full post-sampling pipeline: wait for instruct sampling to finish,
# then score (ICL + baselines), run external NB scoring, run analysis.
#
# Intended to be invoked *after* all four OLMo-2-7B stages have been sampled.
# Auto-waits for the sampler process to exit before proceeding.

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "[postsample] waiting for any running sampler to exit..."
while pgrep -f "python scripts/rlhf_experiment/2_sample_responses.py" > /dev/null; do
  sleep 20
done
echo "[postsample] sampler clear at $(date)"

mkdir -p results/rlhf_experiment/logs

echo "[postsample] Stage 4: ICL scoring (all 4 stages × 2 prompt sets)"
CUDA_VISIBLE_DEVICES=1 uv run python scripts/rlhf_experiment/3_score_icl_diversity.py \
  2>&1 | tee results/rlhf_experiment/logs/stage4_icl_scoring.log

echo "[postsample] Stage 5: baselines (EAD + distinct-n + SentBERT)"
CUDA_VISIBLE_DEVICES=1 uv run python scripts/rlhf_experiment/4_score_baselines.py \
  2>&1 | tee results/rlhf_experiment/logs/stage5_baselines.log

echo "[postsample] Stage 5b: external NB responses"
CUDA_VISIBLE_DEVICES=1 uv run python scripts/rlhf_experiment/5b_score_external_nb.py \
  2>&1 | tee results/rlhf_experiment/logs/stage5b_external_nb.log

echo "[postsample] Stage 6: analysis + figures + macros"
uv run python scripts/rlhf_experiment/5_analyze_and_figures.py \
  2>&1 | tee results/rlhf_experiment/logs/stage6_analyze.log

echo "[postsample] done at $(date)"
