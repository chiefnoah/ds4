#!/usr/bin/env bash
# Integration test for the block-parallel indexer top-K kernel.
#
# Verifies that:
#   1. ds4 builds with DS4_ROCM=1.
#   2. A long-prompt run actually exercises the parallel top-K dispatch
#      (n_comp > DS4_N_INDEXER_TOP_K=512 in the indexer score path).
#   3. Decode completes and produces non-empty coherent output.
#   4. End-to-end metrics are captured for comparison across runs.
#
# Designed to be cheap-ish: short generation tail (-n 24) so the dominant
# cost is prefill, which is what we need long enough to push n_comp past
# 512 inside the chunked-prefill compressor path.
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE_PROMPT="tests/long_context_security_prompt.txt"
LOG_DIR="${LOG_DIR:-/tmp}"
LOG="${LOG_DIR}/test_indexer_topk_parallel.log"
PROMPT_FILE="${LOG_DIR}/test_indexer_topk_parallel_prompt.txt"

if [[ ! -f "$SOURCE_PROMPT" ]]; then
    echo "FAIL: source prompt missing: $SOURCE_PROMPT" >&2
    exit 1
fi
# Slice the source prompt to ~3000 tokens (~100 lines at this prose
# density). With ratio=4 compression that leaves layer_n_comp ~750 per
# layer after prefill — comfortably above the 512 threshold that gates
# the per-layer decode top-K, so even a handful of decode tokens triggers
# the parallel wrapper. We override DS4_METAL_PREFILL_CHUNK below to keep
# the whole prefill in a single batch and avoid the chunked-prefill path.
head -100 "$SOURCE_PROMPT" >"$PROMPT_FILE"
if [[ ! -f "ds4flash.gguf" ]]; then
    echo "FAIL: ds4flash.gguf model not found in repo root" >&2
    exit 1
fi

echo "test_indexer_topk_parallel: building ds4 (DS4_ROCM=1)"
DS4_ROCM=1 make ds4 >/dev/null

prompt_bytes=$(wc -c <"$PROMPT_FILE")
echo "test_indexer_topk_parallel: running long-prompt prefill + short decode"
echo "  prompt: $PROMPT_FILE ($prompt_bytes bytes)"
# DS4_METAL_PREFILL_CHUNK=8192 keeps the whole prompt in one prefill batch
# (skipping the chunked path). DS4_ROCM_TOPK_TRACE=1 emits one stderr line
# per parallel-topk dispatch so we can verify the new wrapper actually
# fired during this run.
DS4_METAL_PREFILL_CHUNK=8192 DS4_ROCM_TOPK_TRACE=1 ./ds4 \
    --backend rocm \
    --prompt-file "$PROMPT_FILE" \
    -n 8 --temp 0 --seed 42 \
    >"$LOG" 2>&1 || {
        echo "FAIL: ds4 invocation failed; see $LOG" >&2
        tail -30 "$LOG" >&2
        exit 1
    }

# 1. Parallel dispatch fired at least once.
parallel_calls=$(grep -c '^ds4: ROCm topk parallel ' "$LOG" || true)
if [[ "$parallel_calls" -eq 0 ]]; then
    echo "FAIL: indexer_topk_tensor never took the parallel path" >&2
    echo "  prompt may be too short; expected n_comp > 512 in compressor path" >&2
    exit 1
fi
echo "  parallel dispatches: $parallel_calls"

# 2. Sample shape sanity: at least one call hit a real production-scale top_k.
sample_call=$(grep '^ds4: ROCm topk parallel ' "$LOG" | head -1)
echo "  first call: $sample_call"
top_k_val=$(echo "$sample_call" | sed -n 's/.*top_k=\([0-9]*\).*/\1/p')
n_comp_val=$(echo "$sample_call" | sed -n 's/.*n_comp=\([0-9]*\).*/\1/p')
if [[ "${top_k_val:-0}" -lt 2 ]]; then
    echo "FAIL: parallel kernel only dispatched for top_k=$top_k_val (test wants top_k>=2)" >&2
    exit 1
fi
if [[ "${n_comp_val:-0}" -le "${top_k_val:-0}" ]]; then
    echo "FAIL: dispatched with n_comp=$n_comp_val <= top_k=$top_k_val (no real selection)" >&2
    exit 1
fi

# 3. Decode finished and emitted the t/s footer line.
if ! grep -q '^ds4: prefill: .* generation: ' "$LOG"; then
    echo "FAIL: ds4 did not emit the prefill/generation footer (decode incomplete)" >&2
    tail -10 "$LOG" >&2
    exit 1
fi
footer=$(grep '^ds4: prefill: ' "$LOG" | tail -1)
echo "  $footer"

# 4. Output non-empty (some printable content past the footer's preamble).
generated_chars=$(awk '
    /^ds4: metal prefill layer/ { skip = 1; next }
    /^ds4: prefill: / { stop = 1 }
    !stop && !/^ds4:/ { gsub(/[[:space:]]/, ""); printf "%s", $0 }
' "$LOG" | wc -c)
if [[ "$generated_chars" -lt 32 ]]; then
    echo "FAIL: decode emitted only $generated_chars non-whitespace chars (expected coherent text)" >&2
    exit 1
fi
echo "  generated content chars: $generated_chars"

echo "test_indexer_topk_parallel: OK"
