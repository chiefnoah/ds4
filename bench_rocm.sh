#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./bench_rocm.sh [options]

Runs reproducible ROCm CLI benchmarks and writes logs under bench-results/.

Options:
  -m, --model FILE       Model path. Default: ./ds4flash.gguf
  -p, --prompt TEXT      Prompt text. Default: Strix Halo validation prompt
  -n, --tokens N         Generated tokens. Default: 200
  --ctx N                Context size passed to ds4. Default: unset
  --seed N               Sampling seed. Default: 42
  --temp F               Temperature. Default: 0
  --runs N               Measured runs. Default: 3
  --warmup N             Warmup runs before measurement. Default: 1
  --out DIR              Output directory. Default: bench-results
  --profile              Enable DS4_METAL_GRAPH_TOKEN_PROFILE and PREFILL_PROFILE
  --rocm-time            Enable DS4_ROCM_TIME=1
  --sync-each            Enable DS4_ROCM_SYNC_EACH=1
  --no-build             Do not run make before benchmarking
  -h, --help             Show this help

Environment:
  EXTRA_DS4_ARGS         Extra arguments appended to ./ds4.
  EXTRA_ENV              Extra env assignments, e.g. 'FOO=1 BAR=2'.
EOF
}

model="./ds4flash.gguf"
prompt="Briefly explain what makes the AMD Strix Halo APU well-suited for running large language models on a single chip."
tokens=200
ctx=""
seed=42
temp=0
runs=3
warmup=1
out_dir="bench-results"
do_build=1
profile=0
rocm_time=0
sync_each=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        -m|--model)
            model="${2:?missing value for $1}"
            shift 2
            ;;
        -p|--prompt)
            prompt="${2:?missing value for $1}"
            shift 2
            ;;
        -n|--tokens)
            tokens="${2:?missing value for $1}"
            shift 2
            ;;
        --ctx)
            ctx="${2:?missing value for $1}"
            shift 2
            ;;
        --seed)
            seed="${2:?missing value for $1}"
            shift 2
            ;;
        --temp)
            temp="${2:?missing value for $1}"
            shift 2
            ;;
        --runs)
            runs="${2:?missing value for $1}"
            shift 2
            ;;
        --warmup)
            warmup="${2:?missing value for $1}"
            shift 2
            ;;
        --out)
            out_dir="${2:?missing value for $1}"
            shift 2
            ;;
        --profile)
            profile=1
            shift
            ;;
        --rocm-time)
            rocm_time=1
            shift
            ;;
        --sync-each)
            sync_each=1
            shift
            ;;
        --no-build)
            do_build=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "bench_rocm.sh: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$runs" in (*[!0-9]*|"") echo "bench_rocm.sh: --runs must be an integer" >&2; exit 2;; esac
case "$warmup" in (*[!0-9]*|"") echo "bench_rocm.sh: --warmup must be an integer" >&2; exit 2;; esac

if [ "$do_build" -ne 0 ]; then
    DS4_ROCM=1 make ds4
fi

if [ ! -x ./ds4 ]; then
    echo "bench_rocm.sh: ./ds4 is missing or not executable" >&2
    exit 1
fi

mkdir -p "$out_dir"
stamp="$(date +%Y%m%d-%H%M%S)"
summary="$out_dir/rocm-$stamp.tsv"
meta="$out_dir/rocm-$stamp.meta"

{
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "pwd=$(pwd)"
    echo "model=$model"
    echo "tokens=$tokens"
    echo "ctx=$ctx"
    echo "seed=$seed"
    echo "temp=$temp"
    echo "runs=$runs"
    echo "warmup=$warmup"
    echo "profile=$profile"
    echo "rocm_time=$rocm_time"
    echo "sync_each=$sync_each"
    echo "extra_ds4_args=${EXTRA_DS4_ARGS:-}"
    echo "extra_env=${EXTRA_ENV:-}"
    if command -v rocm-smi >/dev/null 2>&1; then
        echo "--- rocm-smi --showproductname --showuse --showmemuse ---"
        rocm-smi --showproductname --showuse --showmemuse || true
    fi
} >"$meta"

printf "kind\trun\texit\twall_sec\tprefill_tps\tgeneration_tps\tlog\n" >"$summary"

run_one() {
    kind="$1"
    idx="$2"
    log="$out_dir/rocm-$stamp-$kind-$idx.log"
    tmp="$log.tmp"

    args=(./ds4 --backend rocm -m "$model" -p "$prompt" -n "$tokens" --temp "$temp" --seed "$seed")
    if [ -n "$ctx" ]; then
        args+=(--ctx "$ctx")
    fi
    if [ -n "${EXTRA_DS4_ARGS:-}" ]; then
        # Intentional word splitting for caller-provided CLI fragments.
        # shellcheck disable=SC2206
        extra_args=(${EXTRA_DS4_ARGS})
        args+=("${extra_args[@]}")
    fi

    env_args=()
    if [ "$profile" -ne 0 ]; then
        env_args+=(DS4_METAL_GRAPH_TOKEN_PROFILE=1 DS4_METAL_GRAPH_PREFILL_PROFILE=1)
    fi
    if [ "$rocm_time" -ne 0 ]; then
        env_args+=(DS4_ROCM_TIME=1)
    fi
    if [ "$sync_each" -ne 0 ]; then
        env_args+=(DS4_ROCM_SYNC_EACH=1)
    fi
    if [ -n "${EXTRA_ENV:-}" ]; then
        # Intentional word splitting for KEY=VALUE fragments.
        # shellcheck disable=SC2206
        extra_env=(${EXTRA_ENV})
        env_args+=("${extra_env[@]}")
    fi

    start_ns="$(date +%s%N)"
    set +e
    env "${env_args[@]}" "${args[@]}" >"$tmp" 2>&1
    rc="$?"
    set -e
    end_ns="$(date +%s%N)"
    mv "$tmp" "$log"

    wall_sec="$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.3f", (e - s) / 1000000000.0 }')"
    final_line="$(grep -E 'prefill: .*generation:' "$log" | tail -n 1 || true)"
    prefill="$(printf '%s\n' "$final_line" | sed -n 's/.*prefill:[[:space:]]*\([0-9.][0-9.]*\)[[:space:]]*t\/s.*/\1/p')"
    generation="$(printf '%s\n' "$final_line" | sed -n 's/.*generation:[[:space:]]*\([0-9.][0-9.]*\)[[:space:]]*t\/s.*/\1/p')"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$kind" "$idx" "$rc" "$wall_sec" "${prefill:-NA}" "${generation:-NA}" "$log" >>"$summary"

    if [ "$rc" -ne 0 ]; then
        echo "bench_rocm.sh: $kind run $idx failed with exit $rc; see $log" >&2
        return "$rc"
    fi
}

for i in $(seq 1 "$warmup"); do
    run_one warmup "$i"
done

for i in $(seq 1 "$runs"); do
    run_one measured "$i"
done

echo "summary: $summary"
echo "metadata: $meta"
awk -F '\t' '
    NR == 1 { next }
    $1 == "measured" && $3 == "0" && $5 != "NA" && $6 != "NA" {
        n++
        pre += $5
        gen += $6
        wall += $4
    }
    END {
        if (n > 0) {
            printf "measured_avg: runs=%d wall=%.3f sec prefill=%.3f t/s generation=%.3f t/s\n",
                   n, wall / n, pre / n, gen / n
        }
    }
' "$summary"
