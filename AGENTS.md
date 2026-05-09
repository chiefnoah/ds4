# Agent Development Approach

Notes on how Claude works in this repo. `AGENT.md` covers project-level rules
(API surface, no-C++, etc.); this file is about *workflow* — what tools to
reach for, how to validate changes, and the cadence the user has asked for.

## Required tools

- **`tsk`** — Git-backed task tracker. The user wants outstanding work tracked
  here so future sessions can pick up cold. Use `tsk` for non-trivial findings,
  follow-up experiments, regressions, and performance notes.

  Basic commands:
  - `tsk list -c 20` shows the active queue, top-of-stack first.
  - `tsk show -T tsk-N` shows a task by visible id.
  - `tsk append -t "Title" -b $'Body...'` creates a new task at the bottom of
    the active queue.
  - `tsk push -t "Title" -b $'Body...'` creates a new task at the top when it
    should be next.
  - `tsk prop ...` can tag or query tasks when properties are useful.
  - `tsk help <command>` is authoritative for exact flags.

  Prefer creating new tasks over losing context in chat. A good task body has:
  status, concise goal, evidence or file/function pointers, exact next steps,
  validation commands, and related task ids. For ROCm performance work, include
  host facts, env vars, baseline numbers, and whether the task is safe to do
  alongside other active work.

  Do **not** use `tsk edit` from a non-interactive shell. It spawns `$EDITOR`
  and can hang indefinitely. To change an existing task, either use a normal
  interactive editor yourself or, when a plain file exists under `.tsk/tasks/`,
  edit that file directly. If in doubt, append a new corrective task instead
  of risking a hung `tsk edit`.

  The command writes Git-backed task state under the repo, so sandboxed agents
  may need approval. Keep privileged task commands homogeneous so the user can
  blanket-approve them; prefer the exact shape
  `tsk append -t "..." -b $'...'` for task creation.

  The Claude-side `TaskCreate`/`TaskUpdate` is fine for scratchpad tracking
  *during* a session, but the canonical record for future sessions is `tsk`.

## Profiling and benchmarking

- **End-to-end timing**: `time ./ds4 --backend rocm -p "..." -n N --temp 0
  --seed 42`. Final line is `prefill: X t/s, generation: Y t/s`. Capture both
  from a quiet system (avoid kernel/rust builds in parallel — load average
  >5 produces noisy measurements).
- **Per-kernel histogram**: `DS4_ROCM_TIME=1 ./ds4 --backend rocm ...`. The
  cleanup hook prints `ROCm timing histogram` sorted by total time, with
  call count and ms/call per wrapper. Implemented via
  `ROCM_TIME_SCOPE("name")` (cleanup-attribute RAII) on every wrapper.
- **Per-kernel call trace**: `DS4_ROCM_TRACE=1 ./ds4 ... 2>&1 | grep "ROCm
  enter" | sort | uniq -c | sort -rn`. Useful for seeing how many times each
  kernel fires per token / per layer.
- **Microbench an isolated kernel**: build a small driver that links
  `ds4_rocm.o`, calls `ds4_metal_set_model_map(...)` once (otherwise every
  call re-uploads the synthetic model and dominates timing), then loops the
  target wrapper. Pattern: compile the driver with `cc -c`, then link with
  `hipcc` to pick up `__hipRegisterFatBinary` and friends.
- **Disk speed sanity**: `dd if=<file> of=/dev/null bs=64M iflag=direct
  count=64` — large blocks saturate the NVMe; bs=1M does not.

## Hardware context (gfx1151, Strix Halo APU)

- 40 CUs, RDNA 3.5, **wave size = 32** (use `__shfl_down(v, off)` for in-wave
  reductions; no `_sync` suffix needed in HIP).
- UMA: VRAM is a 96 GiB BIOS carve-out from 128 GiB system RAM (32 GiB stays
  with the OS). `hipMemcpy` H2D is just a memcpy across the same memory bus,
  so threading disk reads against H2D doesn't add bandwidth — only latency
  hiding. See `set_model_map_range` and the load-path comments.
- `hipHostRegister` on the whole 80 GiB GGUF fails because BIOS reserves
  most of UMA for the iGPU; only ~30 GiB is host-pinnable in practice. We
  fall through to a device-resident copy.
- LDS budget: ≤16 KiB per block keeps occupancy on RDNA 3.5; an 8-wave
  block with `in_dim=4096` floats fits exactly.
- Memory bandwidth: LPDDR5X-8000 at 256-bit ≈ 256 GB/s peak. Production
  decode hits ~75 GiB/s effective on the routed-MoE matvec — that's the
  realistic ceiling for memory-bound kernels.

## Validation gates (after every kernel-level change)

1. `DS4_ROCM=1 make ds4 ds4_rocm_kernel_test`
2. `./ds4_rocm_kernel_test` — must end with `OK` (40 tests).
3. `time ./ds4 --backend rocm -p "Briefly explain what makes the AMD Strix
    Halo APU well-suited for running large language models on a single chip."
    -n 200 --temp 0 --seed 42` — output must stay coherent (multi-paragraph
    thinking-mode reply).

A kernel test that times-out or returns 0 silently is *not* a green test —
the kernel test harness is the canonical correctness oracle, including
edge cases and quant-format roundtrips. Do not skip it.

## Code rules specific to ds4_rocm.c

- File is wrapped in `extern "C"` so **no C++ templates** in the .c. Use
  preprocessor macros to instantiate kernel variants (see
  `ROCM_DEFINE_MATMUL_Q8_0_LDS(N)`).
- Wrappers are *expected* to be flat C functions taking `ds4_metal_*`
  pointers — they share the ABI with the Metal backend so `ds4.c`'s graph
  scheduler doesn't need to know which backend is live.
- Every new wrapper should add `if (rocm_trace()) fprintf(...)` for
  diagnostics and `ROCM_TIME_SCOPE("name")` for the histogram. Both are
  free at runtime when their env-vars aren't set.
- Reduction tail: `rocm_wave32_sum(v)` is the canonical wave32 shuffle
  sum. Use it instead of LDS-based fan-ins for any block whose threads
  per row equals the wave size.

## Subagents

Use `Agent` (subagent_type=general-purpose or Explore) when:
- A search will need >3 tool calls and the results would crowd context.
- Independent investigations can be run in parallel (one Agent call per
  investigation, all in the same message).

Don't delegate kernel writes — those benefit from interactive iteration
against the build/test loop in the main session.

## Commit cadence

Commit at clean checkpoints (kernel passes tests AND end-to-end produces
coherent output AND benchmark numbers are captured). Keep commits to one
logical change per commit. Do not commit on red.
