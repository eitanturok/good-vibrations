# perf investigation for scripts/masked_attn_gastro_m.sh

Goal: make the 4 runs in masked_attn_gastro_m.sh faster without changing behavior/results.
Original run killed (tmux session `train`, was mid baseline run #1) so we can profile cleanly.

Workflow: profile -> find bottleneck -> propose a SINGLE targeted change as a diff -> show user
-> only apply to src/ after approval. Nothing here touches src/ until approved.

## Log

- [ ] 1. Baseline profiling pass (composer Profiler + torch profiler, --debug 1, short run)
      to see where wall-clock goes: dataloader / forward / backward / optimizer / eval / compile.
- [ ] 2. --num-workers is 4 (script default), machine has 32 cores. Check if dataloading is
      actually a bottleneck before touching this.
- [ ] 3. torch.compile mode is "default" -- check compile time amortization and whether
      reduce-overhead / max-autotune helps for this shape.
- [ ] 4. FreqEncoder note in script header: runs batch_size * n_lasers (256*100=25600) sequences
      per step, much bigger effective batch than laser encoder/decoder. Confirm with profiler.

## Findings

(filled in as we go)

## Applied speedups (approved by user)

(none yet)
