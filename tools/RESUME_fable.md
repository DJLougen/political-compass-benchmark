# Resuming the Fable benchmark run

The batched Claude-CLI runner (`tools/run_fable_cli.py`) is **crash-safe and
resumable**. If it stops for any reason — crash, killed process, credits hitting
$0, machine reboot — you lose **nothing already written** and can pick up exactly
where it left off.

## Why nothing is lost

- Each batch's answers are appended to `data/responses/mass_<run_label>.jsonl` and
  `flush()`ed to disk immediately, so on-disk progress is always current.
- On restart, `load_done()` reads every existing response file and skips every
  `(item_id, rep)` already recorded. A partially written / truncated final line
  (from a hard kill mid-write) is detected as invalid JSON and simply redone —
  verified.
- The three passes are separate files, so a completed pass re-scans in a second
  and does no work; a partial pass resumes item-by-item.

## Resume command (identical to the original launch)

```bash
cd /home/samcardillo/ai-neutrality-benchmark
export PATH="$HOME/.local/bin:$PATH"
python3 tools/run_fable_cli.py --model claude-fable-5 --sample 0 --reps 1 \
    --batch 12 --concurrency 5 --cap-usd 135
```

Run the **exact same command**. It prints `N already done` per pass and only does
the remainder, then runs fit + anchor and prints the result.

## Notes

- **Cost cap is per invocation.** `--cap-usd` counts spend for *this* run from $0.
  On resume only the *remaining* items are done, so total spend across restarts
  stays bounded by the full-run estimate (~$110–123) regardless — the cap is a
  runaway guard, not a cumulative ledger.
- **Don't run two copies at once** against the same files — one process only.
- Final output: `data/stances/mass_anchored_claude-fable-5.json`.
- Responses live under `data/responses/mass_claude-fable-5*.jsonl` (git-ignored).
