# Deferred work: testing & self-training

The `pseudo_labeling` package ships as a lean, working pipeline. Two things were
intentionally left out to keep the initial implementation simple and are documented here so
they can be added back cleanly later.

## 1. Automated tests (property-based + unit)

The full spec (`.kiro/specs/semi-supervised-pseudo-labeling/`) defines 36 correctness
properties plus example/edge unit tests and an opt-in integration smoke test. None of that
test code is in the repo yet, and the in-memory test double that made it runnable without a
GPU was removed along with it.

### What to add back

- **A fake teacher backend** under `tests/` (not in the package). The pipeline only depends
  on the `TeacherBackend` / `LoadedModel` protocols in `pseudo_labeling/backends.py`, so a
  test double just needs to implement:
  - `train(data_yaml, base_model, seed, run_dir) -> str` (return a fake weights path)
  - `load(weights_path) -> LoadedModel` where `LoadedModel.infer(image_path)` returns a
    scripted `list[CandidateDetection]`.
  Make it deterministic and able to simulate train/inference failures so the error paths
  (Req 2.8, 3.7) can be exercised. Because it lives in `tests/`, it never ships in the
  package.
- **Dev dependencies**: recreate `requirements-dev.txt` with `pytest` and `hypothesis`.
- **Shared strategies** under `tests/strategies.py` (e.g. a `bboxes()` strategy biased
  toward out-of-range / boundary / degenerate boxes, and an `image_paths()` strategy).
- **One property test per correctness property**, each tagged
  `# Feature: semi-supervised-pseudo-labeling, Property {N}: ...`, min 100 examples.
- **Example/edge unit tests** for the side-effect and error-branch criteria.
- **An opt-in integration smoke test** using the real `UltralyticsBackend`, auto-skipped
  when no GPU/model is available.

The task list in `.kiro/specs/semi-supervised-pseudo-labeling/tasks.md` already enumerates
every test task (the `*`-marked subtasks) and maps each property to a task.

### Running tests in the Nix environment

`pytest`/`hypothesis` are not in the read-only Nix env. Use a throwaway venv:

```bash
python -m venv --system-site-packages .venv-test
. .venv-test/bin/activate
pip install pytest hypothesis
pytest -q
```

## 2. Iterative self-training (Req 10)

The self-training orchestrator (`selftrain.py`) and its config (`self_training`,
`max_iterations`) were removed because they were never wired into `system.py` and are not
needed for the core "generate once, merge, train" flow.

### What to add back

- Reintroduce `SelfTrainingOrchestrator` (retrain on ground truth + previously merged
  pseudo-labels; bounded 1..100 iterations; record each iteration in the manifest; stop and
  keep the last good model on failure).
- Re-add `self_training` / `max_iterations` to `PseudoLabelingConfig` and their validation
  in `config.validate()`.
- Add an `iterations` field back to `RunManifest`.
- In `PseudoLabelingSystem.run`, when `self_training` is enabled, loop
  "retrain -> generate -> merge" instead of the single pass, feeding merged pseudo-labels
  into the next iteration's training data.
- Restore the `--self-train` / `--max-iterations` CLI flags.
