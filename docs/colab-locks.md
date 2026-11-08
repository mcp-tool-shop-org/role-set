# Time-Bounded Locks (`--exclude-newer`)

`uv.lock` has a committed upper timestamp bound beside it in `uv.lock.exclude-newer`. Both the
drift check (`scripts/colab/preflight_resolve.sh`) and lock regeneration (`scripts/colab/relock.sh`)
pass that bound to `uv 0.13.5` via `--exclude-newer <timestamp>` (see `scripts/colab/lock_common.sh`),
so the resolver sees only package files uploaded before that instant and a later upstream release on
PyPI cannot cause `LOCK_DRIFT`.

## Timestamp File Format (`uv.lock.exclude-newer`)

- `#` comment lines and blank lines are ignored.
- Exactly one remaining line must be an RFC 3339 UTC timestamp with whole seconds
  (`YYYY-MM-DDTHH:MM:SSZ`) representing a valid calendar date and time that is not in the future.
- A missing timestamp file refuses `LOCK_TIMESTAMP_MISSING`; an empty, multi-line, non-UTC,
  invalid, or future timestamp refuses `LOCK_TIMESTAMP_MALFORMED`.

## Drift Check and Bumping the Lock

- **Verify no drift:**
  ```bash
  bash scripts/colab/preflight_resolve.sh
  ```
- **Bump the bound and re-lock (in a deliberate bump PR):**
  ```bash
  bash scripts/colab/relock.sh --exclude-newer <YYYY-MM-DDTHH:MM:SSZ|now>
  ```
  `relock.sh` compiles the lock in a clean staging directory and writes `uv.lock` and
  `uv.lock.exclude-newer` atomically only when the compile succeeds.
