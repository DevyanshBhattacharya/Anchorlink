# Privacy and safety notes for this repository

A short record of what was checked before this was made public, and what you
should decide for yourself.

## What is deliberately not here

| | why |
| --- | --- |
| `student_resource/` — the challenge dataset | the organisers' data; 2.3 GB; see SETUP.md to obtain it |
| `output/*.tsv` — the submission files | derived from that data, ~900 MB |
| `work/` — caches, indexes, features, models | derived, ~40 GB |
| `*.zip` — the packaged submission | contains the outputs |
| the source PDFs | large, and not ours to redistribute |

## Challenge records are masked, everywhere

The error analysis needs worked examples to be worth reading, but the records
themselves are the organisers' data. `ber/redact.py` masks them so the *structure*
survives and the identity does not:

```
S1:   W1 W2 Private Limited   ·  36/2342 W5 W6 W7
cand: W1 W2 Limited           ·  36/2342 W5 W6 W7
```

You can still see the dropped word, which is the entire point of that example,
without seeing the business. Legal forms, street vocabulary and digits are kept
because they carry the pattern; everything else becomes a stable per-example
placeholder.

* `results/` is generated with masking **on by default**.
* `python -m ber.final_report --no-redact` writes the real records, for local
  reading only. That output lands in `work/reports/`, which is not committed.
* Test fixtures use invented names and streets, not dataset records.
* The one test that must use real records (`test_features.py`) **looks them up by
  entity id** from the dataset at run time and skips when the data is absent, so
  nothing is written down in the repository.

## Checked and clean

* **No credentials.** No API keys, tokens, passwords or private keys.
* **No `.env`**, and `.env*` is ignored.
* **No absolute paths.** Reports are written with paths relative to the
  repository root, so a checkout's user name is never published.
* **No `eval`/`exec`.** The one place a stored decision rule is parsed back uses
  `ast.literal_eval`.
* **No `shell=True`.** Every `subprocess` call passes an argument list.
* **No network at inference.** `HF_HUB_OFFLINE=1` is exported by `run_all.sh`,
  which is also a competition requirement.

## Two things to be aware of

**1. `pickle` is used for local caches.** Indexes, corpus statistics and models
are pickled into `work/`. Unpickling executes code, so this is only safe because
the pipeline writes those files itself. **Never point this at a `work/` directory
you downloaded from someone else.** Nothing pickled is committed.

**2. Your commit email is public.** Commits carry whatever `git config user.email`
holds, and GitHub shows it forever. If you would rather not publish yours, set a
noreply address *before* pushing:

```bash
git config user.email "<id>+<username>@users.noreply.github.com"
git commit --amend --reset-author --no-edit          # last commit
# or rewrite every commit made so far:
git rebase -r --root --exec 'git commit --amend --reset-author --no-edit'
```

Find your noreply address at GitHub → Settings → Emails.

## Licence

There is no `LICENSE` file yet. Without one, default copyright applies and nobody
may reuse the code. Add one before publishing if you want it reusable — MIT is
consistent with every dependency here. The dataset is **not** yours to license
either way, which is another reason it is not in the repository.
