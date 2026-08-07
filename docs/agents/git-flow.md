---
featureBase: dev
deployTrigger: prod
---

# Git flow — clear-pipeline

Promotion chain: **feature → dev → staging → prod**.

`main` is **inactive** (a v0 stub; it appears on the remote as
`origin/main(inactive)` and is ~84 commits behind `dev`). Do **not** branch
from or open PRs against `main`.

- **featureBase: `dev`** — branch new work off `origin/dev`; feature PRs target `dev`.
- **deploy trigger: `prod`** — merges into `prod` are what deploy.
- `staging` sits between `dev` and `prod` for pre-prod validation.

An Exponential merge-hook promotes tickets QA → DONE on merge (added in #30).
