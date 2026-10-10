# Public Python issue regression suite

This non-formal suite freezes three resolved public GitHub issues as a
reproducible single-mode evaluation set:

| Task | Upstream issue | Production target |
| --- | --- | --- |
| `py-oss-001` | PySnooper #2 | `pysnooper/pysnooper.py` |
| `py-oss-002` | HTTPie #235 | `httpie/models.py` |
| `py-oss-003` | The Fuck #271 | `thefuck/rules/ls_lah.py` |

Each healthy baseline contains the exact target file from the upstream fix
commit, its upstream license, and small dependency-free public tests authored
for this project. `setup.patch` reverses only the upstream production change;
`gold.patch` reapplies that production change. Independently authored hidden
tests establish the reported regression without copying upstream test code.

This is deliberately a curated source snapshot, not a claim that the Agent was
run against each complete upstream repository or its complete historical test
matrix. `provenance.json` records the issue, exact commits, and commit-addressed
target/license URLs with SHA-256 hashes. This avoids treating repackagable
archive bytes as a stable source identity.

Validate structure and content locks without executing tests:

```powershell
python .\benchmarks\tasks\python\oss\validate.py --structure-only
```

Run the complete zero-model lifecycle (healthy, injected bug, hidden failure,
and restored fix):

```powershell
python .\benchmarks\tasks\python\oss\validate.py
```

The suite is intentionally non-formal. It works with the existing single eval
mode but is rejected by matrix, canary, shard, and merge release paths:

```powershell
repo-agent eval --benchmark-dir .\benchmarks\tasks\python\oss --variant full
```
