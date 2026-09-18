# Server-source synchronization

## Why this document exists

The research server is where the production experiments run. Its
`LingbotVA-OPD` directory is an unversioned working snapshot and is therefore
not automatically identical to this Git repository. A public checkout must
not be advertised as the exact server implementation unless the relevant files
have been compared.

## Synchronization contract

For every server file considered for publication, record:

```text
relative path
server SHA-256
public SHA-256
source date or run identifier
classification: canonical runtime | contract | diagnostic | historical
portability review: passed | needs parameterization | excluded
verification command
```

The server snapshot may contain more scripts than the public repository. That
is expected. One-off recovery jobs and private task wrappers should remain
internal unless they become a supported, portable entry point.

## Read-only comparison procedure

Run this from a machine with read access to the server. Replace the placeholders
with the configured SSH alias and server checkout path; do not commit those
private values into the repository.

```bash
server_host=<ssh-alias>
server_root=<remote-checkout>
public_root="$PWD"

ssh "$server_host" "cd '$server_root' && \
  find experiments scripts tests -type f \\
    \( -name '*.py' -o -name '*.sh' -o -name '*.json' \) -print0 | \\
  sort -z | xargs -0 sha256sum" > /tmp/wam-opd-server.sha256

python3 -m compileall -q experiments tests
git status --short
```

The comparison is a discovery step, not permission to overwrite the public
tree. Before copying a server file, inspect imports, absolute paths, output
roots, and serialized schemas. Parameterize infrastructure paths through
`.env`, CLI arguments, or a manifest and add a focused verification check.

## Current boundary

The public tree contains the portable orchestration and model-side contracts.
Large models, checkpoints, trajectories, simulator assets, logs, videos, and
server-specific output roots are intentionally excluded by `.gitignore` and
`docs/ARTIFACT_POLICY.md`.

If a server experiment depends on a file not yet present here, the correct
action is to classify it and synchronize it deliberately; do not silently
substitute a similarly named local script.
