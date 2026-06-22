# Publishing the KGCaRe Artifact

This checklist prepares a clean GitHub checkout from the working tree while
excluding local environments, generated indexes, run outputs, copied PDFs, and
private workspace notes.

## 1. Clone a clean target checkout

```bash
git -c http.sslCAInfo=/etc/ssl/certs/ca-certificates.crt clone \
  https://github.com/AI-Researchers/KGCaRe.git \
  /tmp/kgcare-publish
```

Use a new path each time if you want a completely fresh preview.

## 2. Prepare the publish artifact

Run this from the working tree root:

```bash
scripts/prepare_publish_artifact.sh /tmp/kgcare-publish
```

The script refuses to run if the target is the working tree itself.

## 3. Audit before committing

```bash
find /tmp/kgcare-publish -type d \
  \( -name __pycache__ -o -name adapter_runs -o -name indexes -o -name workspaces -o -name adapter_data -o -name outputs -o -name storage -o -name .venv \) \
  -not -path '/tmp/kgcare-publish/.git' -print

rg -n '([P]assw0rd|[P]@ssw0rd|[t]oken-abc123|/home/[s]imsam)' \
  /tmp/kgcare-publish -g '!**/.git/**'

rg -n '\bsk-[A-Za-z0-9_-]{20,}\b' \
  /tmp/kgcare-publish -g '!**/.git/**'
```

All three commands should produce no output except the first command when there
are actual generated directories to remove.

## 4. Commit with artifact-safe local Git identity

```bash
git -C /tmp/kgcare-publish config user.name "KGCaRe Developers"
git -C /tmp/kgcare-publish config user.email "kgcare-artifact@example.com"
git -C /tmp/kgcare-publish switch -c paper-artifact-refresh
git -C /tmp/kgcare-publish add -A
git -C /tmp/kgcare-publish commit -m "Prepare KGCaRe paper artifact"
```

Use a branch first. Do not push directly to `main` unless the team explicitly
wants to replace the existing repository layout.

## 5. Push and refresh anonymous mirror

```bash
git -C /tmp/kgcare-publish -c http.sslCAInfo=/etc/ssl/certs/ca-certificates.crt \
  push origin paper-artifact-refresh
```

After the public GitHub branch is updated, refresh the mirror on
`anonymous.4open.science` using the same repository and branch.
