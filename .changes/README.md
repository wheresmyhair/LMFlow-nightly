# Change fragments

Every significant product or infrastructure pull request to `nightly` adds one small JSON fragment at:

```text
.changes/<module>/<short-slug>.json
```

Supported modules are `core`, `data`, `agentic`, `ci`, `docs`, `packaging`, and `tooling`. Copy `_template.json`, place the copy in the matching module directory, and use a lowercase, hyphenated filename.

The `upstream` field records publication intent:

- `candidate`: designed for modular export to upstream LMFlow.
- `private`: intentionally stays in LMFlow Nightly.
- `blocked`: requires more work or a decision before export.

Validate active fragments with:

```bash
python .github/scripts/change_notes.py check
```

On a release branch, generate the review document with:

```bash
python .github/scripts/change_notes.py build \
  --version 2026.08 \
  --output docs/releases/2026.08.md
```

Review and commit the generated document, then remove the fragments consumed by that release in the same release pull request. The generated document remains as the durable record.
