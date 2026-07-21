# Contributing to AutoTimm

## Setup

```bash
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Running tests

```bash
pytest tests/ -v --tb=short --ignore=tests/test_hf_hub_backbones.py

# HF Hub backbone tests download real weights and are excluded above.
# Run the fast subset:
pytest tests/test_hf_hub_backbones.py -v -m "not slow"
```

## Lint and format

CI runs `ruff` and `black` and fails on violations — run both locally before
opening a PR:

```bash
ruff check src/
black --check src/         # or `black src/` to auto-format
```

## Pull requests

- Keep PRs focused; unrelated cleanup makes review harder.
- Add or update tests for behavior changes — CI runs the full suite
  (`.github/workflows/ci.yml`) across Python 3.10–3.13 on Linux and Windows.
- Update `CHANGELOG.md` under `[Unreleased]` for user-facing changes.

## Releasing

Releases are tag-driven (`.github/workflows/publish.yaml`): pushing a `vX.Y.Z`
tag runs the test suite, verifies the `setuptools-scm`-derived version matches
the tag, builds the package, and publishes to TestPyPI then PyPI.

Publishing uses [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/) —
no API token is stored in repo secrets. This requires the GitHub repository to
be registered as a trusted publisher on both
[pypi.org](https://pypi.org/manage/project/autotimm/settings/publishing/) and
[test.pypi.org](https://test.pypi.org/manage/project/autotimm/settings/publishing/)
for the `publish-pypi` / `publish-testpypi` workflow jobs before the first
release after this is configured.
