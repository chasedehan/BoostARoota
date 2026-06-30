#!/usr/bin/env bash
set -euo pipefail

# Automated PyPI publish script for BoostARoota
# Usage:
#   ./scripts/publish.sh            # build, check, and upload to PyPI
#   ./scripts/publish.sh --test     # upload to TestPyPI instead
#   ./scripts/publish.sh --check-only  # build and check only, no upload
#
# Credentials:
#   - Uses ~/.pypirc by default (username = __token__, password = pypi-...)
#   - Or set PYPI_API_TOKEN env var for CI / GitHub Actions

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

MODE="pypi"
CHECK_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --test) MODE="testpypi" ;;
    --check-only) CHECK_ONLY=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $arg" && exit 1 ;;
  esac
done

echo "==> Cleaning previous builds"
make clean

echo "==> Installing build tools"
python3 -m pip install --quiet --break-system-packages --upgrade build twine 2>/dev/null || \
python3 -m pip install --quiet --upgrade build twine

echo "==> Building sdist and wheel"
python3 -m build

echo "==> Checking distributions"
python3 -m twine check dist/*

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "==> Check only mode — skipping upload"
  ls -lh dist/
  exit 0
fi

if [[ "$MODE" == "testpypi" ]]; then
  echo "==> Uploading to TestPyPI"
  python3 -m twine upload --repository testpypi dist/*
else
  echo "==> Uploading to PyPI"
  if [[ -n "${PYPI_API_TOKEN:-}" ]]; then
    python3 -m twine upload -u __token__ -p "$PYPI_API_TOKEN" dist/*
  else
    python3 -m twine upload dist/*
  fi
fi

echo "==> Done. dist/ contents:"
ls -lh dist/
