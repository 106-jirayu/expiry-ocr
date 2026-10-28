#!/usr/bin/env bash
# macOS / Linux: ./run.sh
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
[ -f .env ] || cp .env.example .env
exec .venv/bin/python run.py "$@"
