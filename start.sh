#!/usr/bin/env bash
# Start Astro Explorer (backend + frontend). Ctrl-C stops both.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

# Auto-load .env (gitignored; see .env.example).
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

command -v uv >/dev/null || { echo "error: 'uv' not found. Install: https://docs.astral.sh/uv/"; exit 1; }
command -v npm >/dev/null || { echo "error: 'npm' not found. Install Node.js: https://nodejs.org/"; exit 1; }

if [ ! -d "$FRONTEND/node_modules" ]; then
  echo "==> Installing frontend deps (first run)…"
  (cd "$FRONTEND" && npm install)
fi

cleanup() {
  echo
  echo "==> Shutting down…"
  [ -n "${BACKEND_PID:-}" ] && kill "$BACKEND_PID" 2>/dev/null || true
  [ -n "${FRONTEND_PID:-}" ] && kill "$FRONTEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting backend on http://127.0.0.1:8000"
(cd "$BACKEND" && uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload) &
BACKEND_PID=$!

echo "==> Starting frontend on http://localhost:5173"
(cd "$FRONTEND" && npm run dev) &
FRONTEND_PID=$!

echo
echo "Astro Explorer is running. Open http://localhost:5173"
echo "Press Ctrl-C to stop."
wait
