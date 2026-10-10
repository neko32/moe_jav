#!/usr/bin/env bash
# Local helper for moe_jav (Linux/macOS).
#
#   scripts_local/run.sh check          fmt + clippy + tests + coverage report
#   scripts_local/run.sh live           tests against LM Studio + Jev (needs API_KEY_JEV)
#   scripts_local/run.sh apply FILE     underwrite one application
#   scripts_local/run.sh eval [MODE]    routing | e2e | all  (default: routing)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/app"
cd "$APP"

cmd="${1:-check}"
shift || true

case "$cmd" in
  check)
    cargo fmt --check
    cargo clippy --all-targets -- -W clippy::pedantic -D warnings
    cargo llvm-cov --fail-under-lines 90 --ignore-filename-regex 'main\.rs' \
      --html --output-dir "$ROOT/docs/coverage"
    echo "Coverage report: $ROOT/docs/coverage/html/index.html"
    ;;
  live)
    cargo test --test live_services -- --ignored --test-threads=1
    ;;
  apply)
    [[ $# -ge 1 ]] || { echo "usage: $0 apply FILE [extra args]" >&2; exit 2; }
    file="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    shift
    cargo run --release -- apply --file "$file" "$@"
    ;;
  eval)
    mode="${1:-routing}"
    shift || true
    cargo run --release -- eval --mode "$mode" "$@"
    ;;
  *)
    echo "unknown command: $cmd" >&2
    exit 2
    ;;
esac
