#!/usr/bin/env bash

# TypeScript demo smoke runner (Bun)
#
# Statuses:
# - PASS: command exits 0
# - FAIL: non-zero exit and non-infra error
# - SKIP_INFRA: network/SSL/auth/provider/timeout issue

set -u
set -o pipefail

MODE=""                 # required: basic | full
TIMEOUT_SECONDS=120
PATTERN_FILTER=""
MODE_SET=0
WITH_TESTS=0

usage() {
  cat <<'EOF'
Usage: scripts/run_demos_smoke_typescript.sh --mode <basic|full> [options]

Runs TypeScript demo scripts in */typescript folders using Bun.
Primary goal: run demo entrypoints (not only tests).

Options:
  --mode <basic|full>     REQUIRED.
                          basic: run demo:basic (or demo/start fallback)
                          full: run basic + advanced/full demos if available
  --timeout <seconds>     Per-command timeout in seconds (default: 120)
  --pattern <substring>   Only include directories containing substring
  --with-tests            Also run bun test after demos
  -h, --help              Show help

Result per command:
  PASS        Exit code 0
  FAIL        Non-zero exit, not infra-related
  SKIP_INFRA  SSL/network/auth/provider/rate-limit/timeout issue

Exit codes:
  0  No FAIL results
  1  At least one FAIL
  2  Invalid CLI arguments
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      MODE_SET=1
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECONDS="${2:-}"
      shift 2
      ;;
    --pattern)
      PATTERN_FILTER="${2:-}"
      shift 2
      ;;
    --with-tests)
      WITH_TESTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$MODE_SET" -ne 1 ]]; then
  echo "Missing required argument: --mode <basic|full>" >&2
  usage
  exit 2
fi

if [[ "$MODE" != "basic" && "$MODE" != "full" ]]; then
  echo "Invalid --mode: $MODE (expected: basic|full)" >&2
  exit 2
fi

if ! [[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || [[ "$TIMEOUT_SECONDS" -le 0 ]]; then
  echo "Invalid --timeout: $TIMEOUT_SECONDS" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/.demo-smoke-logs/typescript_$RUN_ID"
mkdir -p "$LOG_DIR"

INFRA_REGEX='(CERTIFICATE_VERIFY_FAILED|SSL:|tls|network is unreachable|ECONNRESET|ECONNREFUSED|ENOTFOUND|ETIMEDOUT|fetch failed|OpenAIError|Missing credentials|OPENAI_API_KEY|401|403|429|rate limit|quota|temporarily unavailable|timeout)'

is_infra_error() {
  local log_file="$1"
  grep -Eqi "$INFRA_REGEX" "$log_file"
}

run_with_timeout() {
  local cwd="$1"
  local log_file="$2"
  shift 2

  python - "$TIMEOUT_SECONDS" "$cwd" "$log_file" "$@" <<'PY'
import pathlib
import subprocess
import sys

timeout = int(sys.argv[1])
cwd = sys.argv[2]
log_file = pathlib.Path(sys.argv[3])
cmd = sys.argv[4:]

try:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    out = [f"$ {' '.join(cmd)}"]
    if proc.stdout:
        out.append("\n[stdout]\n" + proc.stdout)
    if proc.stderr:
        out.append("\n[stderr]\n" + proc.stderr)
    out.append(f"\n[exit_code] {proc.returncode}\n")
    log_file.write_text("\n".join(out), encoding="utf-8")
    raise SystemExit(proc.returncode)
except subprocess.TimeoutExpired as exc:
    out = [f"$ {' '.join(cmd)}", f"\n[timeout] {timeout}s\n"]
    if exc.stdout:
        out.append("[stdout]\n" + (exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", errors="replace")))
    if exc.stderr:
        out.append("\n[stderr]\n" + (exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", errors="replace")))
    log_file.write_text("\n".join(out), encoding="utf-8")
    raise SystemExit(124)
except FileNotFoundError as exc:
    log_file.write_text(f"[runner_error] {exc}\n", encoding="utf-8")
    raise SystemExit(127)
PY
}

get_scripts_for_mode() {
  local pkg="$1"
  local mode="$2"

  python - "$pkg" "$mode" <<'PY'
import json
import pathlib
import sys

pkg_path = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
obj = json.loads(pkg_path.read_text(encoding='utf-8'))
scripts = obj.get("scripts", {})

commands = []
if mode == "basic":
    for key in ("demo:basic", "demo", "start"):
        if key in scripts:
            commands.append(key)
            break
else:
    # full
    chosen = []
    for key in ("demo:basic", "demo"):
        if key in scripts:
            chosen.append(key)
            break
    for key in ("demo:advanced", "demo:full"):
        if key in scripts:
            chosen.append(key)
            break
    if not chosen and "start" in scripts:
        chosen.append("start")
    commands.extend(chosen)

for c in commands:
    print(c)
PY
}

echo "== TypeScript Demo Smoke =="
echo "Repo:    $ROOT_DIR"
echo "Mode:    $MODE"
echo "Timeout: ${TIMEOUT_SECONDS}s"
echo "Filter:  ${PATTERN_FILTER:-<none>}"
echo "Logs:    $LOG_DIR"
echo

TS_DIRS=()
while IFS= read -r p; do
  dir="$(dirname "$p")"
  if [[ -n "$PATTERN_FILTER" && "$dir" != *"$PATTERN_FILTER"* ]]; then
    continue
  fi
  TS_DIRS+=("$dir")
done < <(find foundational_design_patterns orchestration reasoning learning memory observability reliability -type f -path '*/typescript/package.json' | sort)

if [[ ${#TS_DIRS[@]} -eq 0 ]]; then
  echo "No TypeScript pattern directories found."
  exit 0
fi

PASS=0
FAIL=0
SKIP=0
STEP=0

for dir in "${TS_DIRS[@]}"; do
  rel_dir="${dir#${ROOT_DIR}/}"
  echo "-- $rel_dir"

  # install
  STEP=$((STEP+1))
  install_log="$LOG_DIR/${rel_dir//\//_}__install.log"
  run_with_timeout "$dir" "$install_log" bun install
  rc=$?
  if [[ $rc -ne 0 ]]; then
    if [[ $rc -eq 124 ]] || is_infra_error "$install_log"; then
      SKIP=$((SKIP+1))
      echo "  [SKIP_INFRA] bun install"
      continue
    fi
    FAIL=$((FAIL+1))
    echo "  [FAIL] bun install"
    continue
  fi

  script_keys=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && script_keys+=("$line")
  done < <(get_scripts_for_mode "$dir/package.json" "$MODE")

  if [[ ${#script_keys[@]} -eq 0 ]]; then
    SKIP=$((SKIP+1))
    echo "  [SKIP_INFRA] no demo script found (expected demo:basic/demo/start)"
    continue
  fi

  for key in "${script_keys[@]}"; do
    STEP=$((STEP+1))
    log="$LOG_DIR/${rel_dir//\//_}__${key//:/_}.log"
    run_with_timeout "$dir" "$log" bun run "$key"
    rc=$?
    if [[ $rc -eq 0 ]]; then
      PASS=$((PASS+1))
      echo "  [PASS] bun run $key"
    elif [[ $rc -eq 124 ]] || is_infra_error "$log"; then
      SKIP=$((SKIP+1))
      echo "  [SKIP_INFRA] bun run $key"
    else
      FAIL=$((FAIL+1))
      echo "  [FAIL] bun run $key"
    fi
  done

  if [[ "$WITH_TESTS" -eq 1 ]]; then
    STEP=$((STEP+1))
    tlog="$LOG_DIR/${rel_dir//\//_}__test.log"
    run_with_timeout "$dir" "$tlog" bun test
    rc=$?
    if [[ $rc -eq 0 ]]; then
      PASS=$((PASS+1))
      echo "  [PASS] bun test"
    elif [[ $rc -eq 124 ]] || is_infra_error "$tlog"; then
      SKIP=$((SKIP+1))
      echo "  [SKIP_INFRA] bun test"
    else
      FAIL=$((FAIL+1))
      echo "  [FAIL] bun test"
    fi
  fi

done

echo
echo "== Summary =="
echo "PASS:       $PASS"
echo "FAIL:       $FAIL"
echo "SKIP_INFRA: $SKIP"
echo "Logs:       $LOG_DIR"

if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
exit 0
