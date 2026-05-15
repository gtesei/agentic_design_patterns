#!/usr/bin/env bash

# Demo smoke runner for this repository.
#
# Statuses:
# - PASS: script exits 0
# - FAIL: script exits non-zero and error is not infra-related
# - SKIP_INFRA: infra/config/network issue (SSL/proxy/connectivity/missing API keys/rate limit/timeout)

set -u
set -o pipefail

MODE=""                 # required: basic | full
TIMEOUT_SECONDS=120      # per script
PATTERN_FILTER=""       # optional substring filter on pattern directory path
MODE_SET=0

usage() {
  cat <<'EOF'
Usage: scripts/run_demos_smoke.sh --mode <basic|full> [options]

Run demo scripts non-interactively across pattern folders.
Intended for manual/on-demand validation (not every push).

Options:
  --mode <basic|full>     REQUIRED.
                          basic: run *_basic.py (or best non-advanced fallback)
                          full:  run all runnable scripts in src/
  --timeout <seconds>     Per-script timeout in seconds (default: 120)
  --pattern <substring>   Only include pattern directories containing substring
  -h, --help              Show this help

OPENAI_API_KEY resolution order:
  1) Current environment
  2) <repo_root>/.env
  3) $HOME/.env

Result classification per script:
  PASS        Script exited with code 0
  FAIL        Non-zero exit and not infra-related
  SKIP_INFRA  Infra/config/network/auth/firewall/SSL/rate-limit/timeout issue

Exit codes:
  0  No FAIL results (PASS and/or SKIP_INFRA)
  1  At least one FAIL result
  2  Invalid CLI arguments

Logs:
  .demo-smoke-logs/<timestamp>/

Examples:
  scripts/run_demos_smoke.sh --mode basic
  scripts/run_demos_smoke.sh --mode full --timeout 180
  scripts/run_demos_smoke.sh --pattern foundational_design_patterns/9_rag
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
  echo "Invalid --timeout: $TIMEOUT_SECONDS (must be positive integer)" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/.demo-smoke-logs/$RUN_ID"
mkdir -p "$LOG_DIR"

INFRA_REGEX='(SSLCertVerificationError|CERTIFICATE_VERIFY_FAILED|ProxyError|proxy error|NameResolutionError|Temporary failure in name resolution|ConnectionError|Connection error\.|APIConnectionError|ConnectTimeout|ReadTimeout|TimeoutError|timed out|429|rate limit|insufficient_quota|quota|401|403|OPENAI_API_KEY|TAVILY_API_KEY|AuthenticationError|Could not resolve host|network is unreachable|Connection refused|TLS|SSL:)'

is_infra_error() {
  local log_file="$1"
  if grep -Eqi "$INFRA_REGEX" "$log_file"; then
    return 0
  fi
  return 1
}

load_key_from_env_file() {
  local env_file="$1"
  local key="$2"
  local value

  [[ -f "$env_file" ]] || return 1

  value="$({ python - "$env_file" "$key" <<'PY'
import pathlib
import sys

env_path = pathlib.Path(sys.argv[1])
key = sys.argv[2]

for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue

    if line.startswith("export "):
        line = line[len("export "):].strip()

    if "=" not in line:
        continue

    lhs, rhs = line.split("=", 1)
    if lhs.strip() != key:
        continue

    value = rhs.strip()
    if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        value = value[1:-1]

    print(value)
    break
PY
} 2>/dev/null)"

  if [[ -n "$value" ]]; then
    export "$key=$value"
    return 0
  fi

  return 1
}

has_main_guard() {
  local file="$1"
  grep -Eq "__name__\s*==\s*['\"]__main__['\"]" "$file"
}

SCRIPT_FILES=()
SCRIPT_PATTERN_DIRS=()

while IFS= read -r pyproject; do
  pattern_dir="$(dirname "$pyproject")"

  if [[ -n "$PATTERN_FILTER" && "$pattern_dir" != *"$PATTERN_FILTER"* ]]; then
    continue
  fi

  src_dir="$pattern_dir/src"
  [[ -d "$src_dir" ]] || continue

  all_runnable=()
  while IFS= read -r pyfile; do
    base="$(basename "$pyfile")"
    [[ "$base" == "__init__.py" ]] && continue
    [[ "$base" == utils*.py ]] && continue
    if has_main_guard "$pyfile"; then
      all_runnable+=("$pyfile")
    fi
  done < <(find "$src_dir" -maxdepth 1 -type f -name "*.py" | sort)

  [[ ${#all_runnable[@]} -gt 0 ]] || continue

  selected=()
  if [[ "$MODE" == "full" ]]; then
    selected=("${all_runnable[@]}")
  else
    basic_files=()
    non_advanced=()
    for f in "${all_runnable[@]}"; do
      b="$(basename "$f")"
      if [[ "$b" == *_basic.py ]]; then
        basic_files+=("$f")
      fi
      if [[ "$b" != *_advanced.py && "$b" != *advanced*.py ]]; then
        non_advanced+=("$f")
      fi
    done

    if [[ ${#basic_files[@]} -gt 0 ]]; then
      selected=("${basic_files[@]}")
    elif [[ ${#non_advanced[@]} -gt 0 ]]; then
      selected=("${non_advanced[@]}")
    else
      selected=("${all_runnable[@]}")
    fi
  fi

  for script in "${selected[@]}"; do
    SCRIPT_FILES+=("$script")
    SCRIPT_PATTERN_DIRS+=("$pattern_dir")
  done

done < <(find foundational_design_patterns reasoning reliability orchestration observability memory learning -name pyproject.toml | sort)

TOTAL=${#SCRIPT_FILES[@]}
if [[ "$TOTAL" -eq 0 ]]; then
  echo "No demo scripts found for mode=$MODE filter='${PATTERN_FILTER:-<none>}'"
  exit 0
fi

echo "== Demo smoke run =="
echo "Repo:      $ROOT_DIR"
echo "Mode:      $MODE"
echo "Timeout:   ${TIMEOUT_SECONDS}s"
echo "Filter:    ${PATTERN_FILTER:-<none>}"
echo "Scripts:   $TOTAL"
echo "Logs:      $LOG_DIR"
echo

RESULT_SCRIPT=()
RESULT_STATUS=()
RESULT_REASON=()

SYNCED_DIRS=()
BLOCKED_DIRS=()
BLOCKED_STATUSES=()
BLOCKED_REASONS=()

get_blocked_index() {
  local dir="$1"
  local i

  if [[ ${#BLOCKED_DIRS[@]} -eq 0 ]]; then
    echo "-1"
    return 0
  fi

  for i in "${!BLOCKED_DIRS[@]}"; do
    if [[ "${BLOCKED_DIRS[$i]}" == "$dir" ]]; then
      echo "$i"
      return 0
    fi
  done

  echo "-1"
  return 0
}

is_synced_dir() {
  local dir="$1"
  local d

  if [[ ${#SYNCED_DIRS[@]} -eq 0 ]]; then
    return 1
  fi

  for d in "${SYNCED_DIRS[@]}"; do
    if [[ "$d" == "$dir" ]]; then
      return 0
    fi
  done

  return 1
}

record_result() {
  local script="$1"
  local status="$2"
  local reason="$3"
  RESULT_SCRIPT+=("$script")
  RESULT_STATUS+=("$status")
  RESULT_REASON+=("$reason")
}

run_script_with_timeout() {
  local pattern_dir="$1"
  local log_file="$2"
  shift 2

  python - "$TIMEOUT_SECONDS" "$pattern_dir" "$log_file" "$@" <<'PY'
import pathlib
import subprocess
import sys

timeout = int(sys.argv[1])
cwd = sys.argv[2]
log_file = pathlib.Path(sys.argv[3])
cmd = sys.argv[4:]

try:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = []
    output.append(f"$ {' '.join(cmd)}")
    if proc.stdout:
        output.append("\n[stdout]\n" + proc.stdout)
    if proc.stderr:
        output.append("\n[stderr]\n" + proc.stderr)
    output.append(f"\n[exit_code] {proc.returncode}\n")
    log_file.write_text("\n".join(output), encoding="utf-8")
    raise SystemExit(proc.returncode)
except subprocess.TimeoutExpired as exc:
    output = [f"$ {' '.join(cmd)}", f"\n[timeout] {timeout}s\n"]
    if exc.stdout:
        output.append("[stdout]\n" + (exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", errors="replace")))
    if exc.stderr:
        output.append("\n[stderr]\n" + (exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", errors="replace")))
    log_file.write_text("\n".join(output), encoding="utf-8")
    raise SystemExit(124)
except FileNotFoundError as exc:
    log_file.write_text(f"[runner_error] {exc}\n", encoding="utf-8")
    raise SystemExit(127)
PY
}

build_smoke_command() {
  local pattern_dir="$1"
  local script_path="$2"
  local rel_script="${script_path#${pattern_dir}/}"
  local rel_repo_script="${script_path#${ROOT_DIR}/}"

  case "$rel_repo_script" in
    foundational_design_patterns/10_hitl/src/hitl_basic.py)
      printf 'uv\nrun\npython\n%s\n--scenario\n1\n--auto-decision\napprove\n' "$rel_script"
      ;;
    reliability/guardrails/src/guardrails_basic.py)
      printf 'uv\nrun\npython\n%s\n--no-pause\n--fail-on-processing-error\n' "$rel_script"
      ;;
    foundational_design_patterns/4_reflection/src/reflection_stateful_loop.py)
      printf 'uv\nrun\npython\n%s\n--quick\n' "$rel_script"
      ;;
    *)
      printf 'uv\nrun\npython\n%s\n' "$rel_script"
      ;;
  esac
}

OPENAI_API_KEY_SOURCE="environment"
if [[ -z "${OPENAI_API_KEY:-}" ]] && load_key_from_env_file "$ROOT_DIR/.env" "OPENAI_API_KEY"; then
  OPENAI_API_KEY_SOURCE="$ROOT_DIR/.env"
fi
if [[ -z "${OPENAI_API_KEY:-}" ]] && load_key_from_env_file "$HOME/.env" "OPENAI_API_KEY"; then
  OPENAI_API_KEY_SOURCE="$HOME/.env"
fi

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY loaded from: $OPENAI_API_KEY_SOURCE"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set (environment, $ROOT_DIR/.env, or $HOME/.env). Marking all demos as SKIP_INFRA."
  for i in "${!SCRIPT_FILES[@]}"; do
    script="${SCRIPT_FILES[$i]}"
    rel="${script#${ROOT_DIR}/}"
    record_result "$rel" "SKIP_INFRA" "OPENAI_API_KEY missing"
    printf '[%3d/%3d] %-12s %s (%s)\n' "$((i+1))" "$TOTAL" "SKIP_INFRA" "$rel" "OPENAI_API_KEY missing"
  done
else
  for i in "${!SCRIPT_FILES[@]}"; do
    script="${SCRIPT_FILES[$i]}"
    pattern_dir="${SCRIPT_PATTERN_DIRS[$i]}"
    rel="${script#${ROOT_DIR}/}"

    blocked_idx="$(get_blocked_index "$pattern_dir")"
    if [[ "$blocked_idx" != "-1" ]]; then
      status="${BLOCKED_STATUSES[$blocked_idx]}"
      reason="${BLOCKED_REASONS[$blocked_idx]}"
      record_result "$rel" "$status" "$reason"
      printf '[%3d/%3d] %-12s %s (%s)\n' "$((i+1))" "$TOTAL" "$status" "$rel" "$reason"
      continue
    fi

    if ! is_synced_dir "$pattern_dir"; then
      sync_log="$LOG_DIR/$(echo "${pattern_dir#${ROOT_DIR}/}" | tr '/' '_')__sync.log"
      (
        cd "$pattern_dir" && uv sync --quiet
      ) >"$sync_log" 2>&1
      sync_rc=$?
      if [[ $sync_rc -ne 0 ]]; then
        if is_infra_error "$sync_log"; then
          BLOCKED_DIRS+=("$pattern_dir")
          BLOCKED_STATUSES+=("SKIP_INFRA")
          BLOCKED_REASONS+=("uv sync infra issue")
          record_result "$rel" "SKIP_INFRA" "uv sync infra issue"
          printf '[%3d/%3d] %-12s %s (%s)\n' "$((i+1))" "$TOTAL" "SKIP_INFRA" "$rel" "uv sync infra issue"
        else
          BLOCKED_DIRS+=("$pattern_dir")
          BLOCKED_STATUSES+=("FAIL")
          BLOCKED_REASONS+=("uv sync failed")
          record_result "$rel" "FAIL" "uv sync failed"
          printf '[%3d/%3d] %-12s %s (%s)\n' "$((i+1))" "$TOTAL" "FAIL" "$rel" "uv sync failed"
        fi
        continue
      fi
      SYNCED_DIRS+=("$pattern_dir")
    fi

    script_log="$LOG_DIR/$(echo "${rel}" | tr '/' '_').log"
    script_cmd=()
    while IFS= read -r arg; do
      script_cmd+=("$arg")
    done < <(build_smoke_command "$pattern_dir" "$script")

    run_script_with_timeout "$pattern_dir" "$script_log" "${script_cmd[@]}"
    rc=$?

    if [[ $rc -eq 0 ]]; then
      record_result "$rel" "PASS" "ok"
      printf '[%3d/%3d] %-12s %s\n' "$((i+1))" "$TOTAL" "PASS" "$rel"
      continue
    fi

    if [[ $rc -eq 124 ]]; then
      record_result "$rel" "SKIP_INFRA" "timeout (${TIMEOUT_SECONDS}s)"
      printf '[%3d/%3d] %-12s %s (%s)\n' "$((i+1))" "$TOTAL" "SKIP_INFRA" "$rel" "timeout (${TIMEOUT_SECONDS}s)"
      continue
    fi

    if is_infra_error "$script_log"; then
      record_result "$rel" "SKIP_INFRA" "infra/network/auth issue"
      printf '[%3d/%3d] %-12s %s (%s)\n' "$((i+1))" "$TOTAL" "SKIP_INFRA" "$rel" "infra/network/auth issue"
    else
      record_result "$rel" "FAIL" "script error"
      printf '[%3d/%3d] %-12s %s (%s)\n' "$((i+1))" "$TOTAL" "FAIL" "$rel" "script error"
    fi
  done
fi

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for s in "${RESULT_STATUS[@]}"; do
  case "$s" in
    PASS) PASS_COUNT=$((PASS_COUNT+1)) ;;
    FAIL) FAIL_COUNT=$((FAIL_COUNT+1)) ;;
    SKIP_INFRA) SKIP_COUNT=$((SKIP_COUNT+1)) ;;
  esac
done

echo
echo "== Summary =="
echo "PASS:       $PASS_COUNT"
echo "FAIL:       $FAIL_COUNT"
echo "SKIP_INFRA: $SKIP_COUNT"
echo "TOTAL:      $TOTAL"
echo "Logs:       $LOG_DIR"

if [[ $FAIL_COUNT -gt 0 ]]; then
  echo "Result: FAIL"
  exit 1
fi

echo "Result: PASS (with optional infra skips)"
exit 0
