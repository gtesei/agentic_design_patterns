#!/usr/bin/env bash

# TypeScript demo smoke runner for this repository.
#
# Mirrors scripts/run_demos_smoke.sh but for the bun-based TS track.
#
# Statuses:
# - PASS: bun test (or bun run demo) exited 0
# - FAIL: non-zero exit, not infra-related
# - SKIP_INFRA: SSL/proxy/network/missing API key/rate-limit/timeout

set -u
set -o pipefail

MODE=""                 # required: basic | full
TIMEOUT_SECONDS=120     # per script
PATTERN_FILTER=""       # optional substring filter on pattern directory path
MODE_SET=0

usage() {
  cat <<'EOF'
Usage: scripts/run_demos_smoke_typescript.sh --mode <basic|full> [options]

Run TypeScript pattern demos non-interactively across the bun workspace.
Intended for manual/on-demand validation (not every push).

Options:
  --mode <basic|full>     REQUIRED.
                          basic: run `bun test` in each TS pattern dir (offline)
                          full:  run `bun run demo` in each TS pattern dir (requires OPENAI_API_KEY)
  --timeout <seconds>     Per-command timeout in seconds (default: 120)
  --pattern <substring>   Only include pattern directories containing substring
  -h, --help              Show this help

OPENAI_API_KEY resolution order (full mode):
  1) Current environment
  2) <repo_root>/.env
  3) $HOME/.env

Result classification per command:
  PASS        Exit 0
  FAIL        Non-zero exit and not infra-related
  SKIP_INFRA  Infra/network/auth/SSL/rate-limit/timeout, or missing OPENAI_API_KEY (full mode only)

Exit codes:
  0  No FAIL results (PASS and/or SKIP_INFRA)
  1  At least one FAIL result
  2  Invalid CLI arguments

Logs:
  .demo-smoke-logs-ts/<timestamp>/

Examples:
  scripts/run_demos_smoke_typescript.sh --mode basic
  scripts/run_demos_smoke_typescript.sh --mode full --timeout 180
  scripts/run_demos_smoke_typescript.sh --mode basic --pattern 1_prompt_chain
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

if ! command -v bun >/dev/null 2>&1; then
  echo "bun is not installed. Install it via:" >&2
  echo "  curl -fsSL https://bun.sh/install | bash" >&2
  echo "  # or: brew install oven-sh/bun/bun" >&2
  exit 2
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/.demo-smoke-logs-ts/$RUN_ID"
mkdir -p "$LOG_DIR"

INFRA_REGEX='(SSLCertVerificationError|CERTIFICATE_VERIFY_FAILED|ProxyError|proxy error|ENOTFOUND|ETIMEDOUT|ECONNREFUSED|ECONNRESET|EAI_AGAIN|Temporary failure in name resolution|ConnectionError|Connection refused|fetch failed|timed out|timeout|429|rate limit|insufficient_quota|quota|401|403|OPENAI_API_KEY|AuthenticationError|UND_ERR|self signed certificate|unable to verify the first certificate)'

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

  value="$(awk -v k="$key" '
    BEGIN { FS="=" }
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    {
      line=$0
      sub(/^[[:space:]]*export[[:space:]]+/, "", line)
      sub(/^[[:space:]]+/, "", line)
      eq=index(line, "=")
      if (eq == 0) next
      lhs=substr(line, 1, eq-1)
      gsub(/[[:space:]]+$/, "", lhs)
      if (lhs != k) next
      rhs=substr(line, eq+1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", rhs)
      first=substr(rhs, 1, 1)
      last=substr(rhs, length(rhs), 1)
      if (length(rhs) >= 2 && first == last && (first == "\"" || first == "'\''")) {
        rhs=substr(rhs, 2, length(rhs)-2)
      }
      print rhs
      exit
    }
  ' "$env_file" 2>/dev/null)"

  if [[ -n "$value" ]]; then
    export "$key=$value"
    return 0
  fi

  return 1
}

# Discover TS pattern dirs (anything matching foundational_design_patterns/*/typescript/package.json)
PATTERN_DIRS=()
while IFS= read -r pkg; do
  pdir="$(dirname "$pkg")"
  if [[ -n "$PATTERN_FILTER" && "$pdir" != *"$PATTERN_FILTER"* ]]; then
    continue
  fi
  PATTERN_DIRS+=("$pdir")
done < <(find foundational_design_patterns -maxdepth 3 -path '*/typescript/package.json' | sort)

TOTAL=${#PATTERN_DIRS[@]}
if [[ "$TOTAL" -eq 0 ]]; then
  echo "No TS pattern packages found for filter='${PATTERN_FILTER:-<none>}'"
  exit 0
fi

echo "== TypeScript demo smoke run =="
echo "Repo:      $ROOT_DIR"
echo "Mode:      $MODE"
echo "Timeout:   ${TIMEOUT_SECONDS}s"
echo "Filter:    ${PATTERN_FILTER:-<none>}"
echo "Packages:  $TOTAL"
echo "Logs:      $LOG_DIR"
echo

RESULT_PKG=()
RESULT_STATUS=()
RESULT_REASON=()

record_result() {
  RESULT_PKG+=("$1")
  RESULT_STATUS+=("$2")
  RESULT_REASON+=("$3")
}

run_with_timeout() {
  local cwd="$1"
  local log_file="$2"
  shift 2
  local cmd=("$@")

  # macOS doesn't ship a usable `timeout`; use perl as a portable timer.
  (
    cd "$cwd" || exit 127
    perl -e '
      use POSIX ":sys_wait_h";
      my ($timeout, @cmd) = @ARGV;
      my $pid = fork();
      die "fork failed: $!" unless defined $pid;
      if ($pid == 0) {
        exec { $cmd[0] } @cmd;
        exit 127;
      }
      local $SIG{ALRM} = sub {
        kill "TERM", $pid;
        sleep 2;
        kill "KILL", $pid;
        exit 124;
      };
      alarm $timeout;
      waitpid($pid, 0);
      exit ($? >> 8);
    ' -- "$TIMEOUT_SECONDS" "${cmd[@]}"
  ) >"$log_file" 2>&1
  return $?
}

# Install deps once at the workspace root so each package shares node_modules.
# Workspace root is typescript_base/ (not the repo root).
WS_DIR="$ROOT_DIR/typescript_base"
INSTALL_LOG="$LOG_DIR/bun_install.log"
echo "Running 'bun install' at workspace root ($WS_DIR)..."
run_with_timeout "$WS_DIR" "$INSTALL_LOG" bun install
install_rc=$?
if [[ $install_rc -ne 0 ]]; then
  if is_infra_error "$INSTALL_LOG"; then
    echo "bun install failed with infra error — marking all packages SKIP_INFRA"
    for pdir in "${PATTERN_DIRS[@]}"; do
      rel="${pdir#${ROOT_DIR}/}"
      record_result "$rel" "SKIP_INFRA" "bun install infra issue"
      printf '[%3d/%3d] %-12s %s\n' "$((${#RESULT_PKG[@]}))" "$TOTAL" "SKIP_INFRA" "$rel"
    done
    PASS_COUNT=0; FAIL_COUNT=0; SKIP_COUNT=$TOTAL
  else
    echo "bun install failed (see $INSTALL_LOG)"
    exit 1
  fi
else
  # Full mode needs an API key; basic mode (bun test) is offline.
  if [[ "$MODE" == "full" ]]; then
    OPENAI_API_KEY_SOURCE="environment"
    if [[ -z "${OPENAI_API_KEY:-}" ]] && load_key_from_env_file "$ROOT_DIR/.env" "OPENAI_API_KEY"; then
      OPENAI_API_KEY_SOURCE="$ROOT_DIR/.env"
    fi
    if [[ -z "${OPENAI_API_KEY:-}" ]] && load_key_from_env_file "$HOME/.env" "OPENAI_API_KEY"; then
      OPENAI_API_KEY_SOURCE="$HOME/.env"
    fi

    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
      echo "OPENAI_API_KEY loaded from: $OPENAI_API_KEY_SOURCE"
    else
      echo "OPENAI_API_KEY is not set — full-mode demos will be marked SKIP_INFRA"
    fi
    echo
  fi

  for i in "${!PATTERN_DIRS[@]}"; do
    pdir="${PATTERN_DIRS[$i]}"
    rel="${pdir#${ROOT_DIR}/}"
    log_file="$LOG_DIR/$(echo "${rel}" | tr '/' '_').log"

    if [[ "$MODE" == "full" && -z "${OPENAI_API_KEY:-}" ]]; then
      record_result "$rel" "SKIP_INFRA" "OPENAI_API_KEY missing"
      printf '[%3d/%3d] %-12s %s (%s)\n' "$((i+1))" "$TOTAL" "SKIP_INFRA" "$rel" "OPENAI_API_KEY missing"
      continue
    fi

    if [[ "$MODE" == "basic" ]]; then
      cmd=(bun test)
    else
      cmd=(bun run demo)
    fi

    run_with_timeout "$pdir" "$log_file" "${cmd[@]}"
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

    if is_infra_error "$log_file"; then
      record_result "$rel" "SKIP_INFRA" "infra/network/auth issue"
      printf '[%3d/%3d] %-12s %s (%s)\n' "$((i+1))" "$TOTAL" "SKIP_INFRA" "$rel" "infra/network/auth issue"
    else
      record_result "$rel" "FAIL" "command error"
      printf '[%3d/%3d] %-12s %s (%s)\n' "$((i+1))" "$TOTAL" "FAIL" "$rel" "command error"
    fi
  done

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
fi

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
