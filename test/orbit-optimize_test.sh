#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -f -- "$WORK/liborbit.so" "$WORK/conf.json" "$WORK/cli" "$WORK/app" "$WORK/log" "$WORK/output"; rmdir -- "$WORK"' EXIT

die() { echo "FAIL: $*" >&2; exit 1; }
"${CC:-cc}" -shared -fPIC -x c /dev/null -o "$WORK/liborbit.so"
printf '{}\n' > "$WORK/conf.json"

cat > "$WORK/cli" <<'EOF'
#!/usr/bin/env bash
set -eu
[[ "$1 $2 $3" == "stat --cpu-energy --" ]] || exit 90
[[ -z "${LD_PRELOAD:-}" && -z "${ORBIT_OPTIMIZED_CONF:-}" ]] || exit 91
echo profiler >> "$TEST_LOG"
shift 3
case "${TEST_MODE:-}" in
    profiler-failure) exit 23 ;;
    no-child) exit 0 ;;
esac
"$@" || true
echo "CPU energy report"
# Mimic OPTKIT's current behavior: child failures are not propagated.
exit 0
EOF
cat > "$WORK/app" <<'EOF'
#!/usr/bin/env bash
set -eu
[[ $# -eq 2 && "$1" == "argument with spaces" && "$2" == "--flag" ]] || exit 92
if [[ -n "${ORBIT_OPTIMIZED_CONF:-}" ]]; then
    [[ "$ORBIT_OPTIMIZED_CONF" == "$TEST_CONF" ]] || exit 93
    [[ "$LD_PRELOAD" == "$TEST_LIB" && "$ORBIT_TIME_THRESHOLD" == "10ms" ]] || exit 94
    echo "app:${ORBIT_OPTIMIZE_POLICY}" >> "$TEST_LOG"
else
    [[ -z "${LD_PRELOAD:-}" ]] || exit 95
    echo app:baseline >> "$TEST_LOG"
fi
echo "application output"
exit "${TEST_APP_STATUS:-0}"
EOF
chmod +x "$WORK/cli" "$WORK/app"
export OPTKIT_CLI="$WORK/cli" ORBIT_LIB="$WORK/liborbit.so"
export TEST_LOG="$WORK/log" TEST_CONF="$WORK/conf.json" TEST_LIB="$WORK/liborbit.so"
unset LD_PRELOAD ORBIT_OPTIMIZED_CONF

run() {
    bash "$ROOT/tools/orbit-optimize.sh" -c "$WORK/conf.json" -t 10ms "$@" \
        -- "$WORK/app" "argument with spaces" --flag > "$WORK/output" 2>&1
}

run -p both --baseline
expected=$'profiler\napp:baseline\nprofiler\napp:edp\nprofiler\napp:fastest'
[[ "$(cat "$WORK/log")" == "$expected" ]] || die "wrong command order or environment"
[[ "$(grep -c '^CPU energy report$' "$WORK/output")" -eq 3 ]] || die "missing energy reports"
[[ "$(grep -c '^application output$' "$WORK/output")" -eq 3 ]] || die "missing application output"
grep -q 'profiler overhead' "$WORK/output" || die "missing timing summary"

run
[[ "$(tail -n 1 "$WORK/log")" == app:edp ]] || die "default policy is not edp"

expect_failure() {
    local expected="$1" status=0
    run -p both --baseline || status=$?
    [[ "$status" -eq "$expected" ]] || die "expected status $expected, got $status"
    ! grep -q 'wall-clock time' "$WORK/output" || die "failure printed a success summary"
}

export TEST_APP_STATUS=7
: > "$WORK/log"
expect_failure 7
[[ "$(cat "$WORK/log")" == $'profiler\napp:baseline' ]] || die "continued after application failure"
unset TEST_APP_STATUS

export TEST_MODE=profiler-failure
expect_failure 23
export TEST_MODE=no-child
expect_failure 1
grep -q 'did not record' "$WORK/output" || die "missing child status was not explained"
unset TEST_MODE

export OPTKIT_CLI="$WORK/missing"
expect_failure 1
grep -q 'optkit-cli not found' "$WORK/output" || die "missing CLI was not explained"

echo "orbit-optimize tests passed"
