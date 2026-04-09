#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAN_ROOT="$ROOT_DIR"
RULES_FILE="$ROOT_DIR/tools/public_repo_rules.sh"

usage() {
  cat <<EOF
Usage: tools/public_repo_secret_scan.sh [--root PATH]

Scans tracked or exported files for obvious secret values and committed secret
assignments. This is the second public release gate after tools/public_repo_audit.sh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      SCAN_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$SCAN_ROOT" ]]; then
  echo "Secret scan root does not exist: $SCAN_ROOT" >&2
  exit 1
fi

if [[ ! -f "$RULES_FILE" ]]; then
  echo "Rule file not found: $RULES_FILE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$RULES_FILE"

collect_files() {
  local root="$1"
  if git -C "$root" rev-parse --show-toplevel >/dev/null 2>&1 \
    && git -C "$root" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$root" ls-files
  else
    (
      cd "$root"
      find . \
        \( -name .git -o -name .build -o -name .build-cache -o -name .swiftpm -o -name node_modules \) -prune -o \
        -type f -print | sed 's#^\./##'
    )
  fi
}

declare -a SECRET_BLOCK_REGEXES=()
while IFS= read -r regex; do
  [[ -n "$regex" ]] || continue
  SECRET_BLOCK_REGEXES+=("$regex")
done < <(valar_public_repo_secret_block_regexes)

TMP_FILE_LIST="$(mktemp)"
TMP_HITS="$(mktemp)"
cleanup() {
  rm -f "$TMP_FILE_LIST" "$TMP_HITS"
}
trap cleanup EXIT

while IFS= read -r rel; do
  [[ -n "$rel" ]] || continue
  [[ -f "$SCAN_ROOT/$rel" ]] || continue
  case "$rel" in
    tools/public_repo_audit.sh|tools/public_repo_rules.sh|tools/public_repo_secret_scan.sh)
      continue
      ;;
  esac
  printf '%s\0' "$rel" >> "$TMP_FILE_LIST"
done < <(collect_files "$SCAN_ROOT")

if [[ ! -s "$TMP_FILE_LIST" ]]; then
  echo "No files selected for secret scan."
  exit 0
fi

declare -a RG_ARGS=()
for regex in "${SECRET_BLOCK_REGEXES[@]}"; do
  RG_ARGS+=(-e "$regex")
done

if [[ -n "${VALAR_PUBLIC_SECRET_SCAN_EXTRA_PATTERN:-}" ]]; then
  RG_ARGS+=(-e "${VALAR_PUBLIC_SECRET_SCAN_EXTRA_PATTERN}")
fi

if ! (
  cd "$SCAN_ROOT"
  xargs -0 rg -n --no-heading "${RG_ARGS[@]}" < "$TMP_FILE_LIST"
) > "$TMP_HITS"; then
  :
fi

if [[ -s "$TMP_HITS" ]]; then
  echo "Public-repo secret scan failed. Found committed secret-like content:" >&2
  cat "$TMP_HITS" >&2
  exit 1
fi

echo "Public-repo secret scan passed for: $SCAN_ROOT"
