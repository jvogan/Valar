#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAN_ROOT="$ROOT_DIR"
RULES_FILE="$ROOT_DIR/tools/public_repo_rules.sh"

usage() {
  cat <<EOF
Usage: tools/public_repo_history_scan.sh [--root PATH]

Scans git history for blocked private path names plus secret-like or local-only
content that may have existed in older commits.
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
  echo "History scan root does not exist: $SCAN_ROOT" >&2
  exit 1
fi

if [[ ! -f "$RULES_FILE" ]]; then
  echo "Rule file not found: $RULES_FILE" >&2
  exit 1
fi

if ! git -C "$SCAN_ROOT" rev-parse --show-toplevel >/dev/null 2>&1 \
  || ! git -C "$SCAN_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "No git history available for public history scan."
  exit 0
fi

# shellcheck source=/dev/null
source "$RULES_FILE"

declare -a PATH_BLOCK_REGEXES=()
while IFS= read -r regex; do
  [[ -n "$regex" ]] || continue
  PATH_BLOCK_REGEXES+=("$regex")
done < <(valar_public_repo_path_block_regexes)

declare -a CONTENT_BLOCK_REGEXES=()
while IFS= read -r regex; do
  [[ -n "$regex" ]] || continue
  CONTENT_BLOCK_REGEXES+=("$regex")
done < <(valar_public_repo_content_block_regexes)

declare -a SECRET_BLOCK_REGEXES=()
while IFS= read -r regex; do
  [[ -n "$regex" ]] || continue
  SECRET_BLOCK_REGEXES+=("$regex")
done < <(valar_public_repo_secret_block_regexes)

TMP_PATH_HITS="$(mktemp)"
TMP_CONTENT_HITS="$(mktemp)"
cleanup() {
  rm -f "$TMP_PATH_HITS" "$TMP_CONTENT_HITS"
}
trap cleanup EXIT

declare -a GREP_ARGS=()
for regex in "${CONTENT_BLOCK_REGEXES[@]}"; do
  GREP_ARGS+=(-e "$regex")
done
for regex in "${SECRET_BLOCK_REGEXES[@]}"; do
  GREP_ARGS+=(-e "$regex")
done

while IFS= read -r rev; do
  [[ -n "$rev" ]] || continue

  while IFS= read -r rel; do
    [[ -n "$rel" ]] || continue
    case "$rel" in
      tools/public_repo_audit.sh|tools/public_repo_secret_scan.sh|tools/public_repo_history_scan.sh|tools/public_repo_rules.sh)
        continue
        ;;
    esac
    for regex in "${PATH_BLOCK_REGEXES[@]}"; do
      if [[ "$rel" =~ $regex ]]; then
        printf '%s:%s\n' "$rev" "$rel" >> "$TMP_PATH_HITS"
        break
      fi
    done
  done < <(git -C "$SCAN_ROOT" ls-tree -r --name-only "$rev")

  if ! (
    cd "$SCAN_ROOT"
    git grep -n --no-heading "${GREP_ARGS[@]}" "$rev" -- . \
      ':(exclude)tools/public_repo_audit.sh' \
      ':(exclude)tools/public_repo_secret_scan.sh' \
      ':(exclude)tools/public_repo_history_scan.sh' \
      ':(exclude)tools/public_repo_rules.sh'
  ) >> "$TMP_CONTENT_HITS"; then
    :
  fi
done < <(git -C "$SCAN_ROOT" rev-list --all)

if [[ -s "$TMP_PATH_HITS" ]]; then
  echo "Public history scan failed. Found blocked path names in git history:" >&2
  sort -u "$TMP_PATH_HITS" >&2
  exit 1
fi

if [[ -s "$TMP_CONTENT_HITS" ]]; then
  echo "Public history scan failed. Found private or secret-like content in git history:" >&2
  sort -u "$TMP_CONTENT_HITS" >&2
  exit 1
fi

echo "Public history scan passed for: $SCAN_ROOT"
