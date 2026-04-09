#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAN_ROOT="$ROOT_DIR"
EXCLUDE_FILE="$ROOT_DIR/tools/public-repo.exclude"
RULES_FILE="$ROOT_DIR/tools/public_repo_rules.sh"

usage() {
  cat <<EOF
Usage: tools/public_repo_audit.sh [--root PATH] [--exclude-file PATH]

Scans tracked or exported files for public-repo blockers:
- private workstation paths and mounted volumes
- private Claude / launchd / operator assumptions
- private snapshot and corpus scaffolding
- obvious secret/token formats and secret environment names

When run inside a git checkout, the audit uses tracked files only.
When run against an exported tree, it scans regular files under --root.
For a tracked-file secret scan, also run tools/public_repo_secret_scan.sh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      SCAN_ROOT="$2"
      shift 2
      ;;
    --exclude-file)
      EXCLUDE_FILE="$2"
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
  echo "Audit root does not exist: $SCAN_ROOT" >&2
  exit 1
fi

if [[ ! -f "$RULES_FILE" ]]; then
  echo "Rule file not found: $RULES_FILE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$RULES_FILE"

declare -a EXCLUDES=()
if [[ -f "$EXCLUDE_FILE" ]]; then
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    [[ "$line" =~ ^# ]] && continue
    EXCLUDES+=("$line")
  done < "$EXCLUDE_FILE"
fi

path_is_excluded() {
  local rel="$1"
  local pattern
  for pattern in "${EXCLUDES[@]-}"; do
    if [[ "$rel" == $pattern ]]; then
      return 0
    fi
  done
  return 1
}

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

TMP_FILE_LIST="$(mktemp)"
TMP_PATH_HITS="$(mktemp)"
TMP_HITS="$(mktemp)"
cleanup() {
  rm -f "$TMP_FILE_LIST" "$TMP_PATH_HITS" "$TMP_HITS"
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
  if path_is_excluded "$rel"; then
    continue
  fi
  printf '%s\0' "$rel" >> "$TMP_FILE_LIST"
done < <(collect_files "$SCAN_ROOT")

if [[ ! -s "$TMP_FILE_LIST" ]]; then
  echo "No files selected for audit."
  exit 0
fi

while IFS= read -r rel; do
  [[ -n "$rel" ]] || continue
  local_blocked=0
  for regex in "${PATH_BLOCK_REGEXES[@]}"; do
    if [[ "$rel" =~ $regex ]]; then
      printf '%s\n' "$rel" >> "$TMP_PATH_HITS"
      local_blocked=1
      break
    fi
  done
  [[ "$local_blocked" -eq 0 ]] || continue
done < <(tr '\0' '\n' < "$TMP_FILE_LIST")

if [[ -s "$TMP_PATH_HITS" ]]; then
  echo "Public-repo audit failed. Found blocked path names or private file surfaces:" >&2
  cat "$TMP_PATH_HITS" >&2
  exit 1
fi

declare -a RG_ARGS=()
for regex in "${CONTENT_BLOCK_REGEXES[@]}"; do
  RG_ARGS+=(-e "$regex")
done

if [[ -n "${VALAR_PUBLIC_AUDIT_EXTRA_PATTERN:-}" ]]; then
  RG_ARGS+=(-e "${VALAR_PUBLIC_AUDIT_EXTRA_PATTERN}")
fi

if ! (
  cd "$SCAN_ROOT"
  xargs -0 rg -n --no-heading "${RG_ARGS[@]}" < "$TMP_FILE_LIST"
) > "$TMP_HITS"; then
  :
fi

if [[ -s "$TMP_HITS" ]]; then
  echo "Public-repo audit failed. Found local/private markers or secret-like content:" >&2
  cat "$TMP_HITS" >&2
  exit 1
fi

echo "Public-repo audit passed for: $SCAN_ROOT"
