#!/usr/bin/env bash

valar_public_repo_regex_escape() {
  sed 's/[][\\.^$*+?{}()|/-]/\\&/g'
}

valar_public_repo_path_block_regexes() {
  cat <<'EOF'
(^|/)docs/(archive|roadmap|spikes)/
(^|/)docs/benchmark-results-[^/]+\.md$
(^|/)docs/[A-Z0-9_]*PORT_SPEC\.md$
(^|/)docs/analysis-[^/]+\.md$
(^|/)\.env(\..*)?$
(^|/)\.netrc$
(^|/)\.npmrc$
(^|/)\.pypirc$
(^|/)\.aws/(credentials|config)$
(^|/)(id_rsa|id_dsa|id_ecdsa|id_ed25519)$
(^|/)[^/]*\.(pem|p12|pfx|key)$
(^|/)artifacts/
(^|/)bridge/node_modules/
(^|/)tools/private_.*\.sh$
(^|/)\.valartts-public-home(/|$)
(^|/)\.valartts-home(/|$)
(^|/)[^/]+\.valarproject(/|$)
(^|/)valar\.db$
EOF
}

valar_public_repo_content_block_regexes() {
  cat <<'EOF'
/Users/[A-Za-z0-9._-]+/
/Volumes/[A-Za-z0-9._-]+/
ghp_[A-Za-z0-9]{36}
gh[osru]_[A-Za-z0-9_]{36,}
github_pat_[A-Za-z0-9_]{20,}
hf_[A-Za-z0-9]{32,}
sk-[A-Za-z0-9_-]{20,}
sk-proj-[A-Za-z0-9_-]{20,}
sk-ant-[A-Za-z0-9_-]{20,}
xox[baprs]-[A-Za-z0-9-]+
xapp-[A-Za-z0-9-]+
AKIA[0-9A-Z]{16}
ASIA[0-9A-Z]{16}
AIza[0-9A-Za-z_-]{35}
[A-Z][A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[[:space:]]*[:=][[:space:]]*["']?[^[:space:]'"]{8,}
-----BEGIN [A-Z ]*PRIVATE KEY-----
EOF
}

valar_public_repo_secret_block_regexes() {
  cat <<'EOF'
ghp_[A-Za-z0-9]{36}
gh[osru]_[A-Za-z0-9_]{36,}
github_pat_[A-Za-z0-9_]{20,}
hf_[A-Za-z0-9]{32,}
sk-[A-Za-z0-9_-]{20,}
sk-proj-[A-Za-z0-9_-]{20,}
sk-ant-[A-Za-z0-9_-]{20,}
xox[baprs]-[A-Za-z0-9-]+
xapp-[A-Za-z0-9-]+
AKIA[0-9A-Z]{16}
ASIA[0-9A-Z]{16}
AIza[0-9A-Za-z_-]{35}
[A-Z][A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[[:space:]]*[:=][[:space:]]*["']?[^[:space:]'"]{8,}
-----BEGIN [A-Z ]*PRIVATE KEY-----
EOF
}

valar_public_repo_history_block_regexes() {
  cat <<'EOF'
/Users/[A-Za-z0-9._-]+/
/Volumes/[A-Za-z0-9._-]+/
ghp_[A-Za-z0-9]{36}
gh[osru]_[A-Za-z0-9_]{36,}
github_pat_[A-Za-z0-9_]{20,}
hf_[A-Za-z0-9]{32,}
sk-[A-Za-z0-9_-]{20,}
sk-proj-[A-Za-z0-9_-]{20,}
sk-ant-[A-Za-z0-9_-]{20,}
xox[baprs]-[A-Za-z0-9-]+
xapp-[A-Za-z0-9-]+
AKIA[0-9A-Z]{16}
ASIA[0-9A-Z]{16}
AIza[0-9A-Za-z_-]{35}
[A-Z][A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[[:space:]]*[:=][[:space:]]*["']?[^[:space:]'"]{8,}
-----BEGIN [A-Z ]*PRIVATE KEY-----
EOF
}

valar_public_repo_local_block_regexes() {
  local account_name escaped_name volume_path volume_name escaped_volume
  account_name="$(id -un 2>/dev/null || true)"
  case "$account_name" in
    ""|root|runner) ;;
    *)
      escaped_name="$(printf '%s' "$account_name" | valar_public_repo_regex_escape)"
      printf '/Users/%s/\n' "$escaped_name"
      ;;
  esac

  if [[ -d /Volumes ]]; then
    while IFS= read -r volume_path; do
      volume_name="${volume_path##*/}"
      case "$volume_name" in
        ""|Macintosh\ HD|Preboot|Recovery|VM) continue ;;
      esac
      escaped_volume="$(printf '%s' "$volume_name" | valar_public_repo_regex_escape)"
      printf '/Volumes/%s/\n' "$escaped_volume"
    done < <(find /Volumes -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null)
  fi

  if [[ -n "${VALAR_PUBLIC_REPO_SENTINELS_FILE:-}" && -f "$VALAR_PUBLIC_REPO_SENTINELS_FILE" ]]; then
    sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "$VALAR_PUBLIC_REPO_SENTINELS_FILE"
  fi
}
