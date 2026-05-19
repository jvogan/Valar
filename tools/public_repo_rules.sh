#!/usr/bin/env bash

valar_public_repo_path_block_regexes() {
  cat <<'EOF'
(^|/)\.github/workflows/parity\.yml$
(^|/)CLAUDE\.md$
(^|/)docs/private-local-ops\.md$
(^|/)docs/development\.md$
(^|/)docs/bridge-plugin-voice-setup\.md$
(^|/)docs/voxtral-operator-notes\.md$
(^|/)docs/model-distribution-plan\.md$
(^|/)docs/model-licenses\.md$
(^|/)docs/archive/
(^|/)docs/roadmap/
(^|/)docs/spikes/
(^|/)docs/benchmark-results-[^/]+\.md$
(^|/)docs/qwen-production-playbook\.md$
(^|/)docs/runtime-ownership\.md$
(^|/)docs/(TADA_PORT_SPEC|VOXTRAL_PORT_SPEC|VIBEVOICE_PORT_SPEC)\.md$
(^|/)docs/tada-autoregressive-loop-spec\.md$
(^|/)docs/tada_checkpoint_manifest\.json$
(^|/)docs/analysis-[^/]+\.md$
(^|/)docs/agent-loop-playbook\.md$
(^|/)docs/repo-lanes\.md$
(^|/)\.env(\..*)?$
(^|/)\.netrc$
(^|/)\.npmrc$
(^|/)\.pypirc$
(^|/)\.aws/(credentials|config)$
(^|/)(id_rsa|id_dsa|id_ecdsa|id_ed25519)$
(^|/)[^/]*\.(pem|p12|pfx|key)$
(^|/)artifacts/
(^|/)mlx_audio/
(^|/)MANIFEST\.in$
(^|/)pytest\.ini$
(^|/)uv\.lock$
(^|/)pyproject\.toml$
(^|/)apps/ValarMCP/
(^|/)Packages/ValarCompat/
(^|/)bridge/node_modules/
(^|/)bridge/plugins/
(^|/)bridge/src/tools/valar_prepare_channel_reply\.ts$
(^|/)examples/bible-audiobook/
(^|/)scripts/tada/
(^|/)apps/ValarTTSMac/Sources/ValarTTSMacApp/Diagnostics/GoldenCorpusView\.swift$
(^|/)apps/ValarTTSMac/Sources/ValarTTSMacApp/Diagnostics/LegacyImportView\.swift$
(^|/)scripts/com\..*ssk-backup-monitor\.plist$
(^|/)scripts/com\.valartts\.daemon\.plist\.template$
(^|/)scripts/install_valarttsd_launchd\.sh$
(^|/)scripts/monitor_ssk_backup\.sh$
(^|/)scripts/local_services\.sh$
(^|/)scripts/run_valarttsd\.sh$
(^|/)scripts/run_mlx_tts_api\.sh$
(^|/)scripts/run_mlx_tts_ui\.sh$
(^|/)scripts/kill_stale_telegram\.sh$
(^|/)scripts/demo_voice_loop\.sh$
(^|/)scripts/demo_local_agent_flow\.sh$
(^|/)tools/private_snapshot\.sh$
(^|/)tools/private_.*\.sh$
(^|/)tools/golden_corpus\.sh$
(^|/)tools/sync_public_repo\.sh$
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
SSK_Symphony
Library/LaunchAgents
private snapshot
snapshot-and-corpus
Capture private snapshot
private-local-ops
LINEAR_API_KEY
\.claude/
claude-plugins-official
Claude Code
forwardToClaude
claude-fast
claude-quality
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
(OPENAI|ANTHROPIC|LINEAR|TELEGRAM|SLACK|GITHUB|HF|HUGGINGFACE|HUGGING_FACE|RUNPOD|AWS|GOOGLE|GEMINI|DISCORD|ELEVENLABS)[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[[:space:]]*[:=][[:space:]]*["']?[^[:space:]'"]{8,}
-----BEGIN [A-Z ]*PRIVATE KEY-----
-----BEGIN OPENSSH PRIVATE KEY-----
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
(OPENAI|ANTHROPIC|LINEAR|TELEGRAM|SLACK|GITHUB|HF|HUGGINGFACE|HUGGING_FACE|RUNPOD|AWS|GOOGLE|GEMINI|DISCORD|ELEVENLABS)[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[[:space:]]*[:=][[:space:]]*["']?[^[:space:]'"]{8,}
-----BEGIN [A-Z ]*PRIVATE KEY-----
-----BEGIN OPENSSH PRIVATE KEY-----
EOF
}

valar_public_repo_history_block_regexes() {
  cat <<'EOF'
/Users/[A-Za-z0-9._-]+/
/Volumes/[A-Za-z0-9._-]+/
SSK_Symphony
Library/LaunchAgents
private snapshot
snapshot-and-corpus
Capture private snapshot
private-local-ops
LINEAR_API_KEY
\.claude/
claude-plugins-official
Claude Code
forwardToClaude
claude-fast
claude-quality
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
(OPENAI|ANTHROPIC|LINEAR|TELEGRAM|SLACK|GITHUB|HF|HUGGINGFACE|HUGGING_FACE|RUNPOD|AWS|GOOGLE|GEMINI|DISCORD|ELEVENLABS)[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[[:space:]]*[:=][[:space:]]*["']?[^[:space:]'"]{8,}
-----BEGIN [A-Z ]*PRIVATE KEY-----
-----BEGIN OPENSSH PRIVATE KEY-----
EOF
}
