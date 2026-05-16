# Release Maintainers

This doc is for maintainers preparing the public `Valar` repo for GitHub publication.

## Maintenance Model

The public repo is a fresh-history publication repo.

- changes should be staged in a clean source tree first
- the public tree should be regenerated or synced from that clean source
- accepted public PRs should be carried forward into the source tree before the next publication update

Do not publish non-public source history directly.

## Public Release Flow

1. Land the intended change in a clean source tree.
2. Regenerate or sync the public `Valar` tree from that source.
3. Run the public release gates in the public tree:
   - `make audit-and-secret-scan`
   - `make validate-public`
   - `make validate-bridge` when `bridge/` changed
   - `python3 tools/generate_launch_media.py` when launch-facing visuals need refresh
4. If content was imported from another tree, run a history scan before publication.
5. Verify the public repo worktree is clean:
   - no `bridge/node_modules`
   - no build outputs
   - no temporary validation artifacts
6. Commit or update the public repo from the exported tree only.

See [github-repo-settings.md](./github-repo-settings.md) for the repo-level settings to enable before opening the project to public contributors.

## GitHub Settings Checklist

Enable these before opening the repo to public contributors:

- GitHub Security Advisories
- secret scanning
- push protection
- Dependabot alerts and security updates
- branch protection for `main`
- required validation checks before merge

Required checks:

- `validate-public`
- `audit-and-secret-scan`

Optional follow-up:

- CodeQL for Swift and TypeScript once the repo is live and stable

## Public Security Gates

The public repo should keep both of these checks green:

- `tools/public_repo_audit.sh`
- `tools/public_repo_secret_scan.sh`
- `tools/public_repo_history_scan.sh`

The audit catches private/operator content and workstation assumptions in the current tree. The secret scan catches committed token-like material and obvious secret assignments in the current tree. The history scan checks earlier commits before publication.

## First Public Commit

The first public commit should be created directly in the public `Valar` repo from the exported tree. Do not attach or mirror non-public git history.
