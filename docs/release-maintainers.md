# Release Maintainers

This doc is for maintainers preparing the public `Valar` repo for GitHub publication.

## Maintenance Model

The public repo is a fresh-history derived repo.

- changes land in the canonical source tree first
- the public tree is regenerated or synced from that source
- accepted public PRs are ported back into the canonical source tree before the next export

Do not publish the canonical private history directly.

## Public Release Flow

1. Land the intended change in the canonical source tree.
2. Regenerate or sync the public `Valar` tree from that canonical source.
3. Run the public release gates in the public tree:
   - `make audit-public`
   - `make validate-native`
   - `make validate-bridge` when `bridge/` changed
4. Run a private-side history scan as an advisory first-publication check for the canonical source history.
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

The audit catches private/operator content and workstation assumptions. The secret scan catches committed token-like material and obvious secret assignments.

## First Public Commit

The first public commit should be created directly in the public `Valar` repo from the exported tree. Do not attach or mirror the canonical private git history.
