# Post-Publish Checklist

> **Maintainer guide.** This is a one-time setup checklist for the initial public launch. It can be removed after all items are complete.

Tasks to complete after the repository is pushed to GitHub. These require the repo to be live and cannot be done locally.

## GitHub Repository Settings

- [ ] Set repository description: `Local speech stack for Apple Silicon: TTS, ASR, forced alignment, voices, daemon, and MCP bridge.`
- [ ] Add topics: `text-to-speech`, `speech-to-text`, `forced-alignment`, `apple-silicon`, `macos`, `swift`, `mlx`, `mcp`, `local-ai`, `speech-synthesis`, `asr`, `voice-cloning`
- [ ] Upload social preview image from `assets/media/social-preview.png`
- [ ] Enable GitHub Discussions (for the Questions contact link in issue templates)
- [ ] Set default branch to `main`
- [ ] Enable branch protection on `main` with required checks: `validate-public`, `audit-and-secret-scan`
- [ ] Disable force-pushes and branch deletion for `main`

## Security

- [ ] Enable GitHub Security Advisories
- [ ] Enable secret scanning
- [ ] Enable push protection
- [ ] Enable Dependabot alerts and security updates

## Release

- [ ] Create and push tag `v1.0.0`
- [ ] Verify the release workflow creates a GitHub Release automatically
- [ ] Confirm live CI badges render correctly in README

## Community

- [ ] Decide whether to replace the temporary private conduct intake with a dedicated contact address
- [ ] Consider creating a Discord or community channel and linking from README
- [ ] Consider adding a `FUNDING.yml` once a sponsorship handle is ready

## Verification

- [ ] Open the repo in an incognito browser and walk through the README as a first-time visitor
- [ ] Click all README links to verify they resolve
- [ ] File a test issue using each template to verify the forms work
- [ ] Verify the "Cite this repository" button renders correctly from CITATION.cff
