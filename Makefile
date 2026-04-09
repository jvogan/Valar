SHELL := /bin/bash

.PHONY: \
	quickstart \
	first-clip \
	bootstrap-native \
	bootstrap-bridge \
	validate-native \
	validate-bridge \
	validate-live \
	validate-live-blessed \
	validate-bridge-live \
	validate-bridge-live-blessed \
	audit-public \
	audit-secrets-public \
	build-cli \
	build-daemon \
	build-metallib

quickstart:
	bash ./tools/quickstart.sh

first-clip:
	bash ./tools/first_clip.sh

bootstrap-native:
	bash ./tools/bootstrap.sh native

bootstrap-bridge:
	bash ./tools/bootstrap.sh native --with-bridge

validate-native:
	bash ./tools/validate.sh

validate-bridge:
	bash ./tools/validate.sh --with-bridge

validate-live:
	bash ./tools/validate.sh --live

validate-live-blessed:
	bash ./tools/validate.sh --live-blessed

validate-bridge-live:
	bash ./tools/validate.sh --with-bridge --live

validate-bridge-live-blessed:
	bash ./tools/validate.sh --with-bridge --live-blessed

audit-public:
	bash ./tools/public_repo_audit.sh
	bash ./tools/public_repo_secret_scan.sh

audit-secrets-public:
	bash ./tools/public_repo_secret_scan.sh

build-cli:
	swift build --package-path apps/ValarCLI

build-daemon:
	swift build --package-path apps/ValarDaemon

build-metallib:
	bash ./scripts/build_metallib.sh
