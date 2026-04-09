SHELL := /bin/bash

.PHONY: \
	quickstart \
	first-clip \
	bootstrap-native \
	bootstrap-bridge \
	validate-public \
	validate-native \
	validate-bridge \
	validate-live \
	validate-live-blessed \
	validate-bridge-live \
	validate-bridge-live-blessed \
	audit-public \
	audit-and-secret-scan \
	audit-history-public \
	audit-secrets-public \
	build-cli \
	build-daemon \
	build-metallib \
	start-daemon \
	test \
	install

quickstart:
	bash ./tools/quickstart.sh

first-clip:
	bash ./tools/first_clip.sh

bootstrap-native:
	bash ./tools/bootstrap.sh native

bootstrap-bridge:
	bash ./tools/bootstrap.sh native --with-bridge

validate-public:
	bash ./tools/validate.sh --with-bridge

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
	bash ./tools/public_repo_history_scan.sh

audit-and-secret-scan:
	bash ./tools/public_repo_audit.sh
	bash ./tools/public_repo_secret_scan.sh
	bash ./tools/public_repo_history_scan.sh

audit-history-public:
	bash ./tools/public_repo_history_scan.sh

audit-secrets-public:
	bash ./tools/public_repo_secret_scan.sh

build-cli:
	swift build --package-path apps/ValarCLI

build-daemon:
	swift build --package-path apps/ValarDaemon

build-metallib:
	bash ./scripts/build_metallib.sh

start-daemon:
	swift run --package-path apps/ValarDaemon valarttsd

test:
	swift test --package-path Packages/ValarModelKit
	swift test --package-path Packages/ValarAudio
	swift test --package-path Packages/ValarPersistence
	swift test --package-path Packages/ValarCore
	swift test --package-path Packages/ValarMLX
	swift test --package-path apps/ValarCLI

install:
	swift build --package-path apps/ValarCLI --configuration release
	swift build --package-path apps/ValarDaemon --configuration release
	@mkdir -p /usr/local/bin
	cp "$$(swift build --package-path apps/ValarCLI --configuration release --show-bin-path)/valartts" /usr/local/bin/valartts
	cp "$$(swift build --package-path apps/ValarDaemon --configuration release --show-bin-path)/valarttsd" /usr/local/bin/valarttsd
	@echo "Installed valartts and valarttsd to /usr/local/bin (release build)"
