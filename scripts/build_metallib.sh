#!/usr/bin/env bash
# build_metallib.sh — Compile MLX Metal shaders for SPM-built binaries.
#
# SPM's `swift build` does not compile .metal files into .metallib.
# This script does it manually so inference works without Xcode.
#
# Usage:
#   bash scripts/build_metallib.sh [scratch-path]
#
# After running, the metallib is placed next to the valartts binary.
# Extra search/output paths can be supplied with:
#   VALARTTS_METALLIB_EXTRA_SCRATCH_CANDIDATES=/path/one:/path/two
#   VALARTTS_METALLIB_EXTRA_OUTPUT_DIRS=/path/one:/path/two
#   VALARTTS_METALLIB_FALLBACK_PATH=/path/to/mlx.metallib
# If you already have mlx.metallib from another MLX install or checkout,
# you can point VALARTTS_METALLIB_FALLBACK_PATH at it to avoid rebuilding.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_SCRATCH_CANDIDATES=(
  "/tmp/valar-build-core"
  "$REPO_ROOT/apps/ValarDaemon/.build"
  "$REPO_ROOT/apps/ValarCLI/.build"
  "$REPO_ROOT/Packages/ValarCore/.build"
  "$REPO_ROOT/.build-cache"
)
BUILD_DIR="/tmp/metallib-build"
DEFAULT_OUTPUT_DIRS=(
  "$REPO_ROOT/apps/ValarCLI/.build/arm64-apple-macosx/debug"
  "$REPO_ROOT/apps/ValarDaemon/.build/arm64-apple-macosx/debug"
  "$REPO_ROOT/Packages/mlx-audio-swift-valar/.build/arm64-apple-macosx/debug"
)
FALLBACK_METALLIB_CANDIDATES=(
  "$REPO_ROOT/.venv/lib/python3.10/site-packages/mlx/lib/mlx.metallib"
  "$REPO_ROOT/.build-cache/arm64-apple-macosx/debug/mlx.metallib"
  "$REPO_ROOT/.build-stable/mlx.metallib"
)

append_path_list() {
  local value="$1"
  local -n target_ref="$2"
  local item
  local IFS=':'
  read -r -a extra_paths <<< "$value"
  for item in "${extra_paths[@]}"; do
    [[ -n "$item" ]] || continue
    target_ref+=("$item")
  done
}

if [[ -n "${VALARTTS_METALLIB_EXTRA_SCRATCH_CANDIDATES:-}" ]]; then
  append_path_list "$VALARTTS_METALLIB_EXTRA_SCRATCH_CANDIDATES" DEFAULT_SCRATCH_CANDIDATES
fi

if [[ -n "${VALARTTS_METALLIB_EXTRA_OUTPUT_DIRS:-}" ]]; then
  append_path_list "$VALARTTS_METALLIB_EXTRA_OUTPUT_DIRS" DEFAULT_OUTPUT_DIRS
fi

install_metallib() {
  local metallib_path="$1"
  local installed=0
  local target_dir
  local nested_dir

  copy_into_dir() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0
    cp "$metallib_path" "$dir/mlx.metallib"
    cp "$metallib_path" "$dir/default.metallib"
    echo "Installed: $dir/mlx.metallib"
    echo "Installed: $dir/default.metallib"
    installed=1
  }

  for target_dir in "${DEFAULT_OUTPUT_DIRS[@]}"; do
    [[ -d "$target_dir" ]] || continue
    copy_into_dir "$target_dir"
    while IFS= read -r nested_dir; do
      [[ -n "$nested_dir" ]] || continue
      copy_into_dir "$nested_dir"
    done < <(find "$target_dir" -type d -path '*/Contents/MacOS' 2>/dev/null | sort)
  done

  if [[ "$installed" -eq 0 ]]; then
    echo "No supported build output directory found."
    echo "Metallib available at: $metallib_path"
    echo "Copy it next to your valartts or valarttsd binary manually."
  fi
}

find_fallback_metallib() {
  find_python_mlx_metallib() {
    local python_bin
    local candidate

    for python_bin in python3 python; do
      command -v "$python_bin" >/dev/null 2>&1 || continue
      candidate="$("$python_bin" - <<'PY' 2>/dev/null || true
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("mlx")
if spec is None or spec.origin is None:
    raise SystemExit(0)

candidate = Path(spec.origin).resolve().parent / "lib" / "mlx.metallib"
if candidate.is_file():
    print(candidate)
PY
)"
      if [[ -n "$candidate" && -f "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done

    return 1
  }

  if [[ -n "${VALARTTS_METALLIB_FALLBACK_PATH:-}" ]]; then
    if [[ -f "${VALARTTS_METALLIB_FALLBACK_PATH}" ]]; then
      printf '%s\n' "${VALARTTS_METALLIB_FALLBACK_PATH}"
      return 0
    fi

    echo "Warning: VALARTTS_METALLIB_FALLBACK_PATH does not exist: ${VALARTTS_METALLIB_FALLBACK_PATH}" >&2
  fi

  local candidate
  for candidate in "${FALLBACK_METALLIB_CANDIDATES[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if candidate="$(find_python_mlx_metallib)"; then
    printf '%s\n' "$candidate"
    return 0
  fi

  return 1
}

resolve_scratch() {
  local candidate

  if [[ $# -gt 0 && -n "$1" ]]; then
    printf '%s\n' "$1"
    return 0
  fi

  for candidate in "${DEFAULT_SCRATCH_CANDIDATES[@]}"; do
    if [[ -d "$candidate/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

SCRATCH="$(resolve_scratch "${1:-}")" || {
  echo "Error: Could not locate an mlx-swift checkout with generated Metal shaders."
  echo "Tried:"
  printf '  %s\n' "${DEFAULT_SCRATCH_CANDIDATES[@]}"
  echo "Run 'swift build --package-path apps/ValarDaemon' or pass an explicit scratch path."
  exit 1
}

MLX_CHECKOUT="$SCRATCH/checkouts/mlx-swift/Source/Cmlx"
GENERATED_DIR="$MLX_CHECKOUT/mlx-generated/metal"
BACKEND_DIR="$MLX_CHECKOUT/mlx/mlx/backend/metal/kernels"
BIN_DIR="$SCRATCH/arm64-apple-macosx/debug"
DEFAULT_OUTPUT_DIRS=(
  "$BIN_DIR"
  "${DEFAULT_OUTPUT_DIRS[@]}"
)

mkdir -p "$BUILD_DIR"

# Include paths for Metal headers:
# - Generated shaders reference headers in their own dir
# - Backend kernels use "mlx/backend/metal/kernels/..." includes relative to the mlx source root
MLX_SRC_ROOT="$MLX_CHECKOUT/mlx"
INCLUDE_FLAGS=(-I "$GENERATED_DIR" -I "$MLX_SRC_ROOT")
METAL_BIN="$(xcrun --find metal 2>/dev/null || true)"
METALLIB_BIN="$(xcrun --find metallib 2>/dev/null || true)"

if [[ -z "$METAL_BIN" || -z "$METALLIB_BIN" ]]; then
  if fallback_metallib="$(find_fallback_metallib)"; then
    echo "Metal toolchain incomplete; reusing existing metallib from $fallback_metallib"
    install_metallib "$fallback_metallib"
    echo "Done. SPM-built valartts can now run inference."
    exit 0
  fi

  echo "Error: Unable to locate both 'metal' and 'metallib' via xcrun, and no fallback mlx.metallib was found."
  echo "CLI and daemon inference require the full Metal toolchain, not just a partial Command Line Tools setup."
  echo "Recommended fixes:"
  echo "  xcodebuild -downloadComponent MetalToolchain"
  echo "  xcodebuild -runFirstLaunch"
  exit 1
fi

if ! "$METAL_BIN" -help >/dev/null 2>&1; then
  if fallback_metallib="$(find_fallback_metallib)"; then
    echo "Metal compiler exists but is not usable; reusing existing metallib from $fallback_metallib"
    install_metallib "$fallback_metallib"
    echo "Done. SPM-built valartts can now run inference."
    exit 0
  fi

  echo "Error: 'metal' was found, but the Metal Toolchain component is not usable."
  echo "Run these commands, then retry:"
  echo "  xcodebuild -downloadComponent MetalToolchain"
  echo "  xcodebuild -runFirstLaunch"
  exit 1
fi

echo "Compiling Metal shaders..."
compiled=0
skipped=0

# Compile ALL .metal files: backend kernels first (authoritative), then generated shaders.
# Track compiled basenames in a temp file to skip generated duplicates (bash 3 compatible).
compiled_names="$BUILD_DIR/.compiled_basenames"
: > "$compiled_names"

for f in $(find "$BACKEND_DIR" "$GENERATED_DIR" -name "*.metal" 2>/dev/null | sort); do
  [[ -f "$f" ]] || continue
  base=$(basename "$f" .metal)
  # Skip generated duplicates of backend kernels (backend is authoritative)
  if grep -qx "$base" "$compiled_names" 2>/dev/null; then
    skipped=$((skipped + 1))
    continue
  fi
  # Use relative path to create unique .air name
  relpath="${f#$MLX_CHECKOUT/}"
  airname=$(echo "$relpath" | tr '/' '_' | sed 's/\.metal$/.air/')
  if "$METAL_BIN" -c "$f" -o "$BUILD_DIR/$airname" \
    "${INCLUDE_FLAGS[@]}" \
    -std=metal3.1 -target air64-apple-macos14.0 2>/dev/null; then
    compiled=$((compiled + 1))
    echo "$base" >> "$compiled_names"
  elif "$METAL_BIN" -c "$f" -o "$BUILD_DIR/$airname" \
    "${INCLUDE_FLAGS[@]}" \
    -std=metal3.2 -target air64-apple-macos15.0 2>/dev/null; then
    # Some kernels (fence, nax variants) require Metal 3.2 features
    compiled=$((compiled + 1))
    echo "$base" >> "$compiled_names"
  else
    skipped=$((skipped + 1))
    echo "  SKIP $base (compile error)"
  fi
done
rm -f "$compiled_names"

echo "Compiled $compiled shaders ($skipped skipped)."

if [[ $compiled -eq 0 ]]; then
  if fallback_metallib="$(find_fallback_metallib)"; then
    echo "No shaders compiled successfully; reusing existing metallib from $fallback_metallib"
    install_metallib "$fallback_metallib"
    echo "Done. SPM-built valartts can now run inference."
    exit 0
  fi

  echo "Error: No shaders compiled successfully."
  exit 1
fi

echo "Linking mlx.metallib..."
"$METALLIB_BIN" "$BUILD_DIR"/*.air -o "$BUILD_DIR/mlx.metallib"

install_metallib "$BUILD_DIR/mlx.metallib"

echo "Done. SPM-built valartts can now run inference."
