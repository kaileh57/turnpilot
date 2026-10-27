#!/usr/bin/env bash
# Install pyvalhalla on the comma four, into a writable directory, because the system venv is read-only.
# Run it on the device. ValhallaSource adds the directory to sys.path when the system has no valhalla.
#
#   install_valhalla_device.sh [dest]        dest defaults to /data/turnpilot/pyvalhalla
#
# Needs the network once, or PYVALHALLA_WHEEL=/path/to/the/wheel for an offline install (scp it first).
# The wheel is a zip of valhalla/ (the Python package and its extension) and pyvalhalla.libs/ (every library it
# links except glibc, libstdc++ and libgcc_s). The extension finds them through $ORIGIN/../pyvalhalla.libs, so the two
# directories have to stay side by side. It needs glibc 2.27, GLIBCXX_3.4.22 and CXXABI_1.3.11 from the system, and the
# import test below fails if the device is older. Unpacked it is 102 MB; without valhalla/bin, 62 MB.
#
# Then copy the tiles, for example from the Mac:
#   ssh comma@<ip> mkdir -p /data/turnpilot/valhalla/indiana
#   scp ~/turnpilot_data/valhalla/indiana/tiles.tar comma@<ip>:/data/turnpilot/valhalla/indiana/
#   params put TurnPilotValhallaTiles /data/turnpilot/valhalla/indiana
set -euo pipefail

WHEEL_NAME="pyvalhalla-3.9.1-cp312-abi3-manylinux_2_28_aarch64.whl"
WHEEL_URL="https://files.pythonhosted.org/packages/e5/3a/087f21f1d3551ed30fed4a8945f631482482183ca9bb69a2fddc9454402a/$WHEEL_NAME"
WHEEL_SHA256="6ebe5f9dd6ce3b16413f939a271d68e64a4addd44df8a802a8eda0b9ca0bf16f"
DEST="${1:-/data/turnpilot/pyvalhalla}"
PYTHON="${PYTHON:-python3}"
FORCE="${FORCE:-0}"           # skip the checks that need an aarch64 Linux host, and the import test
KEEP_BIN="${KEEP_BIN:-0}"     # keep valhalla_service and the other tools (40 MB), for a local HTTP service

die() { echo "error: $*" >&2; exit 1; }

if [ "$FORCE" != 1 ]; then
  [ "$(uname -s)" = Linux ] && [ "$(uname -m)" = aarch64 ] || die "this wheel is for aarch64 Linux, not $(uname -s) $(uname -m)"
  "$PYTHON" -c 'import sys; sys.exit(sys.version_info[:2] < (3, 12))' || die "$PYTHON is older than Python 3.12 (the wheel is cp312-abi3)"
fi

mkdir -p "$(dirname "$DEST")"
work="$(mktemp -d "$(dirname "$DEST")/.pyvalhalla-install.XXXXXX")"
trap 'rm -rf "$work"' EXIT
avail_kb="$(df -Pk "$work" | awk 'NR==2 {print $4}')"
[ "$avail_kb" -gt 300000 ] || die "less than 300 MB free in $(dirname "$DEST")"

wheel="${PYVALHALLA_WHEEL:-$work/$WHEEL_NAME}"
if [ -z "${PYVALHALLA_WHEEL:-}" ]; then
  echo "downloading $WHEEL_NAME (35 MB)"
  curl -fL --retry 3 -o "$wheel" "$WHEEL_URL"
fi
"$PYTHON" - "$wheel" "$WHEEL_SHA256" <<'PY'
import hashlib
import sys
digest = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
sys.exit(f"sha256 mismatch for {sys.argv[1]}: {digest}" if digest != sys.argv[2] else 0)
PY

"$PYTHON" -m zipfile -e "$wheel" "$work/pyvalhalla"
if [ "$KEEP_BIN" = 1 ]; then
  chmod +x "$work"/pyvalhalla/valhalla/bin/*
else
  rm -rf "$work/pyvalhalla/valhalla/bin"
fi

if [ "$FORCE" != 1 ]; then
  PYTHONPATH="$work/pyvalhalla" "$PYTHON" -c 'import valhalla; print("pyvalhalla", valhalla.__version__, "imports")' || die "the unpacked wheel does not import"
fi

rm -rf "$DEST.old"
[ ! -e "$DEST" ] || mv "$DEST" "$DEST.old"
mv "$work/pyvalhalla" "$DEST"
rm -rf "$DEST.old"
echo "installed in $DEST ($(du -sh "$DEST" | cut -f1))"
