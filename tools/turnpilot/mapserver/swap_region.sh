#!/usr/bin/env bash
# Swap one uploaded region into the live regions directory on the map server VM, restart the server, and check it.
#
#   swap_region.sh <regions dir> <staging dir> <region>
#
# deploy.sh runs this on the VM after each region's upload. The copy goes to .<region>.new beside the live directory
# (the server skips dot directories), then two renames swap it in, so the live name never holds a half-copied region.
# The server keeps each region's index in memory and reads tiles by path, so it restarts right after the swap. If the
# server doesn't come back serving the region, the old one is renamed back, the new one is kept as .<region>.failed,
# and the script fails.
#
# SUDO (default sudo), RESTART and CHECK (the systemd restart and a health request) can be overridden for tests.
set -euo pipefail

REGIONS="${1:?usage: $0 <regions dir> <staging dir> <region>}"
STAGING="${2:?usage: $0 <regions dir> <staging dir> <region>}"
REGION="${3:?usage: $0 <regions dir> <staging dir> <region>}"
SUDO="${SUDO-sudo}"
RESTART="${RESTART:-systemctl restart turnpilot-maps}"
CHECK="${CHECK:-curl -sfk --max-time 10 https://127.0.0.1/v1/health}"
SETTLE="${SETTLE:-3}"

case "$REGION" in
  ''|.*|*/*) echo "bad region name: $REGION" >&2; exit 1 ;;
esac

live="$REGIONS/$REGION"
new="$REGIONS/.$REGION.new"
old="$REGIONS/.$REGION.old"
failed="$REGIONS/.$REGION.failed"

$SUDO rm -rf "$new" "$old" "$failed"
$SUDO cp -a "$STAGING/$REGION" "$new"
if [ -e "$live" ]; then
  $SUDO mv "$live" "$old"
fi
$SUDO mv "$new" "$live"

if $SUDO $RESTART && sleep "$SETTLE" && health="$(bash -c "$CHECK")" && grep -q "'$REGION'\|\"$REGION\"" <<<"$health"; then
  $SUDO rm -rf "$old"
  echo "$health"
else
  echo "the server did not come back serving $REGION; rolling back to the old copy" >&2
  $SUDO mv "$live" "$failed"
  if [ -e "$old" ]; then
    $SUDO mv "$old" "$live"
  fi
  $SUDO $RESTART || true
  exit 1
fi
