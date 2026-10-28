#!/usr/bin/env bash
# Copy the map server and its regions from this machine to the VM, and restart it.
#
#   deploy.sh [region ...]      default: every region built under $REGIONS
#
# The VM's address and SSH key come from TURNPILOT_MAPS_HOST (default turnpilot@136.113.85.41) and
# TURNPILOT_MAPS_SSH_KEY (default ~/.ssh/turnpilot_maps). Run setup_server.sh on the VM once first;
# "deploy.sh --setup" does that. Each region is uploaded to a staging directory and swapped in whole, so
# the server never serves a half-copied region.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${TURNPILOT_MAPS_HOST:-turnpilot@136.113.85.41}"
KEY="${TURNPILOT_MAPS_SSH_KEY:-$HOME/.ssh/turnpilot_maps}"
REGIONS="${REGIONS:-$HOME/turnpilot_data/mapserver/regions}"
SSH=(ssh -i "$KEY" -o BatchMode=yes "$HOST")

if [ "${1:-}" = "--setup" ]; then
  shift
  rsync -e "ssh -i $KEY" "$HERE/setup_server.sh" "$HOST:/tmp/setup_server.sh"
  "${SSH[@]}" "bash /tmp/setup_server.sh ${HOST#*@}"
fi

rsync -e "ssh -i $KEY" "$HERE/server.py" "$HERE/geo.py" "$HERE/tiles.py" "$HOST:/opt/turnpilot-maps/"

if [ $# -eq 0 ]; then
  set -- $(cd "$REGIONS" && ls)
fi
for region in "$@"; do
  src="$REGIONS/$region"
  for f in fence.geojson meta.json index.json tiles.tar; do
    [ -e "$src/$f" ] || { echo "$src/$f is missing; build the region first" >&2; exit 1; }
  done
  echo "== $region"
  rsync -a --delete -e "ssh -i $KEY" --include='/fence.geojson' --include='/meta.json' --include='/index.json' \
    --include='/tiles.tar' --include='/tiles/***' --exclude='*' "$src/" "$HOST:/srv/turnpilot-maps/staging/$region/"
  "${SSH[@]}" "sudo rm -rf /srv/turnpilot-maps/regions/$region.old && \
    { [ ! -e /srv/turnpilot-maps/regions/$region ] || sudo mv /srv/turnpilot-maps/regions/$region /srv/turnpilot-maps/regions/$region.old; } && \
    sudo cp -a /srv/turnpilot-maps/staging/$region /srv/turnpilot-maps/regions/$region && \
    sudo rm -rf /srv/turnpilot-maps/regions/$region.old"
done

"${SSH[@]}" "sudo systemctl restart turnpilot-maps && sleep 3 && systemctl is-active turnpilot-maps && \
  curl -sk https://127.0.0.1/v1/health"
echo
