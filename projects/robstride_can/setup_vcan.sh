#!/usr/bin/env bash
# Create the virtual CAN interface that waveshare_bridge.py feeds. Run once per boot.
#   ./setup_vcan.sh          # creates vcan0
#   ./setup_vcan.sh vcan1
set -euo pipefail
IFACE="${1:-vcan0}"

sudo modprobe vcan
if ! ip link show "$IFACE" &>/dev/null; then
    sudo ip link add "$IFACE" type vcan
fi
sudo ip link set "$IFACE" up
ip -br link show "$IFACE"

if ! id -nG "$USER" | grep -qw dialout; then
    echo
    echo "NOTE: $USER is not in the 'dialout' group, so /dev/ttyUSB0 can't be opened."
    echo "      Run: sudo usermod -aG dialout $USER   then log out and back in."
fi
