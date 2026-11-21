#!/usr/bin/env python3
"""
Copyright (c) 2026 Logitech Europe S.A.
SPDX-License-Identifier: MIT

LogiInsight -- check_connection.py: find out step by step why the camera's
SSH/SFTP link (the AI data) does not work, and how to fix it.

It needs no shell scripts (so it also works where PowerShell is blocked). Run it
before your first RGB example, and whenever "Failed to connect" shows up:

    python check_connection.py

Steps (each prints [PASS], [WARN] or [FAIL] and, on a problem, what to do):

    1. Your PC has an address on the camera's network (192.168.1.x)
    2. That network is not also used by your Wi-Fi / another adapter
    3. The camera's port 22 answers (TCP)
    4. SSH login with your user name and password
    5. The AI result file can be read
    6. The firmware release matches this SDK

The user name and password are on the card in your kit. Give them as
environment variables (never write them into a file):

    Windows (cmd):    set LOGIWEBCAM_USER=<user>       set LOGIWEBCAM_PASSWORD=<password>
    macOS / Linux:    export LOGIWEBCAM_USER='<user>'  export LOGIWEBCAM_PASSWORD='<password>'

Exit code 0 = no [FAIL], 1 = at least one [FAIL].
"""

import argparse
import os
import re
import socket
import subprocess
import sys
import time

from logiwebcam import (
    LogiWebcamClient,
    __version__ as SDK_VERSION,
    firmware_matches_sdk,
)

CAMERA_NET = "192.168.1."
PC_ADDRESS = "192.168.1.20"
SMART_ARTICLE = ("https://community.smarttech.com/Customer/s/article/"
                 "Device-drivers-for-CDC-NCM-device?language=en_US")

_results = []


def report(status, title, detail=""):
    _results.append(status)
    print(f"[{status}] {title}")
    if detail:
        for line in detail.strip("\n").splitlines():
            print(f"       {line}")
    print()


# --------------------------------------------------------------------------- #
# Local network information
# --------------------------------------------------------------------------- #

def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (out.stdout or "") + (out.stderr or "")
    except Exception:
        return ""


def local_ipv4():
    """Return a list of (adapter_name, ipv4) for this PC."""
    found = []
    if sys.platform == "win32":
        adapter = "?"
        for line in _run(["ipconfig"]).splitlines():
            if line and not line.startswith(" ") and line.rstrip().endswith(":"):
                adapter = line.rstrip().rstrip(":")
            m = re.search(r"IPv4[^:]*:\s*(\d+\.\d+\.\d+\.\d+)", line)
            if m:
                found.append((adapter, m.group(1)))
    elif sys.platform == "darwin":
        adapter = "?"
        for line in _run(["ifconfig"]).splitlines():
            m = re.match(r"^(\S+):\s+flags=", line)
            if m:
                adapter = m.group(1)
            m = re.search(r"\binet (\d+\.\d+\.\d+\.\d+)", line)
            if m:
                found.append((adapter, m.group(1)))
    else:
        for line in _run(["ip", "-o", "-4", "addr"]).splitlines():
            m = re.match(r"^\d+:\s+(\S+)\s+inet (\d+\.\d+\.\d+\.\d+)", line)
            if m:
                found.append((m.group(1), m.group(2)))
    return [(a, ip) for a, ip in found if not ip.startswith("127.")]


def route_interface(ip):
    """Name of the interface the OS uses to reach `ip` (macOS / Linux), or None."""
    if sys.platform == "darwin":
        m = re.search(r"interface:\s*(\S+)", _run(["route", "-n", "get", ip]))
        return m.group(1) if m else None
    if sys.platform.startswith("linux"):
        m = re.search(r"\bdev (\S+)", _run(["ip", "route", "get", ip]))
        return m.group(1) if m else None
    return None


# --------------------------------------------------------------------------- #
# Per-OS fix texts
# --------------------------------------------------------------------------- #

def fix_no_address():
    if sys.platform == "win32":
        return (
            "Give the camera's USB network adapter a fixed address:\n"
            "  1. Press Ctrl+R, type  ncpa.cpl  and press Enter.\n"
            "  2. Find the NEW Ethernet adapter that appeared when you plugged in the camera\n"
            "     (it may say 'USB', 'NCM' or 'LVNext', and 'Unidentified network').\n"
            "  3. Right-click it > Properties > 'Internet Protocol Version 4 (TCP/IPv4)' >\n"
            "     Properties > 'Use the following IP address':\n"
            f"       IP address {PC_ADDRESS}   Subnet mask 255.255.255.0\n"
            "       leave Default gateway and DNS EMPTY. OK, OK.\n"
            "  No new adapter in the list? Press Ctrl+R, type  devmgmt.msc :\n"
            "  if 'CDC NCM' sits under 'Other devices': right-click > Update driver > Browse my\n"
            "  computer > Let me pick > Network adapters > Microsoft > 'UsbNcm Host Device'.\n"
            f"  Details: {SMART_ARTICLE}\n"
            "  Also try another USB 3 cable or port."
        )
    if sys.platform == "darwin":
        return (
            "Give the camera's USB network adapter a fixed address:\n"
            "  1. System Settings > Network: pick the NEW entry that appeared when you plugged in\n"
            "     the camera (for example 'LVNext' or 'USB 10/100 LAN').\n"
            "  2. Details... > TCP/IP > Configure IPv4: Manually.\n"
            f"       IP address {PC_ADDRESS}   Subnet mask 255.255.255.0   Router: leave EMPTY.\n"
            "  3. OK / Apply.\n"
            "  No new entry? Try another USB 3 cable or port."
        )
    return (
        "Give the camera's USB network adapter a fixed address:\n"
        "  ip -br link                                   # find the new adapter (usb0 or enx...)\n"
        f"  sudo ip addr add {PC_ADDRESS}/24 dev <adapter>\n"
        "  sudo ip link set <adapter> up"
    )


def fix_clash():
    return (
        "Another network on this PC (most likely your Wi-Fi) also uses 192.168.1.x, so the PC cannot\n"
        "tell which network 192.168.1.10 is on. Do ONE of these:\n"
        "  - turn Wi-Fi off while you work with the camera (and back on when you need the internet),\n"
        "  - or join a Wi-Fi that does not use 192.168.1.x (a phone hotspot usually uses 172.x or 10.x),\n"
        "  - or ask a mentor for help."
    )


def fix_route(camera_if):
    if sys.platform == "darwin":
        return (
            "macOS sends the traffic for the camera over the wrong network interface.\n"
            "  - System Settings > Network > '...' menu > Set Service Order: drag the camera's adapter\n"
            "    ABOVE Wi-Fi, or\n"
            f"  - run:  sudo route -n add -host 192.168.1.10 -interface {camera_if or '<camera interface>'}\n"
            "  - or turn Wi-Fi off while you test."
        )
    return (
        "The system sends the traffic for the camera over the wrong interface.\n"
        f"  sudo ip route add 192.168.1.10/32 dev {camera_if or '<camera interface>'}\n"
        "  or turn Wi-Fi off while you test."
    )


def fix_no_port():
    lines = [
        "The PC has an address, but the camera does not answer on port 22. Check, in this order:",
        "  1. Wait 60 seconds after plugging in (the camera is still booting), then run this again.",
        "  2. 'ping 192.168.1.10' in a terminal. No answer: unplug the camera, plug it in again;",
        f"     check that the address of the adapter is {PC_ADDRESS} (step 1) and the mask is 255.255.255.0.",
    ]
    if sys.platform == "darwin":
        lines.append(
            "  3. macOS: System Settings > Privacy & Security > Local Network: switch ON your terminal\n"
            "     (and VS Code or any other program you start Python from), then run this again.")
    elif sys.platform == "win32":
        lines.append(
            "  3. Windows: a VPN or a firewall can block it. Disconnect the VPN; when Windows asks to\n"
            "     allow Python on a 'Private' network, say yes.")
    lines.append("  4. Make sure no other program or second camera is also using 192.168.1.10.")
    return "\n".join(lines)


def fix_env_missing():
    if sys.platform == "win32":
        how = ("  set LOGIWEBCAM_USER=<user on your card>\n"
               "  set LOGIWEBCAM_PASSWORD=<password on your card>")
    else:
        how = ("  export LOGIWEBCAM_USER='<user on your card>'\n"
               "  export LOGIWEBCAM_PASSWORD='<password on your card>'")
    return ("Set the user name and password from the card in your kit as environment variables\n"
            "in the SAME terminal window, then run this again:\n" + how)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #

def step1_2_addresses():
    addrs = local_ipv4()
    cam_side = [(a, ip) for a, ip in addrs if ip.startswith(CAMERA_NET) and ip != "192.168.1.10"]
    if not cam_side:
        shown = ", ".join(f"{a}={ip}" for a, ip in addrs) or "none found"
        report("FAIL", "1. This PC has no address on the camera network (192.168.1.x)",
               f"Addresses found on this PC: {shown}\n\n{fix_no_address()}")
        return None
    names = ", ".join(f"{a} = {ip}" for a, ip in cam_side)
    report("PASS", "1. This PC has an address on the camera network", names)

    if len(cam_side) > 1:
        report("FAIL", "2. Two adapters of this PC are both on 192.168.1.x",
               f"{names}\n\n{fix_clash()}")
        return cam_side[0][0]
    report("PASS", "2. Only one adapter uses 192.168.1.x")
    return cam_side[0][0]


def step3_port(ip, camera_adapter):
    iface = route_interface(ip)
    if iface and camera_adapter and iface != camera_adapter and sys.platform != "win32":
        report("WARN", f"The system reaches {ip} through '{iface}', not through '{camera_adapter}'",
               fix_route(camera_adapter))
    try:
        with socket.create_connection((ip, 22), timeout=2):
            pass
    except OSError as e:
        report("FAIL", f"3. {ip} port 22 does not answer ({e})", fix_no_port())
        return False
    report("PASS", f"3. {ip} answers on port 22")
    return True


def step4_login(args):
    user = args.user or os.environ.get("LOGIWEBCAM_USER")
    password = args.password or os.environ.get("LOGIWEBCAM_PASSWORD")
    if not user or not password:
        report("FAIL", "4. User name and/or password not set", fix_env_missing())
        return None
    try:
        cam = LogiWebcamClient(ip=args.ip, username=user, password=password, timeout=8)
        cam.connect()
    except Exception as e:
        msg = str(e)
        if "Authentication" in msg:
            report("FAIL", "4. SSH login refused: wrong user name or password",
                   "Compare with the card in your kit (copy the values exactly; no spaces\n"
                   "at the end). In Windows cmd do not put quotes around the values after 'set'.\n"
                   "In macOS / Linux use single quotes.")
        else:
            report("FAIL", "4. SSH login failed", msg)
        return None
    report("PASS", "4. SSH login works")
    return cam


def step5_results(cam):
    data = cam.get_latest_ai_data()
    if data is None:
        report("WARN", "5. The AI result file is empty or missing",
               "Normal right after boot. Start an RGB example (rgb_ai_live.py), then run this again.")
        return
    t1 = data.get("timestamp_ms")
    time.sleep(1.2)
    data2 = cam.get_latest_ai_data() or {}
    t2 = data2.get("timestamp_ms")
    loaded = data.get("plugin_loaded")
    if t1 is not None and t2 is not None and t1 != t2:
        report("PASS", "5. AI results are live", f"plugin_loaded={loaded}, timestamp_ms is advancing")
    else:
        report("WARN", "5. AI results are readable but not updating (timestamp_ms is frozen)",
               f"plugin_loaded={loaded}\n"
               "This is normal while no program has the RGB camera open. The results only update\n"
               "while the RGB camera is open (MJPG, YUY2 or NV12 on firmware 1.5.2 or newer;\n"
               "firmware 1.5.1 needs MJPG). Try rgb_ai_live.py.")


def step6_firmware(cam):
    fw = cam.get_firmware_version()
    if not fw:
        report("WARN", "6. Could not read the firmware version", "Not a problem for the examples.")
        return
    ok = firmware_matches_sdk(fw)
    if ok is None:
        report("WARN", "6. Firmware version text has no release label", fw)
    elif ok:
        report("PASS", f"6. Firmware matches this SDK (v{SDK_VERSION})", fw)
    else:
        report("FAIL", f"6. Firmware does not match this SDK (v{SDK_VERSION})",
               fw + "\nDownload the SDK that matches your firmware, or ask a mentor.")


def main():
    parser = argparse.ArgumentParser(description="LogiInsight -- check the SSH/SFTP connection")
    parser.add_argument("--ip", default="192.168.1.10", help="camera address (default 192.168.1.10)")
    parser.add_argument("--user", default=None, help="default: LOGIWEBCAM_USER")
    parser.add_argument("--password", default=None, help="default: LOGIWEBCAM_PASSWORD")
    args = parser.parse_args()

    print(f"LogiInsight check_connection  (SDK v{SDK_VERSION}, Python "
          f"{sys.version_info.major}.{sys.version_info.minor}, {sys.platform})\n")

    adapter = step1_2_addresses()
    cam = None
    try:
        if adapter is not None and step3_port(args.ip, adapter):
            cam = step4_login(args)
            if cam is not None:
                step5_results(cam)
                step6_firmware(cam)
    finally:
        if cam is not None:
            cam.disconnect()

    failed = _results.count("FAIL")
    print("=" * 60)
    print("RESULT: everything needed for the AI link works." if not failed
          else f"RESULT: {failed} problem(s) found. Fix the first [FAIL] above and run again.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C).")
        sys.exit(130)
