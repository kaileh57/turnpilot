import plistlib
import subprocess

from openpilot.common.hardware import usb
from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, get_darwin_usb_state

# trimmed `ioreg -p IOUSB -r -c IOUSBHostDevice -a`, a hub with a child next to a chestnut
IOREG = [
  {"IOObjectClass": "IOUSBHostDevice", "idVendor": 0x2230, "idProduct": 0x2046, "locationID": 0x00240000, "USB Address": 7,
   "UsbLinkSpeed": 10_000_000_000, "kUSBVendorString": "VIA Labs, Inc.", "kUSBProductString": "USB3.1 Hub",
   "IORegistryEntryChildren": [
     {"IOObjectClass": "IOUSBHostDevice", "idVendor": 0x0bda, "idProduct": 0x8153, "locationID": 0x00244300, "USB Address": 10,
      "UsbLinkSpeed": 5_000_000_000, "kUSBVendorString": "Realtek", "kUSBProductString": "USB 10/100/1000 LAN",
      "USB Product Name": "USB 10_100_1000 LAN"},
   ]},
  {"IOObjectClass": "IOUSBHostDevice", "idVendor": 0x3801, "idProduct": 0x0001, "locationID": 0x01200000, "USB Address": 1,
   "UsbLinkSpeed": 10_000_000_000, "kUSBVendorString": "tiny", "kUSBProductString": CHESTNUT_USB_PRODUCT},
]


def fake_ioreg(out: bytes):
  def run(*args, **kwargs):
    return subprocess.CompletedProcess(args, 0, stdout=out, stderr=b"")
  return run


def raise_oserror(*args, **kwargs):
  raise OSError


class TestDarwinUsbState:
  def test_parse(self, monkeypatch):
    monkeypatch.setattr(usb.subprocess, "run", fake_ioreg(plistlib.dumps(IOREG)))
    devices = get_darwin_usb_state()
    assert [(d["busnum"], d["devnum"]) for d in devices] == [(0, 7), (0, 10), (1, 1)]
    chestnut = devices[-1]
    assert (chestnut["vendorId"], chestnut["productId"], chestnut["speedMbps"]) == (0x3801, 0x0001, 10000)
    assert chestnut["product"] == CHESTNUT_USB_PRODUCT and chestnut["manufacturer"] == "tiny"
    # the product string keeps its slashes, unlike "USB Product Name"
    assert devices[1]["product"] == "USB 10/100/1000 LAN"

  def test_no_devices(self, monkeypatch):
    monkeypatch.setattr(usb.subprocess, "run", fake_ioreg(b""))
    assert get_darwin_usb_state() == []

  def test_ioreg_fails(self, monkeypatch):
    monkeypatch.setattr(usb.subprocess, "run", raise_oserror)
    assert get_darwin_usb_state() == []

  def test_chestnut_present(self, monkeypatch):
    from openpilot.selfdrive.modeld.helpers import chestnut_present
    monkeypatch.setattr(usb.sys, "platform", "darwin")
    monkeypatch.setattr(usb.subprocess, "run", fake_ioreg(plistlib.dumps(IOREG)))
    assert chestnut_present()
    monkeypatch.setattr(usb.subprocess, "run", fake_ioreg(plistlib.dumps(IOREG[:1])))
    assert not chestnut_present()
