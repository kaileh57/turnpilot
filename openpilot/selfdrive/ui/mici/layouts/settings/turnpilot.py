import pyray as rl

from openpilot.selfdrive.navigation.helpers import latest_gps_source
from openpilot.selfdrive.turnpilot.maps.coverage import CoverageCache
from openpilot.selfdrive.turnpilot.web.pairing import pairing_url
from openpilot.selfdrive.turnpilot.web.store import last_gps_position
from openpilot.selfdrive.ui.lib.turnpilot import mapped_area_text
from openpilot.selfdrive.ui.mici.widgets.button import GreyBigButton
from openpilot.selfdrive.ui.mici.widgets.qr import QR
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.scroller import NavScroller

URL_REFRESH_INTERVAL = 5  # seconds; the address changes when the device joins the phone's hotspot


def shown_url(url: str) -> str:
  """The URL for the page, or 'no network' when pairing_url fell back to loopback, which no phone can reach."""
  return "no network" if url.startswith("http://127.") else url


class PairPhoneScroller(NavScroller):
  """The QR code and URL that pair a phone with the TurnPilot web app."""

  def __init__(self):
    super().__init__()
    self._url = pairing_url(ui_state.params)
    self._last_refresh = rl.get_time()
    self._qr = QR(self._url)
    self._url_button = GreyBigButton("", shown_url(self._url))
    self._coverage = CoverageCache()
    self._area_button = GreyBigButton("mapped area", self._area_text())
    self._scroller.scroll_indicator_start_after = self._qr
    self._scroller.add_widgets([
      self._qr,
      GreyBigButton("pair phone", "scan with the\nphone camera",
                    gui_app.texture("icons_mici/settings/device/phone.png", 85, 64)),
      self._url_button,
      GreyBigButton("", "The phone must be on the same network, such as its own hotspot."),
      self._area_button,
    ])

  def _area_text(self) -> str:
    """Which of TurnPilot's mapped areas the car is in, from the live fix onroad or the last saved one."""
    source = latest_gps_source(ui_state.sm)
    position = (ui_state.sm[source].latitude, ui_state.sm[source].longitude) if source else last_gps_position(ui_state.params)
    return mapped_area_text(self._coverage, position, (ui_state.params.get("TurnPilotValhallaTiles") or "").strip())

  def _update_state(self):
    super()._update_state()
    if rl.get_time() - self._last_refresh < URL_REFRESH_INTERVAL:
      return
    self._last_refresh = rl.get_time()
    self._area_button.set_value(self._area_text())
    url = pairing_url(ui_state.params)
    if url != self._url:
      self._url = url
      self._url_button.set_value(shown_url(url))
      self._qr._url = url
      if self._qr._texture and self._qr._texture.id != 0:
        rl.unload_texture(self._qr._texture)
      self._qr._texture = self._qr._generate_qr_code()

  def _update_layout_rects(self):
    super()._update_layout_rects()
    self._qr.set_rect(rl.Rectangle(self._qr.rect.x, self._qr.rect.y, self._rect.height, self._rect.height))


if __name__ == "__main__":
  gui_app.init_window("pair phone")
  page = PairPhoneScroller()
  gui_app.push_widget(page)
  try:
    for _ in gui_app.render():
      pass
  finally:
    del page
