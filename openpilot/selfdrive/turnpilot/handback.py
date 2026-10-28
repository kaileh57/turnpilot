"""TurnPilot handing control back to the driver, on selfdrived's side.

turnpilotd sets turnpilotPlan.handback when TurnPilot can't do its job: the detector stopped producing fresh results
within its deadline, or modeld fell back to the small driving model, which TurnPilot can't navigate on. It stops acting
at once (no new desires, lane changes or turn caps; the scene's stops in force are kept), and selfdrived raises the
matching event, whose SOFT_DISABLE is openpilot's standard "take control" disable: the alert and chime, then a
disengage SOFT_DISABLE_TIME later, not an abrupt cut.

Like upstream's bigModelFailed, the soft disable is latched on the handback starting while engaged and held until
openpilot is disengaged, so a detector that blinks back during the countdown can't cancel it halfway. Engaging later
goes by the event's other types: a stopped detector blocks it (NO_ENTRY) while it stays stopped; after a model
fallback openpilot engages again on the small model, without TurnPilot.
"""
from openpilot.cereal import custom, log

Handback = custom.TurnpilotPlan.Handback
EventName = log.OnroadEvent.EventName

EVENTS = {
  Handback.detectorStopped: EventName.turnpilotDetectorStopped,
  Handback.modelFellBack: EventName.turnpilotModelFellBack,
}


class HandbackEvents:
  def __init__(self):
    self.prev = Handback.none
    self.latched = None  # the handback being driven to a disengage

  def update(self, plan, enabled: bool) -> list:
    """The events to add this frame. plan: the active turnpilotPlan (hooks.active_plan), None when turnpilotd isn't
    publishing or is in shadow mode; enabled: openpilot engaged (selfdrived's self.enabled)."""
    handback = plan.handback.raw if plan is not None else Handback.none
    started = handback != Handback.none and handback != self.prev
    self.prev = handback
    if not enabled:
      self.latched = None
    elif started and self.latched is None:
      self.latched = handback

    events = []
    if self.latched is not None:
      events.append(EVENTS[self.latched])  # the soft disable, to the end
    elif started or handback == Handback.detectorStopped:
      events.append(EVENTS[handback])  # disengaged: the alert once it starts, and no engaging while the detector is out
    return events

  def apply(self, events, plan, enabled: bool) -> None:
    """Add this frame's events to selfdrived's Events. On a fallback upstream's bigModelFailed soft-disables too, and on
    the same frame its alert would win the tie (same priority, lower event id): TurnPilot's replaces it, so the driver
    reads why TurnPilot handed back. The disable is the same; bigModelFailed's own alert shows when TurnPilot's lags it."""
    for event in self.update(plan, enabled):
      events.add(event)
    if EventName.turnpilotModelFellBack in events.names:
      events.events = [e for e in events.events if e != EventName.bigModelFailed]
