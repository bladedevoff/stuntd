from __future__ import annotations

__all__ = ["MODE_CHECK", "MODE_COLLECT", "MODE_LIVE", "MODE_SHADOW"]

MODE_COLLECT = "collect"
"""The site has no model yet, so every request goes to the provider and is recorded."""

MODE_SHADOW = "shadow"
"""The model answers alongside the provider and only its agreement is recorded."""

MODE_LIVE = "live"
"""The model answers on its own, except for the sampled requests that still check it."""

MODE_CHECK = "check"
"""Not a mode a site serves in: one live request sent to the provider anyway, to compare."""
