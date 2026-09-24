"""t07: the style registry for the 100 raisings.  Owner: T07Raised."""
from scenes import t07_styles as _s1
from scenes.t07_styles2 import STYLES2
from scenes.t07_styles3 import STYLES3

STYLES = dict(_s1.STYLES)
for _c in STYLES2 + STYLES3:
    STYLES[_c.name] = _c()

ORDER = ["soviet", "iwo", "delacroix", "moon", "lascaux", "egypt", "bayeux", "ukiyoe", "flaghack", "woodcut",
         "seurat", "blueprint"]
