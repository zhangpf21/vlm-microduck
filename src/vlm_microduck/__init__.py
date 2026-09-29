"""Independent vision and task code for MicroDuck experiments."""

from .detector import RedBox, annotate_red_box, find_red_box

__all__ = ["RedBox", "annotate_red_box", "find_red_box"]
