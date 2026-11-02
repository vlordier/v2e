"""Lowpass IIR filter functions for DVS emulator photoreceptor model.

Re-exports low_pass_filter from emulator_utils for backward compatibility.
"""

from v2ecore.emulator_utils import low_pass_filter

__all__ = ["low_pass_filter"]
