"""
Classes and Methods for Antennas.
"""

import numpy as np

class Antenna:
    def __init__(self, pattern = "dipole"):
        if pattern == "dipole":
            self.pattern = "dipole"
