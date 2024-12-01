"""
Classes and Methods for Transmitters.
"""
import torch

from withray import PI

class Tx:
    def __init__(self, antenna, numerology):
        self.antenna = antenna
        self.numero = numerology
        self.position = torch.zeros([3,0])
        self.orientation = torch.zeros([3,0])

        if

