"""
Classes and Methods for Receiver.
"""
import torch

from withray import PI

class Rx:
    def __init__(self, antenna, numero, num_ant):
        self.antenna = antenna
        self.numero = numero
        self.position = torch.zeros([3,0])
        self.orientation = torch.zeros([3,0])


