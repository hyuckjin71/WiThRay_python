"""
Classes and Methods for Transmitters.
"""
import torch

from withray import PI

class Tx:
    def __init__(self, node):

        self.name = node.name
        self.nmr = node.nmr
        self.ant = node.ant_tx

        self.pnts = node.pnts.unsqueeze(1).unsqueeze(2) + self.ant.pnts
        self.dirs = node.dirs



