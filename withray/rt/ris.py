"""
Classes and Methods for RIS.
"""
import torch

class RIS:
    def __init__(self, name, pnts, dirs,
                 ant_tx = None,
                 ant_rx = None,
                 nmr = "FR1.n41"):

        if not isinstance(name, str):
            raise TypeError("'name' must be of type 'str'.")
        self.name = name

        self.ant_tx = ant_tx
        self.ant_rx = ant_rx
        if ant_tx is None:
            if ant_rx is None:
                raise ValueError("'ant_tx' or 'ant_rx' must be specified.")
            else:
                self.ant_tx = ant_rx
        else:
            if ant_rx is None:
                self.ant_rx = ant_tx

        self.pnts = pnts
        self.dirs = dirs


    @property
    def pnts(self):
        return self._pnts

    @pnts.setter
    def pnts(self, p):
        self._pnts = torch.tensor(p)

    @property
    def dirs(self):
        return self._dirs

    @dirs.setter
    def dirs(self, d):
        self._dirs = torch.tensor(d)