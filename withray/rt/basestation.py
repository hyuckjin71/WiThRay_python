"""
Classes and Methods for BS.
"""
import torch

class BS:
    def __init__(self, name, pnts, dirs, dir_pol = None,
                 ant_tx = None,
                 ant_rx = None,
                 nmr = "FR1.n41"):

        if not isinstance(name, str):
            raise TypeError("'name' must be of type 'str'.")
        self.name = name

        self.nmr = nmr

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

        assert self.ant_tx.pattern != "patch", "Error: 'patch' cannot be assigned to Tx antenna pattern."

        self.pnts = pnts
        self.dirs = dirs

        pol_basis = torch.tensor([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        if dir_pol is None:
            _,idx_max = torch.max(torch.tensor(self.dirs))


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