"""
Classes and Methods for Antennas.
"""
import torch

class Antenna:
    def __init__(self, num_hor, num_ver, intv_hor="half", intv_ver="half", nmr = None, polarization = None, pattern = "dipole"):
        if pattern == "dipole":
            self.pattern = "dipole"
        else:
            self.pattern = pattern

        if nmr is None:
            from withray.rt import Numerology
            nmr = Numerology("up","FR1.n41")

        if isinstance(intv_hor, str):
            if intv_hor == "half":
                self.intv_hor = nmr._wavelength * 0.5
            else:
                raise TypeError("'intv_hor' must be 'half' or certain value.")
        else:
            self.intv_hor = intv_hor

        if isinstance(intv_ver, str):
            if intv_ver == "half":
                self.intv_ver = nmr._wavelength * 0.5
            else:
                raise TypeError("'intv_ver' must be 'half' or certain value.")
        else:
            self.intv_ver = intv_ver

        pnts = (torch.cat([torch.arange(0,num_hor).unsqueeze(0) * self.intv_hor,
                          torch.zeros(1,num_hor),
                          torch.zeros(1,num_hor)]).unsqueeze(2) +
                torch.cat([torch.zeros(1,num_ver),
                           torch.zeros(1, num_ver),
                           torch.arange(0,num_ver).unsqueeze(0) * self.intv_ver]).unsqueeze(2).permute(0,2,1))

        self.pnts = pnts - torch.tensor([(num_hor-1)*self.intv_hor,
                                         0,
                                         (num_ver-1)*self.intv_ver]).unsqueeze(1).unsqueeze(2)