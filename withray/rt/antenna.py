"""
Classes and Methods for Antennas.
"""
import torch

class Antenna:
    def __init__(self, num_hor, num_ver, intv_hor=0, intv_ver=0, numerology = "FR1.n41", polarization = None, pattern = "dipole"):
        if pattern == "dipole":
            self.pattern = "dipole"
        else:
            self.pattern = pattern

        pnts = torch.cat([torch.arange(0,num_hor).unsqueeze(0) * numerology.wavelength * intv_hor,
                          torch.zeros(1,num_hor),
                          torch.zeros(1,num_hor)]).unsqueeze(2) + torch.cat([torch.zeros(1,num_ver),
                                                                             torch.zeros(1, num_ver),
                                                                             torch.arange(0,num_ver).unsqueeze(0) * numerology.wavelength * intv_ver]).unsqueeze(2).permute(0,2,1)

        self.pnts = pnts - torch.tensor([(num_hor-1)*numerology.wavelength*intv_hor/2,
                                         0,
                                         (num_ver-1)*numerology.wavelength*intv_ver/2]).unsqueeze(1).unsqueeze(2)