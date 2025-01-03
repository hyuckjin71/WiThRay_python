"""
Classes and Methods for Numerology.
"""
import torch
from withray import SPEED_OF_LIGHT

class Numerology:
    '''
    List of Numerologies
    [FR 1] n41 n78
    [FR 2] n260 n261
    '''
    def __init__(self, dir_link, nmr="FR1.n41"):
        self.dir_link = dir_link
        self.name = nmr

        if nmr == "FR1.n41":
            self.frequency = 2.5e9                                # Carrier frequency : 2.5 GHz
            self.freq_samp = 122.88e6                             # Sampling frequency : 122,880 samples per 1 msec
            self.bandwidth = 100e6                                # Bandwidth : 100 MHz
            self.num_RB = 273                                                   # Number of Resource blocks : 273
            self.len_fft_symb = 4096 * torch.ones([1,7])                        # FFT size : 4096
            self.len_cyc_symb = [320,288,288,288,288,288,288]     # Length of cyclic prefix symbols : 320,288,288,288,288,288,288

        elif nmr == "FR1.n78":
            self.frequency = 3.5e9                                # Carrier frequency : 3.5 GHz
            self.freq_samp = 122.88e6                             # Sampling frequency : 122,880 samples per 1 msec
            self.bandwidth = 100e6                                # Bandwidth : 100 MHz
            self.num_RB = 273                                                   # Number of Resource blocks : 273
            self.len_fft_symb = 4096 * torch.ones([1,7])                        # FFT size : 4096
            self.len_cyc_symb = [320,288,288,288,288,288,288]     # Length of cyclic prefix symbols : 320,288,288,288,288,288,288

        elif nmr == "FR2.n260":
            self.frequency = 38e9                                 # Carrier frequency: 38 GHz
            self.freq_samp = 122.88e6                             # Sampling frequency : 122,880 samples per 1 msec
            self.bandwidth = 100e6                                # Bandwidth : 100 MHz
            self.num_RB = 66                                                    # Number of Resource blocks : 66
            self.len_fft_symb = 1024 * torch.ones([1,7])                        # FFT size : 1024
            self.len_cyc_symb = [80,72,72,72,72,72,72]           # Length of cyclic prefix symbols : 80,72,72,72,72,72,72

        elif nmr == "FR2.n261":
            self.frequency = 28e9                                 # Carrier frequency : 28 GHz
            self.freq_samp = 245.76e6                             # Sampling frequency : 245,760 samples per 1 msec
            self.bandwidth = 200e6                                # Bandwidth : 200 MHz
            self.num_RB = 132                                                   # Number of Resource blocks : 132
            self.len_fft_symb = 2048 * torch.ones([1,7])                        # FFT size : 2048
            self.len_cyc_symb = [160,144,144,144,144,144,144]     # Length of cyclic prefix symbols : 160,144,144,144,144,144,144

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, f):
        self._frequency = torch.tensor(f)
        self._wavelength = SPEED_OF_LIGHT / torch.tensor(f)

    @property
    def freq_samp(self):
        return self._freq_samp

    @freq_samp.setter
    def freq_samp(self, f):
        self._freq_samp = torch.tensor(f)

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, b):
        self._bandwidth = torch.tensor(b)

    @property
    def len_fft_symb(self):
        return self._len_fft_symb

    @len_fft_symb.setter
    def len_fft_symb(self, l):
        self._len_fft_symb = l

    @property
    def len_cyc_symb(self):
        return self._len_cyc_symb

    @len_cyc_symb.setter
    def len_cyc_symb(self, l):
        self._len_cyc_symb = torch.tensor(l)
