from vedo import *
import torch
from withray import rt

mesh = Mesh("map_data/Manhattan.obj",)
mesh.texture("map_data/Manhattan.png", scale = 1)
# mesh.show()

mesh_f = torch.tensor(mesh.faces())
mesh_v = torch.tensor(mesh.vertices)

numero = rt.Numerology("up","FR1.n41")
ant_tx = rt.Antenna(num_hor = 8,
                    num_ver = 4,
                    intv_hor = numero.wavelength/2,
                    intv_ver = numero.wavelength/2)