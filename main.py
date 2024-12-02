from vedo import *
import torch
from withray import rt

mesh = Mesh("map_data/Manhattan.obj",)
mesh.texture("map_data/Manhattan.png", scale = 1)
# mesh.show()

mesh_f = torch.tensor(mesh.faces())
mesh_v = torch.tensor(mesh.vertices)

numero = rt.Numerology("up","FR1.n41")
ant_bs_tx = rt.Antenna(8,
                       4,
                       intv_hor = 1/2,
                       intv_ver = 1/2,
                       numerology = numero)
ant_bs_rx = ant_bs_tx
ant_ue_tx = rt.Antenna(1,
                       1,
                       numerology = numero)

ant_ue_rx = ant_ue_tx
