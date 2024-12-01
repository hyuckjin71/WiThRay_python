from vedo import *
import torch

mesh = Mesh("map_data/Manhattan.obj",)
mesh.texture("map_data/Manhattan.png", scale = 1)
# mesh.show()

mesh_f = torch.tensor(mesh.faces())
mesh_v = torch.tensor(mesh.vertices)

