from vedo import *
import torch
from withray import PI
from withray.rt import Numerology, Antenna, BS, RIS, UE, RAYS, MESH

def print_mouse_coordinates(evt):
    if evt.picked3d is not None:
        print(f"Mouse Over Coordinates: {evt.picked3d}")

# Select the device for acceleration
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Using CUDA")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Using MPS")
# else:
device = torch.device("cpu")
print("Using CPU")

mesh = MESH("Manhattan",
            device,
            rotation_dir = torch.tensor([0.8409, -0.5411, 0.0]))

# p1 = Point([200*cos(-230/180*PI), 200*sin(-230/180*PI), 60], c='white')
# l1 = Light(p1, c='white')
#
# mesh.mesh_file.cellcolors[:,1:3] = mesh.mesh_file.cellcolors[:,1:3] * 0.5
# mesh.mesh_file.alpha(0.1)
# msg = Text3D('Tx', pos=[25.4229, -167.19, -1])
#
# plotter = Plotter(title="Mouse Over Coordinates Example", axes=1)
# plotter.add_callback("mouse move", print_mouse_coordinates)
# plotter.show(mesh.mesh_file, p1, l1).interactive()]

nmr = [Numerology("up","FR1.n41")]
ant_bs  = [Antenna(1,2, nmr = nmr[0]),]
ant_ris = [Antenna(8,8, intv_hor = 0.1, intv_ver = 0.1, nmr = nmr[0]),]
ant_ue  = [Antenna(1,1, nmr = nmr[0]),]

pnts_bs = [[25.4229, -167.19, -1],]
dirs_bs = [[-1, 0, 0],]

pnts_ris = [[1.3941, -153, 3],]
dirs_ris  = [[0, 1, 0],]

pnts_ue = [[-61.2, -164.2, -13],]
# pnts_ue = [[-23.0, -203.0, -20],]
# pnts_ue = [[-70.0, -190.0, -20],]
dirs_ue = [[0,0,1],]

bs = [BS(f"bs{i}", pnts, dirs_bs[i], ant_tx = ant_bs[i]) for i,pnts in enumerate(pnts_bs)]
bs[0].nmr = nmr[0]

ris = [RIS(f"ris{i}", pnts, dirs_ris[i], ant_tx = ant_ris[i]) for i,pnts in enumerate(pnts_ris)]
ris[0].nmr = nmr[0]

ue = [UE(f"ue{i}", pnts, dirs_ue[i], ant_rx = ant_ue[i]) for i,pnts in enumerate(pnts_ue)]
ue[0].nmr = nmr[0]

rays = RAYS(bs, ris, ue, mesh, device)
rays.collect_rays()

# =====================================================================================================================

msk_f = torch.any(msk_v[start_idx:end_idx, self.mesh.f], dim=2)
i1, i2 = torch.where(msk_f)
i1 += start_idx

mesh_v = mesh_.s[:, 0, i2].unsqueeze(1)
mesh_n = mesh_.n[:, i2].unsqueeze(1)

# (Reflection from diffraction) ===============================================================================

msk_difrct = torch.any(node_prev[:, 0:2, -1, i1] != 0, dim=[0, 1])
msk_reflct = ~msk_difrct

'''
Check whether the next vertices form a valid reflective surface
based on whether the previous node (node_prev) was a reflecting or diffracting point.

The next reflective surface must face the previous node, meaning the directional vector
should align with the surface's normal vector.
'''

msk_difrct[msk_difrct.clone()] = torch.any(
    torch.sum(mesh_n[:, :, msk_difrct] * (node_prev[:, 0:2, -1, i1][:, :, msk_difrct] - mesh_v[:, :, msk_difrct]),
              dim=0) > 0, dim=0
) & ~torch.any(torch.all(mesh_v[:, :, msk_difrct] == node_prev[:, 0:2, -1, i1][:, :, msk_difrct], dim=0), dim=0)

dist = torch.linalg.norm(mesh_v[:, :, msk_difrct] - node_prev[:, 0:2, -1, i1][:, :, msk_difrct], dim=0)
dist = torch.where(dist < 1, torch.tensor(1.0), dist)
alignment = torch.min(
    torch.abs(
        torch.sum(mesh_n[:, :, msk_difrct] * (mesh_v[:, :, msk_difrct] - node_prev[:, 0:2, -1, i1][:, :, msk_difrct]),
                  dim=0)
        ) / torch.linalg.norm(mesh_v[:, :, msk_difrct] - node_prev[:, 0:2, -1, i1][:, :, msk_difrct], dim=0),
    dim=0, keepdim=True)

node_priority_ = mesh_.fa[i2][msk_difrct].unsqueeze(0) * alignment[0] / torch.gather(dist, dim=0, index=alignment[1])
node_priority_ = node_priority_.squeeze() * node_priority_prev[i1][msk_difrct]
node_priority = torch.cat((node_priority, node_priority_), dim=0)