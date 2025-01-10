from vedo import *
import torch
from withray import PI
from withray.rt import Numerology, Antenna, BS, RIS, UE, RAYS, MESH

def print_mouse_coordinates(evt):
    if evt.picked3d is not None:
        print(f"Mouse Over Coordinates: {evt.picked3d}")

# Select the device for acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

mesh = MESH("Manhattan",
            device,
            rotation_dir = torch.tensor([0.8409, -0.5411, 0.0]))

# p1 = Point([200*cos(-230/180*PI), 200*sin(-230/180*PI), 60], c='white')
# l1 = Light(p1, c='white')
#
# # mesh.mesh_file.cellcolors[:,1:3] = mesh.mesh_file.cellcolors[:,1:3] * 0.5
# # mesh.mesh_file.alpha(0.1)
# # msg = Text3D('Tx', pos=[25.4229, -167.19, -1])
#
# plotter = Plotter(title="Mouse Over Coordinates Example", axes=1)
# plotter.add_callback("mouse move", print_mouse_coordinates)
# plotter.show(mesh.mesh_file, p1, l1).interactive()

nmr = [Numerology("up","FR1.n41")]
ant_bs  = [Antenna(1,2, nmr = nmr[0]),]
ant_ris = [Antenna(8,8, intv_hor = 0.1, intv_ver = 0.1, nmr = nmr[0]),]
ant_ue  = [Antenna(1,1, nmr = nmr[0]),]

pnts_bs = [[25.4229, -167.19, -1],]
dirs_bs = [[-1, 0, 0],]

pnts_ris = [[1.3941, -153, 3],]
dirs_ris  = [[0, 1, 0],]

pnts_ue = [[-67.9091, -166.0409, -17],]
dirs_ue = [[0,0,1],]

bs = [BS(f"bs{i}", pnts, dirs_bs[i], ant_tx = ant_bs[i]) for i,pnts in enumerate(pnts_bs)]
bs[0].nmr = nmr[0]

ris = [RIS(f"ris{i}", pnts, dirs_ris[i], ant_tx = ant_ris[i]) for i,pnts in enumerate(pnts_ris)]
ris[0].nmr = nmr[0]

ue = [UE(f"ue{i}", pnts, dirs_ue[i], ant_rx = ant_ue[i]) for i,pnts in enumerate(pnts_ue)]
ue[0].nmr = nmr[0]

rays = RAYS(bs, ris, ue, mesh, device)
rays.collect_rays()