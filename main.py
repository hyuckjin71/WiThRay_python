from vedo import *
import torch
from withray.rt import Numerology, Antenna, BS, RIS, UE, RAYS
from withray.rt.utils import import_mesh

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

mesh_file = Mesh("map_data/Manhattan.obj",)
mesh = import_mesh(mesh_file, device, rotation_dir = torch.tensor([0.8409, -0.5411, 0.0]))

# mesh_file.texture("map_data/Manhattan.png", scale = 1)
# mesh_file.show()

nmr = [Numerology("up","FR1.n41")]
ant_bs  = [Antenna(1,2, nmr = nmr[0]),]
ant_ris = [Antenna(8,8, intv_hor = 0.1, intv_ver = 0.1, nmr = nmr[0]),]
ant_ue  = [Antenna(1,1, nmr = nmr[0]),]

pnts_bs = [[25.4299, -167.19, -1],]
dirs_bs = [[-1, 0, 0],]

pnts_ris = [[0, -150, -1],]
dirs_ris  = [[0, 1, 0],]

pnts_ue = [[-88.2564, -204.805, -20.7447],]
dirs_ue = [[0,0,1],]

bs = [BS(f"bs{i}", pnts, dirs_bs[i], ant_tx = ant_bs[i]) for i,pnts in enumerate(pnts_bs)]
bs[0].nmr = nmr[0]

ris = [RIS(f"ris{i}", pnts, dirs_ris[i], ant_tx = ant_ris[i]) for i,pnts in enumerate(pnts_ris)]
ris[0].nmr = nmr[0]

ue = [UE(f"ue{i}", pnts, dirs_ue[i], ant_rx = ant_ue[i]) for i,pnts in enumerate(pnts_ue)]
ue[0].nmr = nmr[0]

rays = RAYS(bs, ris, ue, mesh, device)
rays.collect_rays()