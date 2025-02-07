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

# mesh = MESH("HealthScienceCampusUnivUtah_1", device)
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
# # plotter = Plotter()
# plotter.show(mesh.mesh_file, p1, l1).interactive()

nmr = [Numerology("up","FR1.n41")]


ant_bs  = [Antenna(1,2, nmr = nmr[0]),]

ant_ris = [Antenna(2,2, intv_hor = 0.1, intv_ver = 0.1, nmr = nmr[0]),]

ant_ue_tx  = [Antenna(1,1, nmr = nmr[0]),]
ant_ue_rx = [Antenna(1,1, pattern="patch", nmr = nmr[0]),]

pnts_bs = [[25.4229, -167.19, -1],]
dirs_bs = [[-1, 0, 0],]

pnts_ris = [[1.3941, -153, 3],]
dirs_ris  = [[0, 1, 0],]

# pnts_ue = [[ -48.10153583, -127.81710658,  -17.45029463],]
pnts_ue = [[  -8.80681734, -205.41460832,  -21.41402496]]
# pnts_ue = [[-10.25, -203.7, -20],]
# pnts_ue = [[ -67.05425681, -162.63761299,  -18],]
# pnts_ue = [[-70.0, -190.0, -20],]
dirs_ue = [[0,0,1],]
'''

ant_bs  = [Antenna(4,2, nmr = nmr[0]),  # ACC
           Antenna(4,2, nmr = nmr[0]),  # USTAR
           Antenna(4,2, nmr = nmr[0]),  # MEB
           Antenna(4,2, nmr = nmr[0])]  # EBC

ant_ris = []

ant_ue  = [Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),
           Antenna(1,1, nmr = nmr[0]),]
           
pnts_bs = [[174.04, 196.26, 49.65],    # ACC
           [-177.35, -23.26, -6.29],    # USTAR
           [-523.64, -23.96, -32.41],    # MEB
           [97.11, -153.15, -4.49]]    # EBC

dirs_bs = [[cos(PI/4), -sin(PI/4), 0],
           [-cos(PI/4), -sin(PI/4), 0],
           [-1, 0, 0],
           [cos(PI/4), sin(PI/4), 0],]

pnts_ris = []
dirs_ris = []

pnts_ue = [[-525.12, -107.79, -45.00],
           [-312.19, -447.57, -42.60],
           [-299.84, -155.23, -34.68],
           [-173.55, -0.41, -24.32],
           [-196.35, -108.00, -28.15],
           [-125.23, -194.67, -28.45],
           [9.84, -184.83, -22.76],
           [91.45, -145.17, -16.60],
           [-101.16, -54.92, -21.74],
           [18.56, 9.26, -12.92],
           [66.51, 103.75, -5.53],
           [123.31, 124.94, -4.25],
           [273.02, -149.54, -6.85],
           [-133.72, 420.07, -3.25],
           [-322.79, 266.80, -26.73],
           [-398.68, 49.90, -33.67],]

dirs_ue = [[0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],
           [0,0,1],]
'''
# plt = Plotter()
#
# p1 = Point([0 * cos(-230 / 180 * PI), 0 * sin(-230 / 180 * PI), 1000], c='white')
# l1 = Light(p1, c='white')
#
# plt += Point(pnts_bs[0]).color("red",1)
# plt += Point(pnts_bs[1]).color("red",1)
# plt += Point(pnts_bs[2]).color("red",1)
# plt += Point(pnts_bs[3]).color("red",1)
#
# plt += Point(pnts_ue[0]).color("blue",1)
# plt += Point(pnts_ue[1]).color("blue",1)
# plt += Point(pnts_ue[2]).color("blue",1)
# plt += Point(pnts_ue[3]).color("blue",1)
# plt += Point(pnts_ue[4]).color("blue",1)
# plt += Point(pnts_ue[5]).color("blue",1)
# plt += Point(pnts_ue[6]).color("blue",1)
# plt += Point(pnts_ue[7]).color("blue",1)
# plt += Point(pnts_ue[8]).color("blue",1)
# plt += Point(pnts_ue[9]).color("blue",1)
# plt += Point(pnts_ue[10]).color("blue",1)
# plt += Point(pnts_ue[11]).color("blue",1)
# plt += Point(pnts_ue[12]).color("blue",1)
# plt += Point(pnts_ue[13]).color("blue",1)
# plt += Point(pnts_ue[14]).color("blue",1)
# plt += Point(pnts_ue[15]).color("blue",1)
#
# plt.show(mesh.mesh_file, p1, l1).interactive()

bs = [BS(f"bs{i}", pnts, dirs_bs[i], ant_tx = ant_bs[i], nmr = nmr[0]) for i,pnts in enumerate(pnts_bs)]

ris = [RIS(f"ris{i}", pnts, dirs_ris[i], ant_tx = ant_ris[i], nmr = nmr[0]) for i,pnts in enumerate(pnts_ris)]

ue = [UE(f"ue{i}", pnts, dirs_ue[i], ant_tx = ant_ue_tx[i], ant_rx = ant_ue_rx[i], nmr = nmr[0]) for i,pnts in enumerate(pnts_ue)]

rays = RAYS(bs, ris, ue, mesh, device)
rays.collect_rays(3, 1e3)