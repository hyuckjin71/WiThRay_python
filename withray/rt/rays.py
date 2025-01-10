"""
Classes and Methods for RAYS.
"""
import time, copy
import torch
from vedo import Point, Light, Plotter, show, sin, cos
from withray import PI
from withray.rt import Tx, Rx
from withray.rt.utils import sorted_inv_s, line_pnt_intersect

class RAYS:
    def __init__(self, bs, ris, ue, mesh, device):

        self.bs = bs
        self.ris = ris
        self.ue = ue
        self.mesh = mesh
        self.device = device

    def collect_rays(self):

        tx_nodes = [Tx(bs) for bs in self.bs if bs.nmr.dir_link == "up"] + [Tx(ris) for ris in self.ris]
        rx_nodes = [Rx(ue) for ue in self.ue if ue.nmr.dir_link == "up"] + [Rx(ris) for ris in self.ris]

        for i in range(tx_nodes.__len__()):
            pnts_tx = tx_nodes[i].pnts.reshape(3, -1)
            pnts_rx = [node.pnts for node in rx_nodes if node.name != tx_nodes[i].name]
            pnts_rx = torch.cat([pnts.reshape(3, -1) for pnts in pnts_rx], dim=1)

            pnts_tx = pnts_tx.to(dtype=torch.float32, device=self.device)
            pnts_rx = pnts_rx.to(dtype=torch.float32, device=self.device)

            msk_in = RAYS.compute_rays(pnts_tx, pnts_rx, self.mesh)

    @staticmethod
    def compute_rays(pnts_tx, pnts_rx, mesh):

        # p1 = Point([200*cos(-230/180*PI), 200*sin(-230/180*PI), 60], c='white')
        # l1 = Light(p1, c='white')
        #
        # p2 = Point(pnts_tx[:, 0].to(device="cpu").numpy() + [0,0,1], c='white')
        # l2 = Light(p2, c='white')

        mesh_ = copy.deepcopy(mesh)
        mesh_.mesh_to_device("cpu")
        msk_v = ~line_pnt_intersect(pnts_tx.to(device="cpu"), mesh_.v.T, mesh_)

        idc_v = torch.nonzero(msk_v[0,:], as_tuple=True)[0]
        msk_f = torch.any( torch.isin(mesh.f, idc_v), dim=1)




