"""
Classes and Methods for RAYS.
"""
import torch
from withray.rt import Tx, Rx
from withray.rt.utils import sorted_inv_s

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

            compute_rays(pnts_tx, pnts_rx, self.mesh)


def compute_rays(pnts_tx, pnts_rx, mesh):

    inv_s_tx = sorted_inv_s(mesh, pnts_tx)
    num_f = inv_s_tx.shape[2]
    inv_s_tx = inv_s_tx.reshape(3,3,-1).permute(1,0,2).unsqueeze(-1)
    pnts_tx = pnts_tx.unsqueeze(-1).unsqueeze(-1).permute(0,2,3,1).unsqueeze(-1)
    pnts_v = torch.tensor(mesh.vertices)[:,[0,2,1]] * torch.tensor([-1, 1, 1])
    pnts_v = pnts_v.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).permute(1,2,3,4,0).to(dtype=torch.float32, device=pnts_tx.device)

    msk_in = torch.zeros(pnts_tx.shape[3], pnts_v.shape[4], dtype=torch.bool, device=pnts_tx.device)

    dir = pnts_v - pnts_tx
    idx = 0

    while idx < num_f and torch.sum(~msk_in) != 0:
        blck_size = int(torch.ceil(1e8 / torch.sum(~msk_in)))
        start_idx = idx + 1
        end_idx = min(idx + blck_size, num_f)
        idc = torch.arange(start_idx, end_idx+1, dtype=torch.long) - 1

        k = torch.sum(inv_s_tx[:,:,idc-1] * dir[:,:,:,~msk_in], axis=0)
        msk_in[~msk_in] = torch.any(torch.sum(k, axis=0) > 1 & torch.all(k > 0, axis=0), axis=0)

        idx = end_idx
