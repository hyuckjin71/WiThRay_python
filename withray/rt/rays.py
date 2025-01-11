"""
Classes and Methods for RAYS.
"""
import time, copy
import torch
from vedo import Point, Light, Plotter, show, sin, cos
from withray import PI
from withray.rt import Tx, Rx
from withray.rt.utils import line_pnt_intersect

class CACHE:
    def __init__(self, level, device):
        self.pnts = torch.zeros(3,3,level,0, device = device)
        self.info = torch.zeros(level+1,3,0, device = device, dtype=torch.int)

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

            msk_in = self.compute_rays(pnts_tx, pnts_rx)

    def compute_rays(self, pnts_tx, pnts_rx):

        # p1 = Point([200*cos(-230/180*PI), 200*sin(-230/180*PI), 60], c='white')
        # l1 = Light(p1, c='white')
        #
        # p2 = Point(pnts_tx[:, 0].to(device="cpu").numpy() + [0,0,1], c='white')
        # l2 = Light(p2, c='white')

        mesh_ = copy.deepcopy(self.mesh)
        mesh_.mesh_to_device("cpu")
        msk_v = ~line_pnt_intersect(pnts_tx.to(device="cpu"), mesh_.v.T, mesh_)

        for i in range(pnts_tx.shape[1]):
            node_prev = torch.cat([torch.zeros(3,2).to(device=pnts_tx.device), pnts_tx[:,i].view(3,1)], dim=1).view(3,3,1,1)
            info_prev = torch.zeros(1,3, dtype=torch.int32, device=pnts_tx.device).view(1,3,1)

            start_time = time.time()
            cache = self.next_nodes(1, i, node_prev, info_prev, msk_v[i,:].unsqueeze(0))
            end_time = time.time()
            print(f"Execution time: {end_time-start_time: .6f} seconds")

    def next_nodes(self, level, idx_tx, node_prev, info_prev, msk_v):

        cache = CACHE(level, self.device)

        for i in range(msk_v.shape[0]):
            idc_v = torch.nonzero(msk_v[i,:], as_tuple=True)[0]
            msk_f = torch.any( torch.isin(self.mesh.f.to(device="cpu"), idc_v), dim=1).to(device=self.device)

            mesh_v = self.mesh.v[self.mesh.f[msk_f,0],:].permute(1,0).view(3,1,1,-1)    # (3 x 1 x 1 x num_r) first point of surfaces in msk_f
            mesh_n = self.mesh.n[:,msk_f].view(3,1,1,-1)                                # (3 x 1 x 1 x num_r) normal vector of surfaces in msk_f

            '''
            Check whether the next vertices form a valid reflective surface 
            based on whether the previous node (node_prev) was a reflecting or diffracting point.
            
            The next reflective surface must face the previous node, meaning the directional vector 
            should align with the surface's normal vector.
            '''
            case_difrct = torch.any(node_prev[:,0:2,-1,i] > 0)
            if case_difrct:
                msk_f[msk_f.clone()] = torch.any(
                    torch.sum( mesh_n * (node_prev[:,0:2,-1,i].view(3,2,1,1) - mesh_v), dim=0) > 0, dim=0
                )
            else:
                msk_f[msk_f.clone()] = torch.sum( mesh_n * (node_prev[:,2,-1,i].view(3,1,1,1) - mesh_v), dim=0) > 0

            mesh_v = self.mesh.v[self.mesh.f[msk_f, 0], :].permute(1,0).view(3,1,1,-1)   # (3 x 1 x 1 x num_r) first point of surfaces in msk_f
            mesh_n = self.mesh.n[:, msk_f].view(3,1,1,-1)                                # (3 x 1 x 1 x num_r) normal vector of surfaces in msk_f

            node_r = 2 * torch.sum( (mesh_v - node_prev[:,:,:,i].unsqueeze(-1)) * mesh_n, dim=0).unsqueeze(0) * mesh_n * (node_prev[:,:,:,i] != 0).unsqueeze(-1) + node_prev[:,:,:,i].unsqueeze(-1)
            num_r = node_r.shape[3]

            pnts = torch.cat((torch.zeros(3,2,1,num_r).to(device=self.device), node_prev[:,2,0,i].view(3,1,1,1).repeat(1,1,1,num_r)), dim=1)    # (3 x 3 x 1 x num_f)
            pnts = torch.cat((pnts[:,:,1:min(1,level-1),:], node_r), dim=2)                                                                     # (3 x 3 x level x num_f)

            info = torch.cat(
                (torch.cat((idx_tx*torch.ones(1,1,num_r), torch.zeros(1,2,num_r)), dim=1).to(device=self.device),
                 info_prev[1:level,:,i].unsqueeze(-1).repeat(1,1,num_r),
                 torch.cat((torch.where(msk_f)[0].view(1,-1), torch.zeros(1,num_r).to(device=self.device), torch.ones(1,num_r).to(device=self.device)), dim=0).view(1,3,-1)
                 ), dim=0
            ).to(dtype = torch.int)                       # ((level+1) x 3 x num_r

            cache.pnts = torch.cat((cache.pnts, pnts), dim=3)
            cache.info = torch.cat((cache.info, info), dim=2)

            msk_d = torch.stack([
                torch.all( torch.isin(self.mesh.f[:,[0,1]].to(device="cpu"), idc_v), dim=1),
                torch.all( torch.isin(self.mesh.f[:,[1,2]].to(device="cpu"), idc_v), dim=1),
                torch.all( torch.isin(self.mesh.f[:,[2,0]].to(device="cpu"), idc_v), dim=1)
            ], dim=0).to(device=self.device)

            '''
            Check whether the next vertices form a valid diffracting edge
            based on whether the previous node (node_prev) was a reflecting or diffracting point.

            A valid diffracting edge can be determined as belonging to the surface that faces the previous node.
            This is because the diffracting edge may also belong to surfaces that do not face the previous node,
            but it must belong to at least one surface that faces the previous node.
            '''
            for ii in range(3):
                mesh_v = self.mesh.v[self.mesh.f[msk_d[ii,:], 0], :].permute(1,0).view(3,1,1,-1)
                mesh_n = self.mesh.n[:, msk_d[ii,:]].view(3,1,1,-1)
                if case_difrct:
                    msk_d[ii,msk_d[ii,:].clone()] = torch.any(
                        torch.sum( mesh_n * (node_prev[:,0:2,-1,i].view(3,2,1,1) - mesh_v), dim=0) >0, dim=0
                    ) * msk_d[ii,msk_d[ii,:]]
                else:
                    msk_d[ii,msk_d[ii,:].clone()] = (
                            torch.sum( mesh_n * (node_prev[:,2,-1,i].view(3,1,1,1) - mesh_v), dim=0) > 0
                    ) * msk_d[ii,msk_d[ii,:]]

            idc_d = torch.cat(
                (torch.where(msk_d[0,:])[0],
                 torch.where(msk_d[1,:])[0],
                 torch.where(msk_d[2,:])[0]), dim=0)
            idc_e = torch.cat(
                (torch.ones(torch.sum(msk_d[0,:])),
                 torch.ones(torch.sum(msk_d[1,:]))*2,
                 torch.ones(torch.sum(msk_d[2,:]))*3), dim=0
            ).to(device=self.device)

            node_d = torch.cat(
                (torch.cat((self.mesh.v[self.mesh.f[msk_d[0,:],0],:].permute(1,0).view(3,-1,1),
                            self.mesh.v[self.mesh.f[msk_d[0,:],1],:].permute(1,0).view(3,-1,1)), dim=2),
                 torch.cat((self.mesh.v[self.mesh.f[msk_d[1,:],1],:].permute(1,0).view(3,-1,1),
                            self.mesh.v[self.mesh.f[msk_d[1,:],2],:].permute(1,0).view(3,-1,1)), dim=2),
                 torch.cat((self.mesh.v[self.mesh.f[msk_d[2,:],2],:].permute(1,0).view(3,-1,1),
                            self.mesh.v[self.mesh.f[msk_d[2,:],0],:].permute(1,0).view(3,-1,1)), dim=2)), dim=1
            ).unsqueeze(-1).permute(0,2,3,1)                # (3 x 2 x 1 x num_d) two points of edge in msk_d
            num_d = node_d.shape[3]

            pnts = torch.cat((node_d, node_prev[:,2,-1,i].view(3,1,1,1).repeat(1,1,1,num_d)), dim=1)
            pnts = torch.cat((node_prev[:,:,0:level-1,i].unsqueeze(-1).repeat(1,1,1,num_d), pnts),dim=2)

            info = torch.cat(
                (torch.cat((idx_tx * torch.ones(1, 1, num_d), torch.zeros(1, 2, num_d)), dim=1).to(device=self.device),
                 info_prev[1:level, :, i].unsqueeze(-1).repeat(1, 1, num_d),
                 torch.cat((idc_d.view(1,-1), idc_e.view(1,-1), (2*torch.ones(1, num_d)).to(device=self.device)), dim=0).view(1, 3, -1)), dim=0
            ).to(dtype=torch.int)

            cache.pnts = torch.cat((cache.pnts, pnts), dim=3)
            cache.info = torch.cat((cache.info, info), dim=2)

            print(f"\r [{100*(i+1)/msk_v.shape[0]: 5.2f} %] {i+1}")

            return cache