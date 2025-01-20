"""
Classes and Methods for RAYS.
"""
import time, copy
import torch
from vedo import Point, Line, Light, Plotter, Text3D, show, sin, cos
from withray import PI
from withray.rt import Tx, Rx
from withray.rt.utils import line_pnt_intersect, pnts_through_surfaces, pnts_through_edge

class CACHE:
    def __init__(self, level, device):
        self.pnts = torch.zeros(3,3,level,0, device = device)
        self.info = torch.zeros(level,3,0, device = device, dtype=torch.int)

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

        mesh_ = copy.deepcopy(self.mesh)
        mesh_.mesh_to_device("cpu")
        msk_v = ~line_pnt_intersect(pnts_tx.to(device="cpu"), mesh_.v.T, mesh_)

        for i in range(pnts_tx.shape[1]):
            msk_v = ~line_pnt_intersect(pnts_tx[:,i].view(3,1).to(device="cpu"), mesh_.v.T, mesh_)
            node_prev = torch.cat([torch.zeros(3,2).to(device=pnts_tx.device), pnts_tx[:,i].view(3,1)], dim=1).view(3,3,1,1)
            info_prev = torch.zeros(1,3, dtype=torch.int32, device=pnts_tx.device).view(1,3,1)

            cache = self.next_nodes(1, i, node_prev, info_prev, msk_v)
            msk_v = self.validity_test(cache)

            cache = self.next_nodes(2, i, cache.pnts, cache.info, msk_v)
            msk_v = self.validity_test(cache)


    def validity_test(self, cache):

        msk_r = cache.info[-1,2,:] == 1
        node_r = cache.pnts[:,2,-1,msk_r]

        mesh_ = copy.deepcopy(self.mesh)
        mesh_.mesh_to_device("mps")

        msk_v_r = pnts_through_surfaces(mesh_, cache.info[-1, 0, msk_r], node_r, mesh_.v.T)

        msk_d = cache.info[-1,2,:] == 2
        node_d = cache.pnts[:,0:2,-1,msk_d]
        pnts_prev = cache.pnts[:,2,-1,msk_d]
        msk_v_d = pnts_through_edge(mesh_, cache.info[-1, 0:2, msk_d], node_d,  pnts_prev, mesh_.v.T)

        num_v_prev = torch.sum(msk_v_d)
        start_time = time.time()
        for i in range(1, msk_v_d.shape[0]):
            idc_v = torch.nonzero(msk_v_d[i, :], as_tuple=True)[0]
            msk_f = torch.any(torch.isin(self.mesh.f.to(device="cpu"), idc_v), dim=1).to(device=self.device)
            msk_f[cache.info[-1, 0, i]] = torch.zeros(1, dtype=torch.bool)

            mesh_ = copy.deepcopy(self.mesh)
            mesh_.mesh_to_device("cpu")
            mesh_.f = mesh_.f[msk_f, :]
            mesh_.n = mesh_.n[:, msk_f]
            mesh_.s = mesh_.s[:,:,msk_f]
            msk_v_d[i, msk_v_d[i, :].clone()] = ~line_pnt_intersect(node_d[:,0,i].view(-1,1), mesh_.v[idc_v,:].view(-1,3).T, mesh_)
            end_time = time.time()
            print(f"Iteration: {i}({(i+1)/msk_v_d.shape[0]*100: .2f} %), Valid surfaces: {torch.sum(msk_v_d[i,:])}, ({end_time-start_time: .2f} sec)")

        num_v_next = torch.sum(msk_v_d)
        print(f"Number of vertices before: {num_v_prev}, after: {num_v_next}")

        msk_v = torch.zeros(msk_v_r.shape[0]+msk_v_d.shape[0], msk_v_r.shape[1], dtype=torch.bool)
        msk_v[msk_r,:] = msk_v_r
        msk_v[msk_d,:] = msk_v_d

        p1 = Point([200 * cos(-230 / 180 * PI), 200 * sin(-230 / 180 * PI), 60], c='white')
        l1 = Light(p1, c='white')

        # p2 = Point(pnts_tx.to(device="cpu").numpy() + [0, 0, 1], c='white')
        # l2 = Light(p2, c='white')

        plotter = Plotter()

        msk_v_ = torch.any(msk_v_d, dim=0) # torch.any(msk_v, dim=0)
        idc_v = torch.nonzero(msk_v_, as_tuple=True)[0]
        msk_f = torch.any( torch.isin(self.mesh.f.to(device="cpu"), idc_v), dim=1)

        self.mesh.mesh_file.cellcolors[msk_f, 0:2] = self.mesh.mesh_file.cellcolors[msk_f, 0:2] * 0.5
        self.mesh.mesh_file.cellcolors[~msk_f, 3] = self.mesh.mesh_file.cellcolors[~msk_f, 3] * 0.1

        for i in range(0,msk_v_d.shape[0]):
            plotter += Line(node_d[:,0,i].numpy(), node_d[:,1,i].numpy()).color("green",0.5)

        plotter.show(self.mesh.mesh_file, p1, l1)

        return msk_v

    def next_nodes(self, level, idx_tx, node_prev, info_prev, msk_v):

        cache = CACHE(level, self.device)
        node_priority = torch.zeros(0)

        start_time = time.time()
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
                mesh_v = self.mesh.v[self.mesh.f[msk_f, 0], :].permute(1, 0).view(3, 1, 1,-1)   # (3 x 1 x 1 x num_r) first point of surfaces in msk_f
                mesh_n = self.mesh.n[:, msk_f].view(3, 1, 1, -1)                                # (3 x 1 x 1 x num_r) normal vector of surfaces in msk_f
                dist = torch.linalg.norm(mesh_v - node_prev[:,0:2,-1,i].view(3,2,1,1), dim=0)
                dist = torch.where(dist < 1, torch.tensor(1.0), dist)
                alignment = torch.min(
                    torch.abs( torch.sum(mesh_n * (mesh_v - node_prev[:,0:2,-1,i].view(3,2,1,1)), dim=0)
                               ) / torch.linalg.norm( mesh_v - node_prev[:,0:2,-1,i].view(3,2,1,1), dim=0),
                dim=0, keepdim=True)
                node_priority_ = self.mesh.fa[msk_f].view(1,1,-1) * alignment[0] /  torch.gather(dist, dim=0, index=alignment[1])
            else:
                msk_f[msk_f.clone()] = torch.sum( mesh_n * (node_prev[:,2,-1,i].view(3,1,1,1) - mesh_v), dim=0) > 0
                mesh_v = self.mesh.v[self.mesh.f[msk_f, 0], :].permute(1, 0).view(3, 1, 1,-1)   # (3 x 1 x 1 x num_r) first point of surfaces in msk_f
                mesh_n = self.mesh.n[:, msk_f].view(3, 1, 1, -1)                                # (3 x 1 x 1 x num_r) normal vector of surfaces in msk_f
                dist = torch.linalg.norm(mesh_v - node_prev[:, 2, -1, i].view(3, 1, 1, 1), dim=0)
                dist = torch.where(dist < 1, torch.tensor(1.0), dist)
                node_priority_ = self.mesh.fa[msk_f].view(1,1,-1) * torch.abs(
                    torch.sum(mesh_n * (mesh_v - node_prev[:,2,-1,i].view(3,1,1,1)), dim=0)
                ) / torch.linalg.norm(mesh_v - node_prev[:,2,-1,i].view(3,1,1,1), dim=0) / dist
            node_priority = torch.cat( (node_priority, node_priority_.view(-1)), dim=0)

            node_r = 2 * torch.sum( (mesh_v - node_prev[:,:,:,i].unsqueeze(-1)) * mesh_n, dim=0).unsqueeze(0) * mesh_n * (node_prev[:,:,:,i] != 0).unsqueeze(-1) + node_prev[:,:,:,i].unsqueeze(-1)
            num_r = node_r.shape[3]

            pnts = torch.cat((torch.zeros(3,2,1,num_r).to(device=self.device), node_prev[:,2,0,i].view(3,1,1,1).repeat(1,1,1,num_r)), dim=1)    # (3 x 3 x 1 x num_f)
            pnts = torch.cat((pnts[:,:,0:min(1,level-1),:], node_r), dim=2)                                                                     # (3 x 3 x level x num_f)

            info = torch.cat(
                (info_prev[0:level-1,:,i].unsqueeze(-1).repeat(1,1,num_r),
                 torch.cat((torch.where(msk_f)[0].view(1,-1), torch.zeros(1,num_r).to(device=self.device), torch.ones(1,num_r).to(device=self.device)), dim=0).view(1,3,-1)
                 ), dim=0                                 # Concatenate previous information (info_prev) with [idx_f 0 1], where '1' stands for indicating the reflection
            ).to(dtype = torch.int)                       # level x 3 x num_r

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
                        torch.sum( mesh_n * (node_prev[:,0:2,-1,i].view(3,2,1,1) - mesh_v), dim=0) <0, dim=0
                    ) * msk_d[ii,msk_d[ii,:]]
                    mesh_v = self.mesh.v[self.mesh.f[msk_d[ii, :], 0], :].permute(1, 0).view(3, 1, 1, -1)

                    vec1 = (self.mesh.s[:, ii % 3, msk_d[ii, :]] - self.mesh.s[:, (ii + 1) % 3,
                                                                   msk_d[ii, :]]) / torch.linalg.norm(
                        self.mesh.s[:, ii % 3, msk_d[ii, :]] - self.mesh.s[:, (ii + 1) % 3, msk_d[ii, :]], dim=0
                    ).unsqueeze(0)
                    vec2 = (mesh_v - node_prev[:,0:2,-1,i].view(3,2,1,1)) / torch.linalg.norm(
                        mesh_v - node_prev[:,0:2,-1,i].view(3,2,1,1), dim=0, keepdim=True
                    )
                    dist = torch.linalg.norm(mesh_v - node_prev[:,0:2,-1,i].view(3,2,1,1), dim=0)
                    dist = torch.where(dist < 1, torch.tensor(1.0), dist)
                    orthogonality = torch.min((1 - torch.sum(vec1.view(3,1,1,-1) * vec2, dim=0)**2), dim=0, keepdim=True)
                    node_priority_ = self.mesh.fl[ii,msk_d[ii,:]].view(1,1,-1) * orthogonality[0]/ torch.gather(dist, dim=0, index=orthogonality[1])
                else:
                    msk_d[ii,msk_d[ii,:].clone()] = (
                            torch.sum( mesh_n * (node_prev[:,2,-1,i].view(3,1,1,1) - mesh_v), dim=0) < 0
                    ) * msk_d[ii,msk_d[ii,:]]
                    mesh_v = self.mesh.v[self.mesh.f[msk_d[ii, :], 0], :].permute(1, 0).view(3, 1, 1, -1)

                    vec1 = (self.mesh.s[:,ii%3,msk_d[ii,:]] - self.mesh.s[:,(ii+1)%3,msk_d[ii,:]]) / torch.linalg.norm(
                        self.mesh.s[:, ii % 3, msk_d[ii,:]] - self.mesh.s[:, (ii + 1) % 3, msk_d[ii,:]], dim=0
                    ).unsqueeze(0)
                    vec2 = (mesh_v - node_prev[:,2,-1,i].view(3,1,1,1)) / torch.linalg.norm(
                        mesh_v - node_prev[:,2,-1,i].view(3,1,1,1), dim=0, keepdim=True
                    )
                    dist = torch.linalg.norm(mesh_v - node_prev[:,2,-1,i].view(3,1,1,1), dim=0)
                    dist = torch.where(dist < 1, torch.tensor(1.0), dist)
                    node_priority_ = self.mesh.fl[ii,msk_d[ii,:]].view(1,1,-1) * (1 - torch.sum(vec1.view(3,1,1,-1) * vec2, dim=0)**2)/ dist

                node_priority = torch.cat( (node_priority, node_priority_.view(-1)), dim=0)

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
                (info_prev[0:level-1, :, i].unsqueeze(-1).repeat(1, 1, num_d),
                 torch.cat((idc_d.view(1,-1), idc_e.view(1,-1), (2*torch.ones(1, num_d)).to(device=self.device)), dim=0).view(1, 3, -1)
                 ), dim=0               # Concatenate previous information (info_prev) with [idx_f 0 1], where '1' stands for indicating the reflection
            ).to(dtype=torch.int)       # level x 3 x num_d

            cache.pnts = torch.cat((cache.pnts, pnts), dim=3)
            cache.info = torch.cat((cache.info, info), dim=2)

            _, idx_top_node = torch.topk(node_priority,
                                         torch.tensor([node_priority.shape[0], 10000]).min()
                                         )
            cache.pnts = cache.pnts[:,:,:,idx_top_node]
            cache.info = cache.info[:,:,idx_top_node]
            node_priority = node_priority[idx_top_node]

            end_time = time.time()
            print(f"\r [{100*(i+1)/msk_v.shape[0]: 5.2f} %] {i+1} ({end_time-start_time:.2f} sec)")

        return cache