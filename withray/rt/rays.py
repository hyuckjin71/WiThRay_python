"""
Classes and Methods for RAYS.
"""
import time, copy
import torch
from vedo import Point, Line, Light, Plotter, Text3D, show, sin, cos, Points
from withray import PI
from withray.rt import Tx, Rx
from withray.rt.utils import *

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

        num_priority = torch.tensor(1e3, dtype=torch.int)

        for i in range(pnts_tx.shape[1]):
            msk_v = ~line_pnt_intersect(pnts_tx[:,i].view(3,1).to(device="cpu"), mesh_.v.T, mesh_)

            node_prev = torch.cat([torch.zeros(3,2).to(device=pnts_tx.device), pnts_tx[:,i].view(3,1)], dim=1).view(3,3,1,1)
            info_prev = torch.zeros(1,3, dtype=torch.int32, device=pnts_tx.device).view(1,3,1)

            plotter = Plotter()
            plotter += Text3D('Rx', pos=pnts_rx[:, 0].numpy())
            plotter += Point(pnts_rx[:,0].numpy()).color("red", 1)
            plotter += Point(pnts_tx[:,0].numpy()).color("red", 1)

            cache, cache_next, node_priority_prev = self.next_nodes(1, num_priority, node_prev, info_prev, torch.ones(1), msk_v)
            # cache_next = cache
            rays, msk_rays,_ = self.find_rays(1, cache, pnts_tx[:,i], pnts_rx)
            msk_v = self.validity_test(cache_next)

            rays_plot = rays[:,:,msk_rays[:,0],0]
            for ii in range(rays_plot.shape[2]):
                plotter += Line(pnts_tx[:,0].numpy(), rays_plot[:,0,ii].numpy()).color("cyan", 1)
                plotter += Line(rays_plot[:, 0, ii].numpy(), pnts_rx[:, 0].numpy()).color("cyan", 1)
                plotter += Point(rays_plot[:,0,ii].numpy()).color("cyan",1)

            cache, cache_next, node_priority_prev = self.next_nodes(2, num_priority, cache_next.pnts, cache_next.info, node_priority_prev, msk_v)
            # cache_next = cache
            rays, msk_rays, _ = self.find_rays(2, cache, pnts_tx[:,i], pnts_rx)
            msk_v = self.validity_test(cache_next)

            rays_plot = rays[:,:,msk_rays[:,0],0]
            for ii in range(rays_plot.shape[2]):
                plotter += Line(pnts_tx[:,0].numpy(), rays_plot[:,0,ii].numpy()).color("magenta", 1)
                plotter += Line(rays_plot[:,0,ii].numpy(), rays_plot[:,1,ii].numpy()).color("magenta", 1)
                plotter += Line(rays_plot[:,1,ii].numpy(), pnts_rx[:,0].numpy()).color("magenta", 1)
                plotter += Point(rays_plot[:,0,ii].numpy()).color("magenta",1)
                plotter += Point(rays_plot[:,1,ii].numpy()).color("magenta",1)

            cache, cache_next, node_priority_prev = self.next_nodes(3, num_priority, cache_next.pnts, cache_next.info, node_priority_prev, msk_v)
            rays, msk_rays, msk_rays_blocked = self.find_rays(3, cache, pnts_tx[:,i], pnts_rx)
            msk_v = self.validity_test(cache_next)

            rays_plot = rays[:,:,msk_rays[:,0],0]
            for ii in range(rays_plot.shape[2]):
                plotter += Line(pnts_tx[:,0].numpy(), rays_plot[:,0,ii].numpy()).color("blue", 1)
                plotter += Point(rays_plot[:, 0, ii].numpy()).color("blue", 1)
                for iii in range(0,2):
                    plotter += Line(rays_plot[:,iii,ii].numpy(), rays_plot[:,iii+1,ii].numpy()).color("blue", 1)
                    plotter += Point(rays_plot[:, iii+1, ii].numpy()).color("blue", 1)
                plotter += Line(rays_plot[:,2,ii].numpy(), pnts_rx[:,0].numpy()).color("blue", 1)

            rays_plot = rays[:, :, msk_rays_blocked[:, 0], 0]
            for ii in range(rays_plot.shape[2]):
                plotter += Line(pnts_tx[:, 0].numpy(), rays_plot[:, 0, ii].numpy()).color("yellow", 1)
                plotter += Point(rays_plot[:, 0, ii].numpy()).color("yellow", 1)
                for iii in range(0, 2):
                    plotter += Line(rays_plot[:, iii, ii].numpy(), rays_plot[:, iii + 1, ii].numpy()).color("yellow",
                                                                                                            1)
                    plotter += Point(rays_plot[:, iii + 1, ii].numpy()).color("yellow", 1)
                plotter += Line(rays_plot[:, 2, ii].numpy(), pnts_rx[:, 0].numpy()).color("yellow", 1)

            cache, cache_next, node_priority_prev = self.next_nodes(4, num_priority, cache_next.pnts, cache_next.info, node_priority_prev, msk_v)
            rays, msk_rays, msk_rays_blocked = self.find_rays(4, cache, pnts_tx[:, i], pnts_rx)
            msk_v = self.validity_test(cache_next)

            rays_plot = rays[:, :, msk_rays[:, 0], 0]
            for ii in range(rays_plot.shape[2]):
                plotter += Line(pnts_tx[:, 0].numpy(), rays_plot[:, 0, ii].numpy()).color("blue", 1)
                plotter += Point(rays_plot[:, 0, ii].numpy()).color("blue", 1)
                for iii in range(0, 3):
                    plotter += Line(rays_plot[:, iii, ii].numpy(), rays_plot[:, iii + 1, ii].numpy()).color("blue",
                                                                                                            1)
                    plotter += Point(rays_plot[:, iii + 1, ii].numpy()).color("blue", 1)
                plotter += Line(rays_plot[:, 3, ii].numpy(), pnts_rx[:, 0].numpy()).color("blue", 1)

            rays_plot = rays[:, :, msk_rays_blocked[:, 0], 0]
            for ii in range(rays_plot.shape[2]):
                plotter += Line(pnts_tx[:, 0].numpy(), rays_plot[:, 0, ii].numpy()).color("yellow", 1)
                plotter += Point(rays_plot[:, 0, ii].numpy()).color("yellow", 1)
                for iii in range(0, 3):
                    plotter += Line(rays_plot[:, iii, ii].numpy(), rays_plot[:, iii + 1, ii].numpy()).color(
                        "yellow",
                        1)
                    plotter += Point(rays_plot[:, iii + 1, ii].numpy()).color("yellow", 1)
                plotter += Line(rays_plot[:, 3, ii].numpy(), pnts_rx[:, 0].numpy()).color("yellow", 1)

            cache, cache_next, node_priority_prev = self.next_nodes(5, 1000, cache_next.pnts, cache_next.info, node_priority_prev, msk_v)
            rays, msk_rays,_ = self.find_rays(5, cache, pnts_tx[:,i], pnts_rx)
            # msk_v = self.validity_test(cache_next)

            rays_plot = rays[:,:,msk_rays[:,0],0]
            for ii in range(rays_plot.shape[2]):
                plotter += Line(pnts_tx[:,0].numpy(), rays_plot[:,0,ii].numpy()).color("blue", 1)
                plotter += Point(rays_plot[:, 0, ii].numpy()).color("blue", 1)
                for iii in range(0,4):
                    plotter += Line(rays_plot[:,iii,ii].numpy(), rays_plot[:,iii+1,ii].numpy()).color("blue", 1)
                    plotter += Point(rays_plot[:, iii+1, ii].numpy()).color("blue", 1)
                plotter += Line(rays_plot[:,4,ii].numpy(), pnts_rx[:,0].numpy()).color("blue", 1)


            p1 = Point([200 * cos(-230 / 180 * PI), 200 * sin(-230 / 180 * PI), 60], c='white')
            l1 = Light(p1, c='white')

            self.mesh.mesh_file.cellcolors[:, 3] = self.mesh.mesh_file.cellcolors[:, 3] * 0.8

            plotter.show(self.mesh.mesh_file, p1, l1)

            return rays

    def find_rays(self, level, cache, pnts_tx, pnts_rx):

        num_difrct = torch.sum(cache.info[:,2,:] == 2, dim=0)
        rays = cache.pnts[:,2,:,:].unsqueeze(-1).repeat(1,1,1,pnts_rx.shape[1])
        msk_rays = torch.ones(cache.pnts.shape[3], pnts_rx.shape[1], dtype=torch.bool)

        mesh_ = copy.deepcopy(self.mesh)
        mesh_.mesh_to_device("mps")

        for i in range(1,level+1):
            msk_node = num_difrct == i
            rays[:,level-i:level,msk_node,:], msk_rays_ = pnts_on_edge(cache.pnts[:,0:2,level-i:level,msk_node],   # pnts_edge
                                                                       cache.pnts[:,2,-1,msk_node],                # pnts1
                                                                       pnts_rx)                                    # pnts2
            msk_rays[msk_node,:] = msk_rays[msk_node,:] & msk_rays_

        for i in range(level,0,-1):
            msk_node = cache.info[i-1,2,:] == 1
            if i == level:
                pnts_end = pnts_rx.view(3,1,-1)
            else:
                pnts_end = rays[:,i,msk_node,:]
            rays[:,0:i,msk_node,:], msk_rays_ = pnts_on_surface(mesh_,
                                                                cache.info[level-1,0,msk_node],
                                                                rays[:,0:i,msk_node,:],
                                                                pnts_end)
            msk_rays[msk_node,:] = msk_rays[msk_node,:] & msk_rays_

        mesh_ = copy.deepcopy(self.mesh)
        mesh_.mesh_to_device("cpu")

        msk_rays_blocked = copy.deepcopy(msk_rays)

        msk_rays[msk_rays.clone()] = (
                msk_rays[msk_rays.clone()] &
                ~line_pnt_intersect(pnts_tx.view(3, -1), rays[:, 0, msk_rays], mesh_)
        )

        for i in range(msk_rays.shape[1]):
            msk_rays[msk_rays[:,i],i] = (
                msk_rays[msk_rays[:,i],i] &
                ~line_pnt_intersect(pnts_rx[:,i].view(3, -1), rays[:, -1, msk_rays[:,i], i] ,mesh_)
            )

        msk_rays_blocked[msk_rays_blocked.clone()] = (
                msk_rays_blocked[msk_rays_blocked.clone()] &
                line_pnt_intersect(pnts_tx.view(3, -1), rays[:, 0, msk_rays_blocked], mesh_)
        )

        for i in range(msk_rays_blocked.shape[1]):
            msk_rays_blocked[msk_rays_blocked[:,i],i] = (
                msk_rays_blocked[msk_rays_blocked[:,i],i] &
                line_pnt_intersect(pnts_rx[:,i].view(3, -1), rays[:, -1, msk_rays_blocked[:,i], i] ,mesh_)
            )

        return rays, msk_rays, msk_rays_blocked


    def validity_test(self, cache):

        msk_r = cache.info[-1,2,:] == 1
        node_r = cache.pnts[:,:,-1,msk_r]
        info_r = cache.info[-1,0,msk_r]

        msk_difrct = torch.any(node_r[0, 0:2, :] != 0, dim=0)
        msk_reflct = ~msk_difrct

        mesh_ = copy.deepcopy(self.mesh)
        msk_v_r = torch.zeros(node_r.shape[2], mesh_.v.shape[0], dtype=torch.bool)
        if torch.any(msk_reflct):
            start_time = time.time()
            mesh_.mesh_to_device("mps")
            msk_v_r1 = pnts_through_surfaces(mesh_, info_r[msk_reflct], node_r[:,2,msk_reflct].view(3,-1), mesh_.v.T)
            i1, i2 = torch.where(msk_v_r1)
            dist = 1 / torch.linalg.norm( node_r[:,2,msk_reflct][:,i1] - mesh_.v[i2,:].T.to(device="cpu"), dim=0)
            _, idx_top_k = torch.topk(dist, min(torch.tensor(1e5, dtype=torch.int), dist.shape[0]))
            msk_updated = torch.zeros_like(msk_v_r1, dtype=torch.bool)
            msk_updated[i1[idx_top_k],i2[idx_top_k]] = True
            msk_v_r1 = msk_updated

            num1 = torch.sum(msk_v_r1)
            i1, i2 = torch.where(msk_v_r1)
            print("Reflecting points are validated.")

            mesh_.mesh_to_device("cpu")
            msk_v_r1[msk_v_r1.clone()] = ~line_pnt_intersect(node_r[:,2,msk_reflct][:,i1].view(3,-1), mesh_.v[i2,:].T, mesh_, dim=1, idc_f_except=info_r[i1])
            num2 = torch.sum(msk_v_r1)
            msk_v_r[msk_reflct, :] = msk_v_r1
            end_time = time.time()
            print(f"[Level: {cache.info.shape[0]}] Validity Test 1: {num2}/{num1}, Time: {end_time - start_time: .2f} sec")

        if torch.any(msk_difrct):
            start_time = time.time()
            mesh_.mesh_to_device("mps")
            num_difrct = torch.sum(cache.info[:,2,msk_r][:,msk_difrct] == 2, dim=0)
            msk_v_r2 = torch.zeros(msk_difrct.sum(), mesh_.v.shape[0], dtype=torch.bool)
            level = cache.info.shape[0]
            for i in range(1, level+1):
                msk_node = num_difrct == i
                if msk_node.sum() != 0:
                    blck_size = torch.ceil(torch.tensor(2e6) / torch.sum(msk_node)).to(dtype=torch.int)
                    ii = 0
                    end_idx = 0
                    pnts_end = torch.zeros(3,0)
                    i1_ = torch.zeros(0, dtype=torch.int)
                    i2_ = torch.zeros(0, dtype=torch.int)
                    while end_idx < mesh_.v.shape[0]:
                        start_idx = ii * blck_size
                        end_idx = min((ii+1) * blck_size, mesh_.v.shape[0])
                        ii += 1
                        pnts_on_edge_, msk_on = pnts_on_edge(cache.pnts[:,:,:,msk_r][:,:,:,msk_difrct][:,0:2,level-i:level,msk_node],
                                                             cache.pnts[:,:,:,msk_r][:,:,:,msk_difrct][:,2,-1,msk_node],
                                                             mesh_.v[start_idx:end_idx,:].T)
                        msk_v_r2[msk_node,start_idx:end_idx] = pnts_through_surfaces(mesh_, info_r[msk_difrct][msk_node],
                                                                                     pnts_on_edge_[:,-1,:,:],
                                                                                     mesh_.v[start_idx:end_idx,:].T) & msk_on
                        i1, i2 = torch.where(msk_v_r2[msk_node,start_idx:end_idx])
                        pnts_end = torch.cat( (pnts_end, pnts_on_edge_[:,-1,i1,i2]), dim=1)
                        i1_ = torch.cat( (i1_, i1), dim=0)
                        i2_ = torch.cat( (i2_, i2+start_idx), dim=0)

                    dist = 1 / torch.linalg.norm( pnts_end - mesh_.v[i2_,:].T.to(device="cpu"), dim=0)
                    _, idx_top_k = torch.topk(dist, min(torch.tensor(1e5, dtype=torch.int),dist.shape[0]))
                    msk_updated = torch.zeros_like(msk_v_r2[msk_node,:], dtype=torch.bool)
                    msk_updated[i1_[idx_top_k],i2_[idx_top_k]] = True
                    msk_v_r2[msk_node,:] = msk_updated

            num1 = torch.sum(msk_v_r2)
            i1, i2 = torch.where(msk_v_r2)
            print("Reflecting points are validated.")

            mesh_.mesh_to_device("cpu")
            msk_v_r2[msk_v_r2.clone()] = (
                ~line_pnt_intersect(node_r[:,0,msk_difrct][:,i1].view(3,-1), mesh_.v[i2,:].T, mesh_, dim=1, idc_f_except=info_r[i1]) |
                ~line_pnt_intersect(node_r[:,1,msk_difrct][:,i1].view(3,-1), mesh_.v[i2,:].T, mesh_, dim=1, idc_f_except=info_r[i1])
            )
            num2 = torch.sum(msk_v_r2)
            msk_v_r[msk_difrct, :] = msk_v_r2
            end_time = time.time()
            print(f"[Level: {cache.info.shape[0]}] Validity Test 2: {num2}/{num1}, Time: {end_time - start_time: .2f} sec")

        msk_d = cache.info[-1,2,:] == 2
        node_d = cache.pnts[:,:,:,msk_d]
        info_d = cache.info[:,:,msk_d]

        start_time = time.time()
        mesh_.mesh_to_device("mps")
        num_difrct = torch.sum(info_d[:,2,:] == 2, dim=0)
        msk_v_d = torch.zeros(info_d.shape[2], mesh_.v.shape[0], dtype=torch.bool)
        level = info_d.shape[0]
        for i in range(1, level+1):
            msk_node = num_difrct == i
            msk_v_d[msk_node,:], pnts_end = pnts_through_edge(i, mesh_.n[:,info_d[level-i:level,0,msk_node]].view(3,i,-1), node_d[:,0:2,level-i:level,msk_node], node_d[:,2,-1,msk_node], mesh_.v.T)
            i1, i2 = torch.where(msk_v_d[msk_node,:])
            dist = 1 / torch.linalg.norm( pnts_end - mesh_.v[i2,:].T.to(device="cpu"), dim=0)
            _, idx_top_k = torch.topk(dist, min(torch.tensor(1e5, dtype=torch.int), dist.shape[0]))
            msk_updated = torch.zeros_like(msk_v_d[msk_node, :], dtype=torch.bool)  # False로 초기화
            msk_updated[i1[idx_top_k], i2[idx_top_k]] = True
            msk_v_d[msk_node,:] = msk_updated

        num1 = torch.sum(msk_v_d)
        i1, i2 = torch.where(msk_v_d)
        print("Diffracting points are validated.")

        mesh_.mesh_to_device("cpu")
        # msk_v_d = v_priority(mesh_, msk_v_d, node_d)
        msk_v_d[msk_v_d.clone()] = (
            ~line_pnt_intersect(node_d[:,0,-1,:][:,i1].view(3,-1), mesh_.v[i2,:].T, mesh_, dim=1) |
            ~line_pnt_intersect(node_d[:,1,-1,:][:,i1].view(3,-1), mesh_.v[i2,:].T, mesh_, dim=1)
        )

        num2 = torch.sum(msk_v_d)
        end_time = time.time()
        print(f"[Level: {cache.info.shape[0]}] Validity Test 3: {num2}/{num1}, Time: {end_time - start_time: .2f} sec")

        msk_v = torch.zeros(msk_v_r.shape[0]+msk_v_d.shape[0], msk_v_r.shape[1], dtype=torch.bool)
        msk_v[msk_r,:] = msk_v_r
        msk_v[msk_d,:] = msk_v_d

        return msk_v

    def next_nodes(self, level, num_next_node, node_prev, info_prev, node_priority_prev, msk_v):

        node_prev = node_prev.to(device=self.device)
        node_priority_prev = node_priority_prev.to(device=self.device)
        info_prev = info_prev.to(device=self.device)
        msk_v = msk_v.to(device=self.device)

        mesh_ = copy.deepcopy(self.mesh)
        mesh_.mesh_to_device(self.device)
        cache = CACHE(level, self.device)
        cache_next = CACHE(level, self.device)
        node_priority = torch.zeros(0)

        start_time = time.time()
        blck_size = 100
        i = 0
        end_idx = 0
        while end_idx < msk_v.shape[0]:
            start_idx = i * blck_size
            end_idx = min((i+1)*blck_size, msk_v.shape[0])
            i += 1

            msk_f = torch.any(msk_v[start_idx:end_idx, self.mesh.f], dim=2)
            i1, i2 = torch.where(msk_f)
            i1 += start_idx

            mesh_v = mesh_.s[:,0,i2].unsqueeze(1)
            mesh_n = mesh_.n[:,i2].unsqueeze(1)


            # (Reflection from diffraction) ===============================================================================

            msk_difrct = torch.any(node_prev[:,0:2,-1,i1] != 0, dim=[0,1])
            msk_reflct = ~msk_difrct

            '''
            Check whether the next vertices form a valid reflective surface
            based on whether the previous node (node_prev) was a reflecting or diffracting point.
    
            The next reflective surface must face the previous node, meaning the directional vector
            should align with the surface's normal vector.
            '''

            msk_difrct[msk_difrct.clone()] = torch.any(
                torch.sum( mesh_n[:,:,msk_difrct] * (node_prev[:,0:2,-1,i1][:,:,msk_difrct] - mesh_v[:,:,msk_difrct]), dim=0) > 0, dim=0
            ) & ~torch.any( torch.all( mesh_v[:,:,msk_difrct] == node_prev[:,0:2,-1,i1][:,:,msk_difrct], dim=0), dim=0)

            dist = torch.linalg.norm(mesh_v[:,:,msk_difrct] - node_prev[:,0:2,-1,i1][:,:,msk_difrct], dim=0)
            dist = torch.where(dist < 1, torch.tensor(1.0), dist)
            alignment = torch.min(
                torch.abs( torch.sum(mesh_n[:,:,msk_difrct] * (mesh_v[:,:,msk_difrct] - node_prev[:,0:2,-1,i1][:,:,msk_difrct]), dim=0)
                           ) / torch.linalg.norm( mesh_v[:,:,msk_difrct] - node_prev[:,0:2,-1,i1][:,:,msk_difrct], dim=0),
                dim=0, keepdim=True)

            node_priority_ = mesh_.fa[i2][msk_difrct].unsqueeze(0) * alignment[0] / torch.gather(dist, dim=0, index=alignment[1])
            node_priority_ = node_priority_.squeeze() * node_priority_prev[i1][msk_difrct]
            node_priority = torch.cat( (node_priority, node_priority_), dim=0)

            node_r = 2 * torch.sum(
                (mesh_v[:,:,msk_difrct].unsqueeze(2) - node_prev[:,:,:,i1][:,:,:,msk_difrct]) * mesh_n[:,:,msk_difrct].unsqueeze(2), dim=0
            ) * mesh_n[:,:,msk_difrct].unsqueeze(2) * (node_prev[:,:,:,i1][:,:,:,msk_difrct] != 0) + node_prev[:,:,:,i1][:,:,:,msk_difrct]
            num_r = node_r.shape[3]

            pnts = torch.cat((torch.zeros(3,2,1,num_r).to(device=self.device),
                 node_prev[:,2,0,i1][:,msk_difrct].view(3,1,1,-1)), dim=1)
            pnts = torch.cat((pnts[:,:,0:min(1,level-1),:], node_r), dim=2)

            info = torch.cat(
                (info_prev[0:level-1,:,i1][:,:,msk_difrct],
                 torch.cat((i2[msk_difrct].view(1,-1), torch.zeros(1, num_r).to(device=self.device),
                            torch.ones(1, num_r).to(device=self.device)), dim=0).view(1,3,-1)), dim=0
            ).to(dtype=torch.int)

            cache.pnts = torch.cat((cache.pnts, pnts), dim=3)
            cache.info = torch.cat((cache.info, info), dim=2)

            # (Reflection from reflection) ================================================================================

            msk_reflct[msk_reflct.clone()] = (torch.sum( mesh_n[:,:,msk_reflct] * (node_prev[:,2,-1,i1][:,msk_reflct].unsqueeze(1) - mesh_v[:,:,msk_reflct]), dim=0) > 0
                                              ) & ~torch.all( mesh_v[:,0,msk_reflct] == node_prev[:,2,-1,i1][:,msk_reflct], dim=0)

            dist = torch.linalg.norm(mesh_v[:,:,msk_reflct] - node_prev[:,2,-1,i1][:,msk_reflct].unsqueeze(1), dim=0)
            dist = torch.where(dist < 1, torch.tensor(1.0), dist)
            alignment = torch.abs(
                torch.sum(mesh_n[:,:,msk_reflct] * (mesh_v[:,:,msk_reflct] - node_prev[:,2,-1,i1][:,msk_reflct].unsqueeze(1)), dim=0)
            ) / torch.linalg.norm(mesh_v[:,:,msk_reflct] - node_prev[:,2,-1,i1][:,msk_reflct].unsqueeze(1), dim=0)

            node_priority_ = mesh_.fa[i2][msk_reflct].unsqueeze(0) * alignment / dist
            node_priority_ = node_priority_.squeeze() * node_priority_prev[i1][msk_reflct]
            node_priority = torch.cat( (node_priority, node_priority_), dim=0)

            node_r = 2 * torch.sum(
                (mesh_v[:,:,msk_reflct].unsqueeze(2) - node_prev[:,:,:,i1][:,:,:,msk_reflct]) * mesh_n[:,:,msk_reflct].unsqueeze(2), dim=0
            ) * mesh_n[:,:,msk_reflct].unsqueeze(2) * (node_prev[:,:,:,i1][:,:,:,msk_reflct] != 0) + node_prev[:,:,:,i1][:,:,:,msk_reflct]
            num_r = node_r.shape[3]

            pnts = torch.cat((torch.zeros(3,2,1,num_r).to(device=self.device),
                              node_prev[:,2,0,i1][:,msk_reflct].view(3,1,1,-1)), dim=1)
            pnts = torch.cat((pnts[:,:,0:min(1,level-1),:], node_r), dim=2)

            info = torch.cat(
                (info_prev[0:level-1,:,i1][:,:,msk_reflct],
                 torch.cat((i2[msk_reflct].view(1,-1), torch.zeros(1,num_r).to(device=self.device),
                            torch.ones(1, num_r).to(device=self.device)), dim=0).view(1,3,-1)), dim=0
            ).to(dtype=torch.int)

            cache.pnts = torch.cat((cache.pnts, pnts), dim=3)
            cache.info = torch.cat((cache.info, info), dim=2)

            # (Diffraction from diffraction) ==============================================================================

            msk_d = torch.stack(
                [torch.any(msk_v[start_idx:end_idx, mesh_.f[:,[0,1]] ], dim=2),
                 torch.any(msk_v[start_idx:end_idx, mesh_.f[:,[1,2]] ], dim=2),
                 torch.any(msk_v[start_idx:end_idx, mesh_.f[:,[2,0]] ], dim=2)], dim=2
            ).to(device=self.device)

            i1, i2, i3 = torch.where(msk_d)
            i1 += start_idx

            mesh_v = mesh_.s[:,0,i2].unsqueeze(1)
            mesh_n = mesh_.n[:,i2].unsqueeze(1)

            msk_difrct = torch.any(node_prev[:,0:2,-1,i1] != 0, dim=[0,1])
            msk_reflct = ~msk_difrct

            msk_difrct[msk_difrct.clone()] = torch.any(
                torch.sum( mesh_n[:,:,msk_difrct] * (node_prev[:,0:2,-1,i1][:,:,msk_difrct] - mesh_v[:,:,msk_difrct]), dim=0) < 0, dim=0
            )  & ~torch.any( torch.all( mesh_v[:,:,msk_difrct] == node_prev[:,0:2,-1,i1][:,:,msk_difrct], dim=0), dim=0)

            vec1 = (mesh_.s[:,i3,i2][:,msk_difrct] - mesh_.s[:,(i3+1)%3,i2][:,msk_difrct]) / torch.linalg.norm(
                mesh_.s[:, i3, i2][:, msk_difrct] - mesh_.s[:, (i3 + 1) % 3, i2][:, msk_difrct], dim=0, keepdim=True
            )
            vec2 = (mesh_v[:,:,msk_difrct] - node_prev[:,0:2,-1,i1][:,:,msk_difrct]) / torch.linalg.norm(
                mesh_v[:, :, msk_difrct] - node_prev[:, 0:2, -1, i1][:, :, msk_difrct], dim=0, keepdim=True
            )
            dist = torch.linalg.norm(mesh_v[:,:,msk_difrct] - node_prev[:,0:2,-1,i1][:,:,msk_difrct], dim=0)
            dist = torch.where(dist < 1, torch.tensor(1.0), dist)
            orthogonality = torch.min(
                (1 - torch.sum(vec1.unsqueeze(1) * vec2, dim=0) ** 2), dim=0, keepdim=True
            )

            node_priority_ = mesh_.fl[i3,i2][msk_difrct] * orthogonality[0] / torch.gather(dist, dim=0, index=orthogonality[1])
            node_priority_ = node_priority_.squeeze() * node_priority_prev[i1][msk_difrct]
            node_priority = torch.cat((node_priority, node_priority_), dim=0)

            node_d = torch.cat( (mesh_.s[:,i3,i2][:,msk_difrct].unsqueeze(1),
                                 mesh_.s[:,(i3+1)%3,i2][:,msk_difrct].unsqueeze(1)), dim=1).unsqueeze(2)
            num_d = node_d.shape[3]

            pnts = torch.cat((node_d, node_prev[:,2,-1,i1][:,msk_difrct].view(3,1,1,-1)), dim=1)
            pnts = torch.cat((node_prev[:,:,0:level-1,i1][:,:,:,msk_difrct], pnts), dim=2)

            info = torch.cat(
                (info_prev[0:level-1,:,i1][:,:,msk_difrct],
                 torch.cat((i2[msk_difrct].unsqueeze(0),
                            i3[msk_difrct].unsqueeze(0),
                            2 * torch.ones(1,num_d).to(device=self.device)), dim=0).view(1,3,-1)), dim=0
            ).to(dtype=torch.int)

            cache.pnts = torch.cat((cache.pnts, pnts), dim=3)
            cache.info = torch.cat((cache.info, info), dim=2)

            # (Diffraction from reflection) ===============================================================================

            msk_reflct[msk_reflct.clone()] = (torch.sum( mesh_n[:,:,msk_reflct] * (node_prev[:,2,-1,i1][:,msk_reflct].unsqueeze(1) - mesh_v[:,:,msk_reflct]), dim=0) < 0
                                              ) & ~torch.all( mesh_v[:,0,msk_reflct] == node_prev[:,2,-1,i1][:,msk_reflct], dim=0)

            vec1 = (mesh_.s[:,i3%3,i2][:,msk_reflct] - mesh_.s[:,(i3+1)%3,i2][:,msk_reflct]) / torch.linalg.norm(
                mesh_.s[:, i3 % 3, i2][:, msk_reflct] - mesh_.s[:, (i3 + 1) % 3, i2][:, msk_reflct], dim=0, keepdim=True
            )
            vec2 = (mesh_v[:,:,msk_reflct] - node_prev[:,2,-1,i1][:,msk_reflct].unsqueeze(1)) / torch.linalg.norm(
                mesh_v[:, :, msk_reflct] - node_prev[:, 2, -1, i1][:, msk_reflct].unsqueeze(1), dim=0, keepdim=True
            )
            dist = torch.linalg.norm(mesh_v[:,:,msk_reflct] - node_prev[:,2,-1,i1][:,msk_reflct].unsqueeze(1), dim=0)
            dist = torch.where(dist < 1, torch.tensor(1.0), dist)
            orthogonality = 1 - torch.sum(vec1.view(3,1,-1) * vec2, dim=0) ** 2

            node_priority_ = mesh_.fl[i3,i2][msk_reflct] * orthogonality / dist
            node_priority_ = node_priority_.squeeze() * node_priority_prev[i1][msk_reflct]
            node_priority = torch.cat((node_priority, node_priority_), dim=0)

            node_d = torch.cat((mesh_.s[:,i3,i2][:,msk_reflct].unsqueeze(1),
                                mesh_.s[:,(i3+1)%3,i2][:,msk_reflct].unsqueeze(1)), dim=1).unsqueeze(2)
            num_d = node_d.shape[3]

            pnts = torch.cat((node_d, node_prev[:,2,-1,i1][:,msk_reflct].view(3,1,1,-1)), dim=1)
            pnts = torch.cat((node_prev[:,:,0:level-1,i1][:,:,:,msk_reflct], pnts), dim=2)

            info = torch.cat(
                (info_prev[0:level-1,:,i1][:,:,msk_reflct],
                 torch.cat((i2[msk_reflct].unsqueeze(0),
                            i3[msk_reflct].unsqueeze(0),
                            2 * torch.ones(1,num_d).to(device=self.device)), dim=0).view(1,3,-1)), dim=0
            ).to(dtype=torch.int)

            cache.pnts = torch.cat((cache.pnts, pnts), dim=3)
            cache.info = torch.cat((cache.info, info), dim=2)

        _, idx_top_node = torch.topk(node_priority,
                                     min(node_priority.shape[0], num_next_node)
                                     )

        cache_next.pnts = cache.pnts[:, :, :, idx_top_node]
        cache_next.info = cache.info[:, :, idx_top_node]
        node_priority = node_priority[idx_top_node]

        end_time = time.time()
        print(f"\n[Level: {level}] VPI Search, Time: {end_time - start_time:.2f} sec")

        return cache, cache_next, node_priority