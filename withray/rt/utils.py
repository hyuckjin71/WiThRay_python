"""
Ray tracer utilities
"""

import torch
import time
from vedo import *

# def inverse_mat(mat_a):
#     """
#     Compute the inverse of multiple 3x3 matrices in blocks.
#
#     Args:
#         mat_a: Tensor of shape (3, 3, H, W) or (3, 3, N).
#                mat_a must be on GPU or CPU (torch.Tensor).
#
#     Returns:
#         inv_mat_a: Tensor of shape (3, 3, H, W) or (3, 3, N).
#     """
#     blck_size = int(1e5)  # Block size
#     device = mat_a.device  # Ensure GPU compatibility
#
#     # Total number of matrices
#     total_matrices = mat_a.shape[2] if mat_a.dim() == 3 else mat_a.shape[2] * mat_a.shape[3]
#
#     # Output tensor initialization
#     inv_mat_a = torch.zeros((3, 3, total_matrices), device=device)
#
#     num_blck = (total_matrices + blck_size - 1) // blck_size  # Number of blocks
#
#     for i in range(num_blck):
#         # Index range for the current block
#         start_idx = i * blck_size
#         end_idx = min((i + 1) * blck_size, total_matrices)
#
#         # Extract block of matrices
#         if mat_a.dim() == 4:
#             idc = torch.arange(start_idx, end_idx, device=device)
#             mat_a_block = mat_a[:, :, idc // mat_a.shape[3], idc % mat_a.shape[3]]
#         else:
#             mat_a_block = mat_a[:, :, start_idx:end_idx]  # Shape: (3, 3, block_size)
#
#         # Compute determinants
#         det = (
#                 mat_a_block[0, 0] * (mat_a_block[1, 1] * mat_a_block[2, 2] - mat_a_block[1, 2] * mat_a_block[2, 1]) -
#                 mat_a_block[0, 1] * (mat_a_block[1, 0] * mat_a_block[2, 2] - mat_a_block[1, 2] * mat_a_block[2, 0]) +
#                 mat_a_block[0, 2] * (mat_a_block[1, 0] * mat_a_block[2, 1] - mat_a_block[1, 1] * mat_a_block[2, 0])
#         )
#
#         # Compute the inverse using the adjugate method
#         adjugate = torch.stack([
#             torch.stack([
#                 mat_a_block[1, 1] * mat_a_block[2, 2] - mat_a_block[1, 2] * mat_a_block[2, 1],
#                 mat_a_block[0, 2] * mat_a_block[2, 1] - mat_a_block[0, 1] * mat_a_block[2, 2],
#                 mat_a_block[0, 1] * mat_a_block[1, 2] - mat_a_block[0, 2] * mat_a_block[1, 1]
#             ]),
#             torch.stack([
#                 mat_a_block[1, 2] * mat_a_block[2, 0] - mat_a_block[1, 0] * mat_a_block[2, 2],
#                 mat_a_block[0, 0] * mat_a_block[2, 2] - mat_a_block[0, 2] * mat_a_block[2, 0],
#                 mat_a_block[0, 2] * mat_a_block[1, 0] - mat_a_block[0, 0] * mat_a_block[1, 2]
#             ]),
#             torch.stack([
#                 mat_a_block[1, 0] * mat_a_block[2, 1] - mat_a_block[1, 1] * mat_a_block[2, 0],
#                 mat_a_block[0, 1] * mat_a_block[2, 0] - mat_a_block[0, 0] * mat_a_block[2, 1],
#                 mat_a_block[0, 0] * mat_a_block[1, 1] - mat_a_block[0, 1] * mat_a_block[1, 0]
#             ])
#         ], dim=0)
#
#         inv_block = adjugate / det.unsqueeze(0).unsqueeze(0)
#         inv_mat_a[:, :, start_idx:end_idx] = inv_block
#
#     # Reshape to original dimensions
#     if mat_a.dim() == 4:
#         inv_mat_a = inv_mat_a.view(3, 3, mat_a.shape[2], mat_a.shape[3])
#
#     return inv_mat_a
#
# def sorted_inv_s(mesh, pnts):
#
#     if pnts.ndim < 3:
#         pnts = pnts.unsqueeze(-1)
#     if pnts.ndim < 4:
#         pnts = pnts.unsqueeze(-1)
#
#     inv_s = inverse_mat(mesh.s.unsqueeze(-1) - pnts.permute(0,2,3,1))
#     area_s = torch.cross(mesh.s[:,0,:]-mesh.s[:,1,:], mesh.s[:,2,:]-mesh.s[:,1,:], dim=0)
#     area_s = torch.norm(area_s, p=2, dim=0)
#     dir_s = pnts - torch.mean(mesh.s.unsqueeze(-1), dim=1, keepdim=True)
#     norm_dir_s = torch.linalg.norm(dir_s, ord=2, dim=0, keepdim=True)
#     dir_s = dir_s / norm_dir_s**4 / torch.abs(dir_s[2,:,:,:]).unsqueeze(0)**0.5
#     area_s = area_s.unsqueeze(0).unsqueeze(-1) * torch.abs(torch.sum(mesh.n.unsqueeze(-1).permute(0,2,1).unsqueeze(-1) * dir_s, dim=0))
#
#     _,idx_sorted = torch.sort(torch.mean(area_s, dim=0).squeeze(), descending=True)
#     inv_s = inv_s[:,:,idx_sorted,:]
#
#     return inv_s

def line_pnt_intersect(pnts1, pnts2, mesh,
                       msk_in_ = None,
                       dim=2,
                       idc_f_except = None):

    if dim == 1:
        msk_in = torch.zeros(pnts1.shape[1], dtype=torch.bool, device=pnts1.device)
        if msk_in_ is None:
            msk_in_ = torch.zeros(pnts1.shape[1], dtype=torch.bool, device=pnts1.device)
    else:
        msk_in = torch.zeros(pnts1.shape[1], pnts2.shape[1], dtype=torch.bool, device=pnts1.device)
        if msk_in_ is None:
            msk_in_ = torch.zeros(pnts1.shape[1], pnts2.shape[1], dtype=torch.bool, device=pnts1.device)

    box_blob = torch.tensor(
        [mesh.v[:, 0].min(),
         mesh.v[:, 1].min(),
         mesh.v[:, 2].min(),
         mesh.v[:, 0].max() - mesh.v[:, 0].min(),
         mesh.v[:, 1].max() - mesh.v[:, 1].min(),
         mesh.v[:, 2].max() - mesh.v[:, 2].min()]
    )

    msk_in = bvh_algorithm(msk_in, mesh, pnts1, pnts2, torch.zeros(3,1), ~msk_in_.to(device=pnts1.device), 1, box_blob, dim, idc_f_except)

    return msk_in

def bvh_algorithm(msk_in, mesh, pnts1, pnts2, msk_box_prev, msk_prev, level, box_blob, dim, idc_f_except):

    msk_box = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ]).T

    for i in range(8):
        msk_box_ = msk_box_prev + msk_box[:,i].unsqueeze(-1) / 2**level
        box_blob_ = box_blob[:3].unsqueeze(-1) + box_blob[3:6].unsqueeze(-1) * (msk_box_ + msk_box/2**level)

        if dim == 1:
            i1 = torch.where(msk_prev)[0]
            msk_line = msk_prev.clone()
            msk_line[msk_line.clone()] = line_cross_box(pnts1[:, i1], pnts2[:, i1], box_blob_)
            i1 = torch.where(msk_line)[0]
            i2 = i1
        else:
            i1, i2 = torch.where(msk_prev)
            msk_line = msk_prev.clone()
            msk_line[msk_line.clone()] = line_cross_box(pnts1[:, i1], pnts2[:, i2], box_blob_)
            i1, i2 = torch.where(msk_line)

        msk_f = torch.any(
            pnts_in_cube(mesh.s, box_blob_), dim=0
        )

        if torch.sum(msk_f) * torch.sum(msk_line) < 1e7:
            if torch.any(msk_line):
                msk_in[msk_line] = msk_in[msk_line] | torch.squeeze(
                    pnts_in_surfaces(mesh, msk_f,
                                     pnts1[:,i1].view(3,1,1,-1),
                                     pnts2[:,i2].view(3,1,1,-1),
                                     idc_f_except[i1] if idc_f_except is not None else None)
                )
        else:
            msk_in = bvh_algorithm(msk_in, mesh, pnts1, pnts2, msk_box_, msk_line, level+1, box_blob, dim, idc_f_except)

        msk_prev = msk_prev & ~msk_in

    return msk_in

def pnts_in_surfaces(mesh, msk_f, pnts1, pnts2, idc_f_except = None):

    ratio_intersect = torch.sum(
        (pnts2 - mesh.s[:,0,msk_f].view(3,-1,1,1)) * mesh.n[:,msk_f].view(3,-1,1,1),
    dim=0) / torch.sum(
        (pnts2-pnts1) * mesh.n[:,msk_f].view(3,-1,1,1),
    dim=0)

    pnts_intersect = (pnts1 - pnts2) * ratio_intersect.unsqueeze(0) + pnts2
    msk_in = (ratio_intersect >= 0) & (ratio_intersect <= 1)

    dir1 = mesh.s[:,0,msk_f] - mesh.s[:,1,msk_f]
    dir2 = mesh.s[:,2,msk_f] - mesh.s[:,1,msk_f]

    det_inv = dir1[0] * dir2[1] - dir1[1]  * dir2[0]

    r1 = (
        dir2[1].view(-1,1,1) * (pnts_intersect[0] - mesh.s[0, 1, msk_f].view(-1,1,1)) -
        dir2[0].view(-1,1,1) * (pnts_intersect[1] - mesh.s[1, 1, msk_f].view(-1,1,1))
    ) / det_inv.view(-1,1,1)

    r2 = (
        dir1[0].view(-1,1,1) * (pnts_intersect[1] - mesh.s[1, 1, msk_f].view(-1,1,1)) -
        dir1[1].view(-1,1,1) * (pnts_intersect[0] - mesh.s[0, 1, msk_f].view(-1,1,1))
    ) / det_inv.view(-1,1,1)

    msk_in = msk_in & (r1 > 0) & (r2 > 0) & (r1+r2 < 1)
    if idc_f_except is not None:
        msk_in = msk_in & (
            torch.nonzero(msk_f, as_tuple=True)[0].unsqueeze(-1) != idc_f_except.unsqueeze(0)
        ).unsqueeze(1)
    msk_in = torch.any(msk_in, dim=0)

    return msk_in

def pnts_through_surfaces(mesh, idc_f, pnts1, pnts2, start_v=0):

    if pnts1.dim() == 2:
        pnts1_ = pnts1.unsqueeze(-1).to(device="mps")                     # (3 x num_pnts1 x 1)
    else:
        pnts1_ = pnts1.to(device="mps")
    pnts2 = pnts2.unsqueeze(-1).permute(0,2,1).to(device="mps")      # (3 x 1 x num_pnts2)

    msk_in = torch.zeros(idc_f.shape[0], mesh.v.shape[0], dtype=torch.bool)
    # msk_in_ = torch.ones(idc_f.shape[0], mesh.v.shape[0], dtype=torch.bool)
    # mesh_f = mesh.f.to(device="cpu")

    blck_size = torch.ceil(torch.tensor(1e6)/pnts1.shape[1]).to(dtype=torch.int)
    i = 0
    end_idx = 0
    start_time = time.time()
    while end_idx < pnts2.shape[2]:
        start_idx = i * blck_size
        end_idx = min((i+1)*blck_size, pnts2.shape[2])
        i += 1

        if pnts1.dim() == 2:
            pnts1_blck = pnts1_
        else:
            pnts1_blck = pnts1_[:,:,torch.arange(start_idx,end_idx)]
        pnts2_blck = pnts2[:,:,torch.arange(start_idx,end_idx)]

        ratio_intersect = torch.sum(
            (pnts2_blck - mesh.s[:,0, idc_f].view(3, -1, 1)) * mesh.n[:, idc_f].view(3, -1, 1), dim=0
        ) / torch.sum(
            (pnts2_blck - pnts1_blck) * mesh.n[:, idc_f].view(3, -1, 1), dim=0
        )

        pnts_intersect = (pnts1_blck - pnts2_blck) * ratio_intersect.unsqueeze(0) + pnts2_blck

        # msk_v = torch.zeros(mesh.v.shape[0], dtype=torch.bool, device="mps")
        # msk_v[start_v+torch.arange(start_idx,end_idx)] = True
        # msk_f = torch.all(msk_v[mesh.f], dim=1)
        #
        # tri1 = pnts_intersect[:,:,mesh.f[msk_f,:]-start_idx].permute(0,3,1,2).unsqueeze(2)
        # tri2 = mesh.s[:,:,idc_f].unsqueeze(1).unsqueeze(-1)
        # dir1 = torch.cat(
        #     (tri1[:,1:2,:,:] - tri1[:,0:1,:,:],
        #      tri1[:,2:3,:,:] - tri1[:,1:2,:,:],
        #      tri1[:,0:1,:,:] - tri1[:,2:3,:,:]), dim=1
        # )
        # dir2 = torch.cat(
        #     (tri2[:,:,1:2,:] - tri2[:,:,0:1,:],
        #      tri2[:,:,2:3,:] - tri2[:,:,1:2,:],
        #      tri2[:,:,0:1,:] - tri2[:,:,2:3,:]), dim=2
        # )
        #
        # det_inv = dir1[0] * dir2[1] - dir1[1] * dir2[0]
        #
        # vec_p = tri1 - tri2
        # # vec_p1 = torch.sum(dir1 * vec_p, dim=0)
        # # vec_p2 = torch.sum(dir2 * vec_p, dim=0)
        # #
        # # d11 = torch.sum(dir1 * dir1, dim=0)
        # # d12 = torch.sum(dir1 * dir2, dim=0)
        # # d21 = d12
        # # d22 = torch.sum(dir2 * dir2, dim=0)
        # #
        # # det_inv = d11 * d22 - d12 * d21
        # #
        # # k0 = -(d22 * vec_p1 - d21 * vec_p2) / det_inv
        # # k1 =  (d11 * vec_p2 - d12 * vec_p1) / det_inv
        #
        # k0 = -(dir2[1] * vec_p[0] - dir2[0] * vec_p[1]) / det_inv
        # k1 = (dir1[0] * vec_p[1] - dir1[1] * vec_p[0]) / det_inv
        #
        # k0 = k0.to(device="cpu")
        # k1 = k1.to(device="cpu")
        # msk_cross = (torch.sum(
        #     ((k0 > 0) & (k0 <  1)) & ((k1 > 0) & (k1 < 1)), dim=[0,1]
        # ) > 1) | (
        #     (torch.sum( torch.any( (k0<0) & (k1>0) & (k1<1), dim=0), dim=0) == 3) &
        #     (torch.sum( torch.any( (k0>1) & (k1>0) & (k1<1), dim=0), dim=0) == 3) &
        #     (torch.sum( ((k0<0) | (k0>1)) & (k1>0) & (k1<1), dim=[0,1]) == 6)
        # ) | (
        #     (torch.sum( torch.any( (k1<0) & (k0>0) & (k0<1), dim=1), dim=0) == 3) &
        #     (torch.sum( torch.any( (k1>1) & (k0>0) ^ (k0<1), dim=1), dim=0) == 3) &
        #     (torch.sum( ((k0<0) | (k0>1)) & (k0>0) & (k0<1), dim=[0,1]) == 6)
        # )
        #
        # msk_f = msk_f.to(device="cpu")
        # msk_in[:,mesh_f[msk_f,0]] = msk_in[:,mesh_f[msk_f,0]] | msk_cross
        # msk_in[:,mesh_f[msk_f,1]] = msk_in[:,mesh_f[msk_f,1]] | msk_cross
        # msk_in[:,mesh_f[msk_f,2]] = msk_in[:,mesh_f[msk_f,2]] | msk_cross
        #
        # msk_in_[:,start_v+torch.arange(start_idx,end_idx)] = (
        #         (ratio_intersect >= 0).to(device="cpu") &
        #         (ratio_intersect <= 1).to(device="cpu")
        # )

        msk_in_ = (ratio_intersect >= 0) & (ratio_intersect <= 1)

        dir1 = mesh.s[:,0, idc_f] - mesh.s[:, 1, idc_f]
        dir2 = mesh.s[:,2, idc_f] - mesh.s[:, 1, idc_f]

        det_inv = dir1[0] * dir2[1] - dir1[1] * dir2[0]

        r1 = (
                 dir2[1].unsqueeze(-1) * (pnts_intersect[0] - mesh.s[0, 1, idc_f].unsqueeze(-1)) -
                 dir2[0].unsqueeze(-1) * (pnts_intersect[1] - mesh.s[1, 1, idc_f].unsqueeze(-1))
        ) / det_inv.unsqueeze(-1)

        r2 = (
                 dir1[0].unsqueeze(-1) * (pnts_intersect[1] - mesh.s[1, 1, idc_f].unsqueeze(-1)) -
                 dir1[1].unsqueeze(-1) * (pnts_intersect[0] - mesh.s[0, 1, idc_f].unsqueeze(-1))
        ) / det_inv.unsqueeze(-1)

        msk_in_ = msk_in_ & (r1 > 0) & (r2 > 0) & (r1 + r2 < 1)
        msk_in[:,torch.arange(start_idx,end_idx)] = msk_in_.to(device=msk_in.device)

    return msk_in

def pnts_on_surface(mesh, idc_f, rays, pnts_end):

    level = rays.shape[1]
    pnts_end = pnts_end.to(device="mps")  # (3 x 1 x num_rx)
    rays = rays.to(device="mps")                         # (3 x level x num_node x num_rx)

    if level > 1:
        rays_ = rays[:,1:level,:,:].to(device="mps")           # (3 x (level-1) x num_node x num_rx

        rays_flip = 2 * torch.sum(
            (mesh.s[:,0,idc_f].view(3,1,-1,1) - rays_) * mesh.n[:,idc_f].view(3,1,-1,1), dim=0, keepdim=True
        ) * mesh.n[:,idc_f].view(3,1,-1,1) + rays_
        rays[:,0:level-1,:,:] = rays_flip

    ratio_intersect = torch.sum(
        (pnts_end - mesh.s[:,0,idc_f].view(3,-1,1)) * mesh.n[:,idc_f].view(3,-1,1), dim=0
    ) / torch.sum(
        (pnts_end - rays[:,-1,:,:]) * mesh.n[:,idc_f].view(3,-1,1), dim=0
    )

    pnts_intersect = (rays[:,-1,:,:] - pnts_end) * ratio_intersect.unsqueeze(0) + pnts_end
    msk_in = (ratio_intersect >= 0) & (ratio_intersect <= 1)

    dir1 = mesh.s[:, 0, idc_f] - mesh.s[:, 1, idc_f]
    dir2 = mesh.s[:, 2, idc_f] - mesh.s[:, 1, idc_f]

    det_inv = dir1[0] * dir2[1] - dir1[1] * dir2[0]

    r1 = (
                 dir2[1].unsqueeze(-1) * (pnts_intersect[0] - mesh.s[0, 1, idc_f].unsqueeze(-1)) -
                 dir2[0].unsqueeze(-1) * (pnts_intersect[1] - mesh.s[1, 1, idc_f].unsqueeze(-1))
         ) / det_inv.unsqueeze(-1)

    r2 = (
                 dir1[0].unsqueeze(-1) * (pnts_intersect[1] - mesh.s[1, 1, idc_f].unsqueeze(-1)) -
                 dir1[1].unsqueeze(-1) * (pnts_intersect[0] - mesh.s[0, 1, idc_f].unsqueeze(-1))
         ) / det_inv.unsqueeze(-1)

    msk_in = msk_in & (r1 > 0) & (r2 > 0) & (r1 + r2 < 1)
    rays[:,-1,:,:] = pnts_intersect

    return rays.to(device="cpu"), msk_in.to(device="cpu")

def pnts_through_edge(level, edge_n, pnts_edge, pnts1, pnts2):

    num_pnts1 = pnts1.shape[1]
    num_pnts2 = pnts2.shape[1]

    pnts1 = pnts1.view(3,1,-1,1).to(device="mps")
    pnts2 = pnts2.view(3,1,1,-1).to(device="mps")

    dir_edge = pnts_edge[:,1,:,:] - pnts_edge[:,0,:,:]
    dir_edge = dir_edge / torch.linalg.norm(dir_edge, dim=0, keepdim=True)

    dir_edge_out = torch.cross(dir_edge.to(device="cpu"), edge_n.to(device="cpu"), dim=0).to(device="mps")
    dir_edge_out = dir_edge_out / torch.linalg.norm(dir_edge_out, dim=0, keepdim=True)

    dir_edge = dir_edge.unsqueeze(-1).to(device="mps")

    pnts_on_edge = pnts_edge[:,0,:,:].unsqueeze(-1).to(device="mps")

    msk_on = torch.ones(num_pnts1, num_pnts2, dtype=torch.bool, device="cpu")

    blck_size = torch.ceil(torch.tensor(2e7)/num_pnts2).to(dtype=torch.int)
    i = 0
    end_idx = 0

    pnts_end = torch.zeros(3,0)
    while end_idx < num_pnts1:
        start_idx = i * blck_size
        end_idx = min((i+1)*blck_size, num_pnts1)
        i += 1

        pnts1_blck = pnts1[:,:,start_idx:end_idx,:].repeat(1,1,1,num_pnts2)
        dir_edge_blck  = dir_edge[:,:,start_idx:end_idx,:]
        pnts_on_edge_blck = pnts_on_edge[:,:,start_idx:end_idx,:]

        pnts_path = torch.cat(
            (pnts1_blck,
             pnts_on_edge_blck.repeat(1, 1, 1, num_pnts2),
             pnts2.repeat(1,1,pnts1_blck.shape[2],1)), dim=1
        )

        k = torch.zeros(level, pnts1_blck.shape[2], num_pnts2, device="mps")
        msk_on_ = torch.ones(pnts1_blck.shape[2], num_pnts2, dtype=torch.bool)

        for ii in range(level):
            dir_out_orthogonal = torch.cross( (pnts_path[:,ii+2,:,:] - pnts_path[:,ii+1,:,:]).to(device="cpu"),
                                              dir_edge_out[:,ii,start_idx:end_idx].unsqueeze(-1).to(device="cpu"), dim=0 ).to(device="mps")
            msk_on_[:] = msk_on_.unsqueeze(0) & ((
                torch.sum( (pnts_path[:,ii+2,:,:] - pnts_path[:,ii+1,:,:]) * (pnts_path[:,ii,:,:] - pnts_path[:,ii+1,:,:]), dim=0, keepdim=True) < 0
            ) & (
                torch.sum( (pnts_path[:,ii+2,:,:] - pnts_path[:,ii+1,:,:]) * edge_n[:,ii,start_idx:end_idx].unsqueeze(-1), dim=0, keepdim=True) > 0
            ) & (
                torch.sum( (pnts_path[:,ii+2,:,:] - pnts_path[:,ii+1,:,:]) * dir_edge_out[:,ii,start_idx:end_idx].unsqueeze(-1), dim=0, keepdim=True) < 0
            ) & (
                torch.sum( (pnts_path[:,ii,:,:] - pnts_path[:,ii+1,:,:]) * dir_out_orthogonal, dim=0, keepdim=True) > 0
            )).to(device="cpu")

        i1, i2 = torch.where(msk_on_)

        for ii in range(1 if level == 1 else 5):
            for iii in range(level):
                dir1 = pnts_path[:,iii,i1,i2]   - pnts_on_edge_blck[:,iii,i1,0]
                dir2 = pnts_path[:,iii+2,i1,i2] - pnts_on_edge_blck[:,iii,i1,0]

                d1 = torch.linalg.norm(
                    dir1 - torch.sum(dir_edge_blck[:,iii,i1,0] * dir1, dim=0, keepdim=True) * dir_edge_blck[:,iii,i1,0], dim=0, keepdim=True
                )   # (1 x num_pnts1 x num_pnts2)
                d2 = torch.linalg.norm(
                    dir2 - torch.sum(dir_edge_blck[:,iii,i1,0] * dir2, dim=0, keepdim=True) * dir_edge_blck[:,iii,i1,0], dim=0, keepdim=True
                )   # (1 x num_pnts1 x num_pnts2)

                k[iii,i1,i2] = d2 / (d1+d2) * torch.sum(dir_edge_blck[:,iii,i1,0] * dir1, dim=0, keepdim=True) + d1 / (d1+d2) * torch.sum(dir_edge_blck[:,iii,i1,0] * dir2, dim=0, keepdim=True)
                pnts_path[:,iii+1,i1,i2] = pnts_on_edge_blck[:,iii,i1,0] + k[iii,i1,i2] * dir_edge_blck[:,iii,i1,0]

        msk_on_copy = msk_on_[msk_on_.clone()]
        for ii in range(level):
            msk_on_copy[:] = msk_on_copy & ((k[ii,i1,i2] > 0) & (k[ii,i1,i2] < 1)).to(device="cpu")


        msk_on_[msk_on_.clone()] = msk_on_copy

        i1, i2 = torch.where(msk_on_)

        msk_on[start_idx:end_idx,:] = msk_on_.to(device="cpu")
        pnts_end = torch.cat((pnts_end, pnts_path[:, level, i1, i2].to(device="cpu")), dim=1)

    return msk_on, pnts_end

def pnts_on_edge(pnts_edge, pnts1, pnts2):

    level = pnts_edge.shape[2]
    num_pnts1 = pnts1.shape[1]
    num_pnts2 = pnts2.shape[1]

    pnts1 = pnts1.view(3,1,-1,1).repeat(1,1,1,num_pnts2).to(device="mps")       # (3 x 1 x num_pnts1 x num_pnts2)
    pnts2 = pnts2.view(3,1,1,-1).repeat(1,1,num_pnts1,1).to(device="mps")       # (3 x 1 x num_pnts1 x num_pnts2)

    dir_edge = pnts_edge[:,0,:,:] - pnts_edge[:,1,:,:]
    dir_edge = dir_edge / torch.linalg.norm(dir_edge, dim=0, keepdim=True)
    dir_edge = dir_edge.unsqueeze(-1).to(device="mps")                           # (3 x level x num_pnts1 x 1)

    pnts_on_edge = pnts_edge[:,0,:,:].unsqueeze(-1).to(device="mps")

    pnts_path = torch.cat(
        (pnts1,
         pnts_on_edge.repeat(1,1,1,num_pnts2),
         pnts2), dim=1
    )

    k = torch.zeros(1,level,num_pnts1, num_pnts2, device="mps")
    msk_on = torch.ones(num_pnts1, num_pnts2, dtype=torch.bool, device="mps")

    for ii in range(1 if level == 1 else 5):
        for iii in range(level):
            dir1 = pnts_path[:,iii,:,:]   - pnts_on_edge[:,iii,:,:]    # (3 x num_pnts1 x num_pnts2)
            dir2 = pnts_path[:,iii+2,:,:] - pnts_on_edge[:,iii,:,:]    # (3 x num_pnts1 x num_pnts2)

            d1 = torch.linalg.norm(
                dir1 - torch.sum(dir_edge[:,iii,:,:] * dir1, dim=0, keepdim=True) * dir_edge[:,iii,:,:], dim=0, keepdim=True
            )   # (1 x num_pnts1 x num_pnts2)
            d2 = torch.linalg.norm(
                dir2 - torch.sum(dir_edge[:,iii,:,:] * dir2, dim=0, keepdim=True) * dir_edge[:,iii,:,:], dim=0, keepdim=True
            )   # (1 x num_pnts1 x num_pnts2)

            k[:,iii,:,:] = d2 / (d1+d2) * torch.sum(dir_edge[:,iii,:,:] * dir1, dim=0, keepdim=True) + d1 / (d1+d2) * torch.sum(dir_edge[:,iii,:,:] * dir2, dim=0, keepdim=True)

            pnts_path[:,iii+1,:,:] = pnts_on_edge[:,iii,:,:] + k[:,iii,:,:] * dir_edge[:,iii,:,:]

    for ii in range(level):
        msk_on[:] = msk_on.unsqueeze(0) & (k[:,ii,:,:,] > 0) & (k[:,ii,:,:,] < 1) & (
            torch.sum( (pnts_path[:,ii+2,:,:] - pnts_path[:,ii+1,:,:]) * (pnts_path[:,ii,:,:] - pnts_path[:,ii+1,:,:]), dim=0, keepdim=True) < 0
        )

    return pnts_path[:,1:level+1,:,:].to(device="cpu"), msk_on.to(device="cpu")

def line_cross_box(pnts1, pnts2, box_blob):

    # AABB Algorithm
    x_min, y_min, z_min = box_blob[:,0]
    x_max, y_max, z_max = box_blob[:,6]

    t_xmin = (x_min - pnts1[0,:]) / (pnts2[0,:] - pnts1[0,:])
    t_xmax = (x_max - pnts1[0,:]) / (pnts2[0,:] - pnts1[0,:])
    t_ymin = (y_min - pnts1[1,:]) / (pnts2[1,:] - pnts1[1,:])
    t_ymax = (y_max - pnts1[1,:]) / (pnts2[1,:] - pnts1[1,:])
    t_zmin = (z_min - pnts1[2,:]) / (pnts2[2,:] - pnts1[2,:])
    t_zmax = (z_max - pnts1[2,:]) / (pnts2[2,:] - pnts1[2,:])

    tmin = torch.max(
        torch.stack(
            [torch.min(t_xmin, t_xmax),torch.min(t_ymin, t_ymax), torch.min(t_zmin, t_zmax)], dim=0
        ), dim=0
    ).values

    tmax = torch.min(
        torch.stack(
            [torch.max(t_xmin, t_xmax), torch.max(t_ymin, t_ymax), torch.max(t_zmin, t_zmax)], dim=0
        ), dim=0
    ).values

    msk_line = (tmax >= tmin) & (tmax >= 0) & (tmin <= 1)

    return msk_line

def pnts_in_cube(pnts, box_blob):

    x_min, y_min, z_min = box_blob[:,0]
    x_max, y_max, z_max = box_blob[:,6]

    msk_pnt = (
            (pnts[0] > x_min) & (pnts[0] < x_max) &
            (pnts[1] > y_min) & (pnts[1] < y_max) &
            (pnts[2] > z_min) & (pnts[2] < z_max)
    )

    return msk_pnt