"""
Ray tracer utilities
"""

import torch
import torch.nn.functional as F

def inverse_mat(mat_a):
    """
    Compute the inverse of multiple 3x3 matrices in blocks.

    Args:
        mat_a: Tensor of shape (3, 3, H, W) or (3, 3, N).
               mat_a must be on GPU or CPU (torch.Tensor).

    Returns:
        inv_mat_a: Tensor of shape (3, 3, H, W) or (3, 3, N).
    """
    blck_size = int(1e5)  # Block size
    device = mat_a.device  # Ensure GPU compatibility

    # Total number of matrices
    total_matrices = mat_a.shape[2] if mat_a.dim() == 3 else mat_a.shape[2] * mat_a.shape[3]

    # Output tensor initialization
    inv_mat_a = torch.zeros((3, 3, total_matrices), device=device)

    num_blck = (total_matrices + blck_size - 1) // blck_size  # Number of blocks

    for i in range(num_blck):
        # Index range for the current block
        start_idx = i * blck_size
        end_idx = min((i + 1) * blck_size, total_matrices)

        # Extract block of matrices
        if mat_a.dim() == 4:
            idc = torch.arange(start_idx, end_idx, device=device)
            mat_a_block = mat_a[:, :, idc // mat_a.shape[3], idc % mat_a.shape[3]]
        else:
            mat_a_block = mat_a[:, :, start_idx:end_idx]  # Shape: (3, 3, block_size)

        # Compute determinants
        det = (
                mat_a_block[0, 0] * (mat_a_block[1, 1] * mat_a_block[2, 2] - mat_a_block[1, 2] * mat_a_block[2, 1]) -
                mat_a_block[0, 1] * (mat_a_block[1, 0] * mat_a_block[2, 2] - mat_a_block[1, 2] * mat_a_block[2, 0]) +
                mat_a_block[0, 2] * (mat_a_block[1, 0] * mat_a_block[2, 1] - mat_a_block[1, 1] * mat_a_block[2, 0])
        )

        # Compute the inverse using the adjugate method
        adjugate = torch.stack([
            torch.stack([
                mat_a_block[1, 1] * mat_a_block[2, 2] - mat_a_block[1, 2] * mat_a_block[2, 1],
                mat_a_block[0, 2] * mat_a_block[2, 1] - mat_a_block[0, 1] * mat_a_block[2, 2],
                mat_a_block[0, 1] * mat_a_block[1, 2] - mat_a_block[0, 2] * mat_a_block[1, 1]
            ]),
            torch.stack([
                mat_a_block[1, 2] * mat_a_block[2, 0] - mat_a_block[1, 0] * mat_a_block[2, 2],
                mat_a_block[0, 0] * mat_a_block[2, 2] - mat_a_block[0, 2] * mat_a_block[2, 0],
                mat_a_block[0, 2] * mat_a_block[1, 0] - mat_a_block[0, 0] * mat_a_block[1, 2]
            ]),
            torch.stack([
                mat_a_block[1, 0] * mat_a_block[2, 1] - mat_a_block[1, 1] * mat_a_block[2, 0],
                mat_a_block[0, 1] * mat_a_block[2, 0] - mat_a_block[0, 0] * mat_a_block[2, 1],
                mat_a_block[0, 0] * mat_a_block[1, 1] - mat_a_block[0, 1] * mat_a_block[1, 0]
            ])
        ], dim=0)

        inv_block = adjugate / det.unsqueeze(0).unsqueeze(0)
        inv_mat_a[:, :, start_idx:end_idx] = inv_block

    # Reshape to original dimensions
    if mat_a.dim() == 4:
        inv_mat_a = inv_mat_a.view(3, 3, mat_a.shape[2], mat_a.shape[3])

    return inv_mat_a

def sorted_inv_s(mesh, pnts):

    if pnts.ndim < 3:
        pnts = pnts.unsqueeze(-1)
    if pnts.ndim < 4:
        pnts = pnts.unsqueeze(-1)

    inv_s = inverse_mat(mesh.s.unsqueeze(-1) - pnts.permute(0,2,3,1))
    area_s = torch.cross(mesh.s[:,0,:]-mesh.s[:,1,:], mesh.s[:,2,:]-mesh.s[:,1,:], dim=0)
    area_s = torch.norm(area_s, p=2, dim=0)
    dir_s = pnts - torch.mean(mesh.s.unsqueeze(-1), dim=1, keepdim=True)
    norm_dir_s = torch.linalg.norm(dir_s, ord=2, dim=0, keepdim=True)
    dir_s = dir_s / norm_dir_s**4 / torch.abs(dir_s[2,:,:,:]).unsqueeze(0)**0.5
    area_s = area_s.unsqueeze(0).unsqueeze(-1) * torch.abs(torch.sum(mesh["n"].unsqueeze(-1).permute(0,2,1).unsqueeze(-1) * dir_s, dim=0))

    _,idx_sorted = torch.sort(torch.mean(area_s, dim=0).squeeze(), descending=True)
    inv_s = inv_s[:,:,idx_sorted,:]

    return inv_s

def line_pnt_intersect(pnts1, pnts2, mesh):

    msk_in = torch.zeros(pnts1.shape[0], pnts2.shape[0], dtype=torch.bool)

    box_blob = torch.tensor(
        [mesh.v[:, 0].min(),
         mesh.v[:, 1].min(),
         mesh.v[:, 2].min(),
         mesh.v[:, 0].max() - mesh.v[:, 0].min(),
         mesh.v[:, 1].max() - mesh.v[:, 1].min(),
         mesh.v[:, 2].max() - mesh.v[:, 2].min()]
    )

    msk_in = bvh_algorithm(msk_in, mesh, pnts1, pnts2, torch.zeros(1,3), ~msk_in, 1, box_blob)

    return msk_in

def bvh_algorithm(msk_in, mesh, pnts1, pnts2, msk_box_prev, msk_prev, level, box_blob):

    msk_box = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ])

    for i in range(8):
        msk_box_ = msk_box_prev + msk_box[i,:] / 2**level
        box_blob_ = box_blob[:2].T + box_blob[3:6] * (msk_box_ + msk_box/2**level)

        i1, i2 = torch.where(msk_prev)

        msk_line = msk_prev.clone()
        msk_line[msk_line] = line_cross_box(pnts1[i1,:], pnts2[i2,:], box_blob)

        i1, i2 = torch.where(msk_line)

        msk_f = torch.any(
            pnts_in_cube(mesh.s, box_blob).view(3, -1), dim=0
        )

        if torch.sum(msk_f) * torch.sum(msk_line) < 4e7:
            msk_in[msk_line] = msk_in[msk_line] | torch.squeeze(
                pnts_in_surfaces(mesh.s, msk_f,
                                 pnts1[i1,:].T.view(3,1,1,-1),
                                 pnts2[i2,:].T.view(3,1,1,-1))
            )
        else:
            msk_in = bvh_algorithm(msk_in, mesh, pnts1, pnts2, msk_box_, msk_line, level+1, box_blob)

    return msk_in

def pnts_in_surfaces(mesh, msk_f, pnts1, pnts2):

    ratio_intersect = torch.sum(
        (pnts2 - mesh.s[:,0,msk_f].unsqueeze(1)) * mesh.n[msk_f,:].T.unsqueeze(-1),
    dim=0) / torch.sum(
        (pnts2-pnts1) * mesh.n[msk_f,:].T.unsqueeze(-1),
    dim=0)

    pnts_intersect = (pnts1 - pnts2) * ratio_intersect + pnts2
    msk_in = (ratio_intersect >= 0) & (ratio_intersect <= 1)

    dir1 = mesh.s[:,0,msk_f] - mesh.s[:,1,msk_f]
    dir2 = mesh.s[:,2,msk_f] - mesh.f[:,1,msk_f]

    det_inv = (
        dir1[0].unsqueeze(-1).unsqueeze(-1) * dir2[1].unsqueeze(-1).unsqueeze(-1) -
        dir1[1].unsqueeze(-1).unsqueeze(-1) * dir2[0].unsqueeze(-1).unsqueeze(-1)
    )

    r1 = (
        dir2[1].unsqueeze(-1).unsqueeze(-1) * (pnts_intersect[0] - mesh.s[0, 1, msk_f].unsqueeze(-1).unsqueeze(-1)) -
        dir2[0].unsqueeze(-1).unsqueeze(-1) * (pnts_intersect[1] - mesh.s[1, 1, msk_f].unsqueeze(-1).unsqueeze(-1))
    ) / det_inv

    r2 = (
        dir1[0].unsqueeze(-1).unsqueeze(-1) * (pnts_intersect[1] - mesh.s[1, 1, msk_f].unsqueeze(-1).unsqueeze(-1)) -
        dir1[1].unsqueeze(-1).unsqueeze(-1) * (pnts_intersect[0] - mesh.s[0, 1, msk_f].unsqueeze(-1).unsqueeze(-1))
    ) / det_inv

    msk_in = torch.any(msk_in & (r1 > 0) & (r2 > 0) & (r1+r2 < 1), dim=0)
    return msk_in

def line_cross_box(pnts1, pnts2, box_blob):

    # AABB Algorithm
    x_min, y_min, z_min = box_blob[0,:]
    x_max, y_max, z_max = box_blob[6,:]

    t_xmin = (x_min - pnts1[:,0]) / (pnts2[:,0] - pnts1[:,0])
    t_xmax = (x_max - pnts1[:,0]) / (pnts2[:,0] - pnts1[:,0])
    t_ymin = (y_min - pnts1[:,1]) / (pnts2[:,1] - pnts1[:,1])
    t_ymax = (y_max - pnts1[:,1]) / (pnts2[:,1] - pnts1[:,1])
    t_zmin = (z_min - pnts1[:,2]) / (pnts2[:,2] - pnts1[:,2])
    t_zmax = (z_max - pnts1[:,2]) / (pnts2[:,2] - pnts1[:,2])

    tmin = torch.max(
        torch.stack(
            [torch.min(t_xmin, t_xmax),torch.min(t_ymin, t_ymax), torch.min(t_zmin, t_zmax)], dim=0
        ), dim=0
    )

    tmax = torch.min(
        torch.stack(
            [torch.max(t_xmin, t_xmax), torch.max(t_ymin, t_ymax), torch.max(t_zmin, t_zmax)], dim=0
        ), dim=0
    )

    msk_line = (tmax >= tmin) & (tmax >= 0) & (tmin <= 1)

    return msk_line

def pnts_in_cube(pnts, box_blob):

    x_min, y_min, z_min = box_blob[0, :]
    x_max, y_max, z_max = box_blob[6, :]

    msk_pnt = (
            (pnts[0] > x_min) & (pnts[0] < x_max) &
            (pnts[1] > y_min) & (pnts[1] < y_max) &
            (pnts[2] > z_min) & (pnts[2] < z_max)
    )

    return msk_pnt