"""
Ray tracer utilities
"""

import torch
import torch.nn.functional as F

def mesh_normals(mesh_s):

    v1 = mesh_s[:,0,:]
    v2 = mesh_s[:,1,:]
    v3 = mesh_s[:,2,:]

    d1 = v2 - v1
    d1 = F.normalize(d1, p=2, dim=0)

    d2 = v3 - v1
    d2 = F.normalize(d2, p=2, dim=0)

    mesh_n = torch.cross(d1, d2, dim=0)
    mesh_n = F.normalize(mesh_n, p=2, dim=0)

    return mesh_n

def import_mesh(mesh_file, device, rotation_dir = torch.tensor([1.0, 0.0, 0.0])):

    mesh_v = torch.tensor(mesh_file.vertices)[:, [0, 2, 1]] * torch.tensor([-1, 1, 1])
    mesh_v = mesh_v.to(dtype=torch.float32)

    rotation_dir = F.normalize(
        torch.tensor([rotation_dir[0], rotation_dir[1], 0.0])
        , p=2, dim=0)
    rotation_mat = torch.cat([
        rotation_dir.unsqueeze(0),
        torch.tensor([-rotation_dir[1], rotation_dir[0], 0.0]).unsqueeze(0),
        torch.tensor([0.0, 0.0, 1]).unsqueeze(0)
    ], dim=0).to(dtype=torch.float32)
    mesh_v = mesh_v @ rotation_mat.T

    mesh_f = torch.tensor(mesh_file.faces(), dtype=torch.long)

    mesh_s = mesh_v[mesh_f].permute(2, 1, 0)

    mesh = {"v":mesh_v.to(device=device),
            "f":mesh_f.to(device=device),
            "s":mesh_s.to(device=device),
            "n":mesh_normals(mesh_s).to(dtype=torch.float32, device=device)}

    return mesh

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

    inv_s = inverse_mat(mesh["s"].unsqueeze(-1) - pnts.permute(0,2,3,1))
    area_s = torch.cross(mesh["s"][:,0,:]-mesh["s"][:,1,:], mesh["s"][:,2,:]-mesh["s"][:,1,:], dim=0)
    area_s = torch.norm(area_s, p=2, dim=0)
    dir_s = pnts - torch.mean(mesh["s"].unsqueeze(-1), dim=1, keepdim=True)
    norm_dir_s = torch.linalg.norm(dir_s, ord=2, dim=0, keepdim=True)
    dir_s = dir_s / norm_dir_s**4 / torch.abs(dir_s[2,:,:,:]).unsqueeze(0)**0.5
    area_s = area_s.unsqueeze(0).unsqueeze(-1) * torch.abs(torch.sum(mesh["n"].unsqueeze(-1).permute(0,2,1).unsqueeze(-1) * dir_s, dim=0))

    _,idx_sorted = torch.sort(torch.mean(area_s, dim=0).squeeze(), descending=True)
    inv_s = inv_s[:,:,idx_sorted,:]

    return inv_s