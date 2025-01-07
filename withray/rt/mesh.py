"""
Classes and Methods for MESH.
"""
from vedo import *
import torch
import torch.nn.functional as F

class MESH:
    def __init__(self, filename, device, rotation_dir = torch.tensor([1.0, 0.0, 0.0])):
        self.mesh_file = Mesh(filename)

        self.v = torch.tensor(self.mesh_file.vertices)[:, [0,2,1]] * torch.tensor([-1,1,1])
        self.v = self.v.to(dtype=torch.float32)

        rotation_dir = F.normalize(
            torch.tensor([rotation_dir[0], rotation_dir[1], 0.0]), p=2, dim=0
        )
        rotation_mat = torch.cat([
            rotation_dir.unsqueeze(0),
            torch.tensor([-rotation_dir[1], rotation_dir[0], 0.0]).unsqueeze(0),
            torch.tensor([0.0, 0.0, 1.0]).unsqueeze(0)
        ], dim=0).to(dtype=torch.float32)

        self.v = self.v @ rotation_mat.T
        self.v = self.v
        self.f = torch.tensor(self.mesh_file.faces(), dtype=torch.long)
        self.merge_vertices()

        self.v = self.v.to(device=device)
        self.f = self.f.to(device=device)
        self.s = self.v[self.f].permute(2,1,0).to(device=device)
        self.n = self.mesh_normals().to(device=device)



    def mesh_normals(self):

        v1 = self.s[:,0,:]
        v2 = self.s[:,1,:]
        v3 = self.s[:,2,:]

        d1 = v2 - v1
        d1 = F.normalize(d1, p=2, dim=0)

        d2 = v3 - v1
        d2 = F.normalize(d2, p=2, dim=0)

        mesh_n = torch.cross(d1, d2, dim=0)
        mesh_n = F.normalize(mesh_n, p=2, dim=0)

        return mesh_n

    def merge_vertices(self):

        blck_size = 10000
        for i in range(1,6):
            ii_end = (self.v.shape[0]+blck_size-1)//blck_size + 1
            size_v_original = self.v.shape[0]
            for ii in range(1, ii_end):
                idx_v_end = self.v.shape[0]

                if (ii-1) * blck_size < idx_v_end:
                    start_idx = (ii - 1) * blck_size
                    end_idx = min(ii * blck_size, idx_v_end)

                    sub_v = self.v[start_idx:end_idx, :]

                    dist_mat = torch.cdist(sub_v, sub_v, p=2)
                    mask = torch.triu(dist_mat < 1e-2, diagonal=1)
                    row, col = torch.where(mask)

                    col_unique, idc_inv = torch.unique(col, sorted=True, return_inverse=True)
                    idc_perm = torch.arange(idc_inv.size(0), dtype=idc_inv.dtype, device=idc_inv.device)
                    idc_inv, idc_perm = idc_inv.flip([0]), idc_perm.flip([0])
                    idc_unique = idc_inv.new_empty(col_unique.size(0)).scatter_(0, idc_inv, idc_perm)
                    col = (ii-1)*blck_size + col_unique
                    row = (ii-1)*blck_size + row[idc_unique]

                    for iii in range(len(col)):
                        self.f[self.f == col[iii]] = row[iii]

                    ncol = torch.tensor([idx for idx in range(start_idx, end_idx) if idx not in col])
                    for iii in range(len(ncol)):
                        self.f[self.f == ncol[iii]] = start_idx + iii

                    self.f[self.f > min(ii*blck_size, idx_v_end-len(col))-1] -= len(col)

                    mask_to_keep = torch.ones(idx_v_end, dtype=bool, device = col.device)
                    mask_to_keep[col] = False
                    self.v = self.v[mask_to_keep, :]

                    print(f"[{i}][{ii}/{ii_end}] Size v: {self.v.shape[0]}({size_v_original}) \n", end ="")