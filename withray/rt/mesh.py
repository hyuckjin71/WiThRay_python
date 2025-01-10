"""
Classes and Methods for MESH.
"""
from vedo import *
import os
import pickle
import torch
import torch.nn.functional as F

class MESH:
    def __init__(self, file_name, device, rotation_dir = torch.tensor([1.0, 0.0, 0.0])):
        dir_path = "map_data/"
        file_path = os.path.join(dir_path, f"{file_name}.pkl")

        rotation_dir = F.normalize(
            torch.tensor([rotation_dir[0], rotation_dir[1], 0.0]), p=2, dim=0
        )
        rotation_mat = torch.cat([
            rotation_dir.unsqueeze(0),
            torch.tensor([-rotation_dir[1], rotation_dir[0], 0.0]).unsqueeze(0),
            torch.tensor([0.0, 0.0, 1.0]).unsqueeze(0)
        ], dim=0).to(dtype=torch.float32)

        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                loaded_obj = pickle.load(file)
                self.__dict__.update(loaded_obj)
                print(f"Loaded mesh from {file_path}")

                file_path = os.path.join(dir_path, f"{file_name}.obj")
                self.mesh_file = Mesh(file_path)
        else:
            file_path = os.path.join(dir_path, f"{file_name}.pkl")

            file_path = os.path.join(dir_path, f"{file_name}.obj")
            self.mesh_file = Mesh(file_path)

            self.v = torch.tensor(self.mesh_file.vertices)[:, [0,2,1]] * torch.tensor([-1,1,1])
            self.v = self.v.to(dtype=torch.float32)

            self.v = self.v @ rotation_mat.T
            self.v = self.v
            self.f = torch.tensor(self.mesh_file.faces(), dtype=torch.long)
            self.merge_vertices()

            self.s = self.v[self.f].permute(2,1,0)
            self.n = self.mesh_normals()

            with open(file_path, 'wb') as file:
                pickle.dump(self.to_serializable_dict(), file)
                print(f"Saved mesh to {file_path}")

        self.mesh_file.vertices = self.mesh_file.vertices[:, [0,2,1]]
        self.mesh_file.vertices[:,0] *= -1
        self.mesh_file.vertices = self.mesh_file.vertices @ rotation_mat.T.numpy()

        self.mesh_to_device(device=device)

        file_path = os.path.join(dir_path, f"{file_name}.png")
        self.mesh_file.texture(file_path, scale=1)

    def to_serializable_dict(self):
        serializable_dict = {}
        for key, value in self.__dict__.items():
            try:
                # 직렬화 가능한지 테스트
                pickle.dumps(value)
                # GPU 텐서를 CPU로 이동
                if isinstance(value, torch.Tensor):
                    serializable_dict[key] = value.cpu()
                else:
                    serializable_dict[key] = value
            except TypeError:
                # 직렬화 불가능한 객체는 제외
                print(f"Excluding non-serializable object: {key} of type {type(value)}")
                continue
        return serializable_dict

    def mesh_to_device(self, device):

        self.v = self.v.to(device=device)
        self.f = self.f.to(device=device)
        self.s = self.s.to(device=device)
        self.n = self.n.to(device=device)

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

                    len_decrease = len(col)
                    for iii in list(range(len(col)))[::-1]:
                        self.f[self.f == col[iii]] = row[iii]

                    ncol = torch.tensor([idx for idx in range(start_idx, end_idx) if idx not in col])
                    for iii in range(len(ncol)):
                        if torch.any(self.f == ncol[iii]):
                            self.f[self.f == ncol[iii]] = start_idx + iii
                        else:
                            col = torch.cat([col, ncol[iii]])

                    self.f[self.f > min(ii*blck_size, idx_v_end-len_decrease)-1] -= len_decrease

                    mask_to_keep = torch.ones(idx_v_end, dtype=bool, device = col.device)
                    mask_to_keep[col] = False
                    self.v = self.v[mask_to_keep, :]

                    print(f"[{i}][{ii}/{ii_end}] Size v: {self.v.shape[0]}({size_v_original}) \n", end ="")