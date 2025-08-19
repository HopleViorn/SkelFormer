import torch
from torch.utils.data import Dataset
import numpy as np
import trimesh
import os
import glob

class SDFDataset(Dataset):
    """
    一个用于SDF模型训练的数据集。
    它会加载一个目录中所有的.obj文件，并为每个文件动态生成训练样本。
    """
    def __init__(self, data_dir, points_per_sample=16384, pc_size=81920, max_files=None, sample_on_surface_ratio=0.5):
        """
        初始化数据集。

        Args:
            data_dir (str): 包含.obj文件的目录路径。
            points_per_sample (int): 用于计算SDF损失的查询点总数。
            pc_size (int): 输入到编码器的点云大小。
            max_files (int, optional): 要加载的最大文件数，用于快速测试。默认为None，加载所有文件。
            sample_on_surface_ratio (float): 在表面上采样的查询点所占的比例。
        """
        super().__init__()
        self.points_per_sample = points_per_sample
        self.pc_size = pc_size
        self.sample_on_surface_ratio = sample_on_surface_ratio
        
        print(f"正在从 '{data_dir}' 搜索.obj文件...")
        self.mesh_files = glob.glob(os.path.join(data_dir, '**', '*.obj'), recursive=True)
            
        if max_files is not None:
            self.mesh_files = self.mesh_files[:max_files]
            
        if not self.mesh_files:
            raise FileNotFoundError(f"在目录 '{data_dir}' 中未找到任何.obj文件。")
            
        print(f"找到了 {len(self.mesh_files)} 个.obj文件。")

    def __len__(self):
        return len(self.mesh_files)

    def __getitem__(self, idx):
        mesh_path = self.mesh_files[idx]
        
        try:
            mesh = trimesh.load(mesh_path, force='mesh', skip_materials=True, process=False)
            
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
                 raise ValueError("加载的不是有效的Trimesh对象或网格为空。")

            # 归一化网格
            center = mesh.bounding_box.centroid
            scale = mesh.bounding_box.extents.max()
            if scale > 1e-8:
                mesh.apply_translation(-center)
                mesh.apply_scale(1.0 / scale)
            
            # --- a. 准备输入点云 ---
            points_surface, face_indices = trimesh.sample.sample_surface(mesh, self.pc_size)
            normals_surface = mesh.face_normals[face_indices]
            labels_surface = np.zeros((self.pc_size, 1), dtype=np.float32)
            
            point_cloud_data = np.concatenate([points_surface, normals_surface, labels_surface], axis=1)
            point_cloud_tensor = torch.from_numpy(point_cloud_data).float()

            # --- b. 准备用于计算SDF损失的查询点和真值 ---
            num_surface_samples = int(self.points_per_sample * self.sample_on_surface_ratio)
            num_space_samples = self.points_per_sample - num_surface_samples

            # 在表面附近采样点
            surface_points, _ = trimesh.sample.sample_surface(mesh, num_surface_samples)
            surface_points += np.random.normal(scale=0.005, size=surface_points.shape)
            
            # 在空间中均匀采样点
            space_points = np.random.uniform(-1.0, 1.0, size=(num_space_samples, 3))
            
            query_points = np.concatenate([surface_points, space_points], axis=0)
            
            proximity_query = trimesh.proximity.ProximityQuery(mesh)
            sdf_values = proximity_query.signed_distance(query_points)

            query_points_tensor = torch.from_numpy(query_points).float()
            sdf_values_tensor = torch.from_numpy(sdf_values).float().unsqueeze(-1)

            return point_cloud_tensor, query_points_tensor, sdf_values_tensor

        except Exception as e:
            print(f"处理文件 '{mesh_path}' 时出错: {e}. 跳过此文件。")
            return self.__getitem__((idx + 1) % len(self))


class GeometricSDFDataset(Dataset):
    """
    一个用于SDF模型训练的几何体数据集。
    它会预先生成带随机旋转的几何体，以提高训练效率和数据多样性。
    """
    def __init__(self, num_samples=1000, points_per_sample=16384, pc_size=81920, sample_on_surface_ratio=0.5):
        """
        初始化几何体数据集。

        Args:
            num_samples (int): 数据集中样本的总数。
            points_per_sample (int): 用于计算SDF损失的查询点总数。
            pc_size (int): 输入到编码器的点云大小。
            sample_on_surface_ratio (float): 在表面上采样的查询点所占的比例。
        """
        super().__init__()
        self.num_samples = num_samples
        self.points_per_sample = points_per_sample
        self.pc_size = pc_size
        self.sample_on_surface_ratio = sample_on_surface_ratio
        self.meshes_data = []
        
        self.geometries = ['box', 'cylinder', 'pyramid']
        print(f"正在预生成 {num_samples} 个带随机旋转的几何体样本...")
        for _ in range(self.num_samples):
            self.meshes_data.append(self._generate_geometry())
        print("几何体预生成完成。")

    def __len__(self):
        return len(self.meshes_data)

    # --- 解析SDF函数 ---
    def sdf_box(self, p, size):
        q = np.abs(p) - size / 2
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)

    def sdf_cylinder(self, p, r, h):
        d = np.abs(np.sqrt(p[..., 0]**2 + p[..., 1]**2)) - r
        d = np.maximum(d, np.abs(p[..., 2]) - h / 2)
        return d

    def sdf_pyramid(self, p, b, h):
        # 近似SDF
        p_copy = p.copy()
        p_copy[:, 1] -= h
        m2 = h*h + b*b
        p_copy[:, 0] = np.abs(p_copy[:, 0])
        p_copy[:, 2] = np.abs(p_copy[:, 2])
        
        mask = p_copy[:, 2] > p_copy[:, 0]
        p_copy[mask, 0], p_copy[mask, 2] = p_copy[mask, 2], p_copy[mask, 0]
        
        p_copy[:, 0] -= b

        q = np.stack([
            p_copy[:, 2],
            p_copy[:, 1]*b + p_copy[:, 0]*h,
            p_copy[:, 1]*h - p_copy[:, 0]*b
        ], axis=-1)
        
        s = np.maximum(0, -q[:, 2]/m2)
        q[:, 0] -= s*b
        q[:, 1] -= s*h
        
        d = np.sqrt(q[:, 0]**2 + q[:, 1]**2) * np.sign(q[:, 1])
        return d

    def _generate_box(self):
        size = np.random.uniform(0.2, 1.0, 3)
        box = trimesh.creation.box(extents=size)
        return box, 'box', {'size': size}

    def _generate_cylinder(self):
        radius = np.random.uniform(0.2, 0.8)
        height = np.random.uniform(0.3, 1.5)
        cylinder = trimesh.creation.cylinder(radius=radius, height=height)
        return cylinder, 'cylinder', {'radius': radius, 'height': height}

    def _generate_pyramid(self):
        base_size = np.random.uniform(0.3, 1.0, 2)
        height = np.random.uniform(0.3, 1.5)
        base_vertices = np.array([
            [base_size[0]/2, base_size[1]/2, 0], [base_size[0]/2, -base_size[1]/2, 0],
            [-base_size[0]/2, -base_size[1]/2, 0], [-base_size[0]/2, base_size[1]/2, 0],
            [0, 0, height]
        ])
        faces = np.array([
            [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
            [0, 3, 2], [0, 2, 1]
        ])
        pyramid = trimesh.Trimesh(vertices=base_vertices, faces=faces)
        return pyramid, 'pyramid', {'base_size': base_size[0], 'height': height} # Use single base size for simplicity

    def _generate_geometry(self):
        geom_type = np.random.choice(self.geometries)
        
        if geom_type == 'box':
            mesh, _, params = self._generate_box()
        elif geom_type == 'cylinder':
            mesh, _, params = self._generate_cylinder()
        else: # pyramid
            mesh, _, params = self._generate_pyramid()
        
        local_params = params.copy()

        rotation_matrix = trimesh.transformations.random_rotation_matrix()
        mesh.apply_transform(rotation_matrix)
        
        # 返回 mesh, 类型, 局部参数, 和旋转矩阵
        return mesh, geom_type, local_params, rotation_matrix

    def __getitem__(self, idx):
        try:
            mesh, geom_type, local_params, rotation_matrix = self.meshes_data[idx]
            
            # --- Perform normalization ---
            center = mesh.bounding_box.centroid
            scale = mesh.bounding_box.extents.max()
            
            # Create a normalized copy for point cloud sampling
            mesh_normalized = mesh.copy()
            if scale > 1e-8:
                mesh_normalized.apply_translation(-center)
                mesh_normalized.apply_scale(1.0 / scale)

            # --- a. 准备输入点云 ---
            points_surface, face_indices = trimesh.sample.sample_surface(mesh_normalized, self.pc_size)
            normals_surface = mesh_normalized.face_normals[face_indices]
            labels_surface = np.zeros((self.pc_size, 1), dtype=np.float32)
            point_cloud_data = np.concatenate([points_surface, normals_surface, labels_surface], axis=1)
            point_cloud_tensor = torch.from_numpy(point_cloud_data).float()

            # --- b. 准备查询点和计算SDF ---
            num_surface_samples = int(self.points_per_sample * self.sample_on_surface_ratio)
            num_space_samples = self.points_per_sample - num_surface_samples
            surface_points, _ = trimesh.sample.sample_surface(mesh_normalized, num_surface_samples)
            surface_points += np.random.normal(scale=0.005, size=surface_points.shape)
            space_points = np.random.uniform(-1.0, 1.0, size=(num_space_samples, 3))
            query_points = np.concatenate([surface_points, space_points], axis=0)

            # --- 使用解析方法计算SDF ---
            # 1. 将查询点从归一化空间转换回原始旋转后的空间
            query_points_rotated = query_points.copy()
            if scale > 1e-8:
                query_points_rotated = query_points_rotated * scale + center

            # 2. 将点应用逆旋转，转换到物体的局部坐标系
            inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
            query_points_homogeneous = np.hstack([query_points_rotated, np.ones((query_points.shape[0], 1))])
            local_query_points = (inverse_rotation_matrix @ query_points_homogeneous.T).T[:, :3]

            # 3. 在局部坐标系中调用相应的解析SDF函数
            if geom_type == 'box':
                sdf_values = self.sdf_box(local_query_points, local_params['size'])
            elif geom_type == 'cylinder':
                sdf_values = self.sdf_cylinder(local_query_points, local_params['radius'], local_params['height'])
            else: # pyramid
                sdf_values = self.sdf_pyramid(local_query_points, local_params['base_size'], local_params['height'])
            
            # 4. SDF值是距离，需要乘以缩放因子以匹配归一化后的空间
            if scale > 1e-8:
                sdf_values /= scale

            query_points_tensor = torch.from_numpy(query_points).float()
            sdf_values_tensor = torch.from_numpy(sdf_values).float().unsqueeze(-1)

            return point_cloud_tensor, query_points_tensor, sdf_values_tensor

        except Exception as e:
            print(f"处理几何体 {idx} 时出错: {e}. 重新尝试。")
            return self.__getitem__((idx + 1) % len(self))


def load_single_geometric_sample(pc_size=81920, points_per_sample=16384):
    """
    加载一个由GeometricSDFDataset生成的几何体样本，用于测试。

    Args:
        pc_size (int): 输入点云的大小。
        points_per_sample (int): 用于SDF评估的查询点数量。

    Returns:
        tuple: 包含 (point_cloud_tensor, query_points_tensor, sdf_values_tensor) 的元组。
    """
    print("正在生成一个用于测试的几何体样本...")
    # 创建一个临时的几何体数据集实例，只包含一个样本
    dataset = GeometricSDFDataset(num_samples=1, pc_size=pc_size, points_per_sample=points_per_sample)
    
    # 获取第一个（也是唯一一个）样本
    point_cloud, query_points, sdf_values = dataset[0]
    
    print("几何体样本生成完毕。")
    return point_cloud, query_points, sdf_values
