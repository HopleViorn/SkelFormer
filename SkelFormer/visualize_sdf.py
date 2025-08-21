import sys
import os
import torch
import trimesh
import numpy as np
import pyviz3d.visualizer as viz
from skimage.measure import marching_cubes

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from skelformer import SkelFormer
from sdf_dataset import load_single_geometric_sample

def visualize_sdf_from_geometric_sample(
    model_path="skelformer_sdf.pth",
    pc_size=81920,
    grid_resolution=128,
    mc_level=0.0,
    num_slices_x=1,
    num_slices_y=1,
    num_slices_z=1
):
    """
    从一个程序化生成的几何体样本加载模型，预测SDF，并可视化结果。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # SkelFormer 使用 float32
    print(f"使用设备: {device}")

    # --- 1. 加载 SkelFormer 模型 ---
    print(f"正在加载 SkelFormer 模型从: {model_path}")
    model = SkelFormer(pc_size=pc_size).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("SkelFormer 模型加载成功。")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # --- 2. 加载一个程序化生成的几何体样本 ---
    # 这个函数返回的已经是处理好的、带法线和标签的(N, 7)形状的张量
    point_cloud_tensor, _, _ = load_single_geometric_sample(pc_size=pc_size)
    point_cloud_tensor = point_cloud_tensor.unsqueeze(0).to(device=device, dtype=dtype)
    
    # 从张量中提取坐标用于可视化
    points = point_cloud_tensor.squeeze(0)[:, :3].cpu().numpy()
    
    print(f"几何体样本加载完成，张量形状: {point_cloud_tensor.shape}")

    # --- 3. 使用模型进行推理 ---
    with torch.no_grad():
        print("正在编码点云到潜在空间...")
        # 根据 skelformer.py, 当 sample_posterior=True 时, 返回一个分布对象
        posterior = model.encode(point_cloud_tensor, sample_posterior=True)
        # 从分布中采样潜在向量
        latents = posterior.sample()
        print(f"编码完成，潜在向量形状: {latents.shape}")

        # --- 4. 生成查询点网格以可视化SDF ---
        print(f"正在生成 {grid_resolution}^3 的查询点网格...")
        grid_min, grid_max = -0.51, 0.51
        x = torch.linspace(grid_min, grid_max, grid_resolution)
        y = torch.linspace(grid_min, grid_max, grid_resolution)
        z = torch.linspace(grid_min, grid_max, grid_resolution)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        query_points = torch.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], dim=1).to(device)

        # --- 5. 优化SDF预测以减少显存占用 ---
        # 首先，对 latents 进行一次性的预处理（通过 post_kl 和 transformer）
        print("正在预处理潜在向量...")
        processed_latents = model.post_kl(latents)
        processed_latents = model.transformer(processed_latents)
        print("潜在向量预处理完成。")

        # 然后，分块将查询点和预处理后的 latents 送入 geo_decoder
        print("正在分块预测SDF值...")
        sdf_values = []
        chunk_size = 4096 * 16 # 可以根据显存大小调整
        for i in range(0, query_points.shape[0], chunk_size):
            chunk_queries = query_points[i:i+chunk_size]
            
            # 将 chunk_queries 增加一个 batch 维度以匹配 geo_decoder 的期望输入
            chunk_queries = chunk_queries.unsqueeze(0)
            
            # geo_decoder 内部会处理 latents 和 queries 之间的 cross-attention
            predicted_sdf_chunk = model.geo_decoder(queries=chunk_queries, latents=processed_latents)
            
            # 移除 batch 维度并移动到 CPU
            sdf_values.append(predicted_sdf_chunk.squeeze(0).cpu())
        
        sdf_grid = torch.cat(sdf_values, dim=0).reshape(grid_resolution, grid_resolution, grid_resolution).numpy()
        print("SDF值预测完成。")
        print(f"预测的SDF值范围: [{sdf_grid.min():.4f}, {sdf_grid.max():.4f}]")
        # 检查是否有负值
        num_negative = np.sum(sdf_grid < 0)
        num_positive = np.sum(sdf_grid > 0)
        print(f"负值数量: {num_negative}, 正值数量: {num_positive}")

    # --- 6. 使用 Marching Cubes 从SDF场提取网格 ---
    print(f"正在使用 Marching Cubes 提取网格 (level={mc_level})...")
    try:
        vertices, faces, normals, _ = marching_cubes(
            sdf_grid,
            level=mc_level,
            spacing=((grid_max - grid_min) / grid_resolution,) * 3
        )
        # 调整顶点位置，因为marching_cubes的输出是基于网格索引的
        vertices += grid_min
        sdf_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        print("网格提取成功。")
    except Exception as e:
        print(f"Marching Cubes 失败: {e}")
        sdf_mesh = None

    # --- 7. 使用 Pyviz3d 可视化 ---
    print("正在使用 Pyviz3d 生成可视化文件...")
    v = viz.Visualizer()

    pc_d = points[::10]
    
    # 可视化原始点云 (蓝色)
    point_colors = np.tile([0, 0, 255], (pc_d.shape[0], 1))
    v.add_points("Input Point Cloud", pc_d, colors=point_colors, point_size=20)
    
    # 可视化从SDF提取的网格 (绿色)
    # if sdf_mesh and sdf_mesh.vertices.shape[0] > 0:
    #     v.add_mesh("Reconstructed SDF Mesh", sdf_mesh.vertices, sdf_mesh.faces, color=np.array([0, 255, 0]))

    # --- 8. 可视化SDF切片 ---
    grid_min, grid_max = -0.5, 0.5
    
    def get_slice_colors(sdf_slice):
        """根据SDF值计算颜色，负值为红色，正值为蓝色。"""
        colors = np.zeros((sdf_slice.shape[0], sdf_slice.shape[1], 3), dtype=np.uint8)
        # 找到SDF绝对值的最大值用于归一化
        max_abs_val = np.max(np.abs(sdf_slice)) + 1e-9

        print(sdf_slice[32-10:32+10,32-10:32+10])

        
        # 负值 (内部): 从红色 (SDF值最小) 到 白色 (SDF值接近0)
        neg_mask = sdf_slice < 0
        if np.any(neg_mask):
            neg_vals = -sdf_slice[neg_mask] / max_abs_val # 归一化到 [0, 1]
            colors[neg_mask, 0] = 255  # R=255
            colors[neg_mask, 1] = 255 * (1 - neg_vals) # G
            colors[neg_mask, 2] = 255 * (1 - neg_vals) # B

        # 正值 (外部): 从蓝色 (SDF值最大) 到 白色 (SDF值接近0)
        pos_mask = sdf_slice >= 0
        if np.any(pos_mask):
            pos_vals = sdf_slice[pos_mask] / max_abs_val # 归一化到 [0, 1]
            colors[pos_mask, 2] = 255 # B=255
            colors[pos_mask, 0] = 255 * (1 - pos_vals) # R
            colors[pos_mask, 1] = 255 * (1 - pos_vals) # G
        return colors.reshape(-1, 3)

    # 可视化 YZ 平面上的切片 (沿 X 轴)
    for i in range(num_slices_x):
        slice_index = int((i + 1) * (grid_resolution / (num_slices_x + 1)))
        if slice_index >= grid_resolution: continue
        sdf_slice = sdf_grid[slice_index, :, :]
        colors = get_slice_colors(sdf_slice)
        coords = np.array(np.meshgrid(
            np.linspace(grid_min, grid_max, grid_resolution),
            np.linspace(grid_min, grid_max, grid_resolution)
        )).T.reshape(-1, 2)
        slice_points = np.insert(coords, 0, grid_min + slice_index * (grid_max - grid_min) / grid_resolution, axis=1)
        v.add_points(f"SDF Slice X-{i+1}", slice_points, colors=colors, point_size=30)

    # 可视化 XZ 平面上的切片 (沿 Y 轴)
    for i in range(num_slices_y):
        slice_index = int((i + 1) * (grid_resolution / (num_slices_y + 1)))
        if slice_index >= grid_resolution: continue
        sdf_slice = sdf_grid[:, slice_index, :]
        colors = get_slice_colors(sdf_slice)
        coords = np.array(np.meshgrid(
            np.linspace(grid_min, grid_max, grid_resolution),
            np.linspace(grid_min, grid_max, grid_resolution)
        )).T.reshape(-1, 2)
        slice_points = np.insert(coords, 1, grid_min + slice_index * (grid_max - grid_min) / grid_resolution, axis=1)
        v.add_points(f"SDF Slice Y-{i+1}", slice_points, colors=colors, point_size=30)

    # 可视化 XY 平面上的切片 (沿 Z 轴)
    for i in range(num_slices_z):
        slice_index = int((i + 1) * (grid_resolution / (num_slices_z + 1)))
        if slice_index >= grid_resolution: continue
        sdf_slice = sdf_grid[:, :, slice_index]
        colors = get_slice_colors(sdf_slice)
        coords = np.array(np.meshgrid(
            np.linspace(grid_min, grid_max, grid_resolution),
            np.linspace(grid_min, grid_max, grid_resolution)
        )).T.reshape(-1, 2)
        slice_points = np.insert(coords, 2, grid_min + slice_index * (grid_max - grid_min) / grid_resolution, axis=1)
        v.add_points(f"SDF Slice Z-{i+1}", slice_points, colors=colors, point_size=30)

    viz_path = "sdf_visualization"
    v.save(viz_path)
    print(f"可视化结果已保存到: {viz_path}")
    
    # 保存重建的网格
    if sdf_mesh:
        output_mesh_path = "reconstructed_sdf_mesh.obj"
        sdf_mesh.export(output_mesh_path)
        print(f"重建的网格已保存到: {output_mesh_path}")


if __name__ == "__main__":
    # 示例：在每个轴上可视化2个切片
    visualize_sdf_from_geometric_sample(num_slices_x=3, num_slices_y=3, num_slices_z=3)