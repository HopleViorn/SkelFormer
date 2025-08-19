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
    grid_resolution=64,
    mc_level=0.0
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
        grid_min, grid_max = -0.5, 0.5
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

    # 可视化SDF切片
    slice_index = grid_resolution // 2
    sdf_slice = sdf_grid[:, slice_index, :]
    
    # 将SDF值映射到颜色 (红 -> 负, 蓝 -> 正)
    colors = np.zeros((grid_resolution, grid_resolution, 3), dtype=np.uint8)
    sdf_normalized = (sdf_slice - sdf_slice.min()) / (sdf_slice.max() - sdf_slice.min() + 1e-9)
    colors[..., 0] = (1 - sdf_normalized) * 255 # 红色通道
    colors[..., 2] = sdf_normalized * 255       # 蓝色通道
    
    slice_coords = np.array(np.meshgrid(
        np.linspace(grid_min, grid_max, grid_resolution),
        np.linspace(grid_min, grid_max, grid_resolution)
    )).T.reshape(-1, 2)
    
    slice_points = np.insert(slice_coords, 1, grid_min + slice_index * (grid_max - grid_min) / grid_resolution, axis=1)

    v.add_points("SDF Slice", slice_points, colors=colors.reshape(-1, 3), point_size=30)

    viz_path = "sdf_visualization"
    v.save(viz_path)
    print(f"可视化结果已保存到: {viz_path}")
    
    # 保存重建的网格
    if sdf_mesh:
        output_mesh_path = "reconstructed_sdf_mesh.obj"
        sdf_mesh.export(output_mesh_path)
        print(f"重建的网格已保存到: {output_mesh_path}")


if __name__ == "__main__":
    visualize_sdf_from_geometric_sample()