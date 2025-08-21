import os
import sys
import argparse
import trimesh
import numpy as np
from tqdm import tqdm

# --- 重要：将 GEM3D 的代码目录添加到 Python 路径中 ---
# 这使得我们可以导入它的工具函数
GEM3D_CODE_PATH = 'RelatedWork/GEM3D_paper_code'
if GEM3D_CODE_PATH not in sys.path:
    sys.path.append(GEM3D_CODE_PATH)

try:
    from utils.skeleton import get_full_min_sdf_skeleton
except ImportError:
    print(f"错误：无法从 '{GEM3D_CODE_PATH}' 导入 'get_full_min_sdf_skeleton'。")
    print("请确保 GEM3D_CODE_PATH 变量指向正确的路径。")
    sys.exit(1)

def process_mesh(obj_file_path, output_root_dir, num_faces, skel_iter):
    """
    处理单个OBJ文件：简化、计算骨架并保存。
    """
    file_name = os.path.basename(obj_file_path)
    model_id = os.path.splitext(file_name)[0]
    
    # 1. 创建输出目录
    simple_mesh_dir = os.path.join(output_root_dir, 'watertight_simple')
    skel_dir = os.path.join(output_root_dir, 'skeletons_min_sdf_iter_50')
    os.makedirs(simple_mesh_dir, exist_ok=True)
    os.makedirs(skel_dir, exist_ok=True)

    simple_mesh_path = os.path.join(simple_mesh_dir, f"{model_id}.off")
    skel_path = os.path.join(skel_dir, f"{model_id}_skel_graph.npz")

    if os.path.exists(skel_path):
        print(f"跳过：骨架文件已存在 '{skel_path}'")
        return

    try:
        # 2. 加载网格
        mesh = trimesh.load(obj_file_path, force='mesh')
        
        # 尝试修复非水密网格
        if not mesh.is_watertight:
            print(f"警告：网格 '{file_name}' 不是水密的，尝试自动修复...")
            mesh.fill_holes()
            
            # 再次检查
            if not mesh.is_watertight:
                print(f"错误：自动修复失败。网格 '{file_name}' 仍然不是水密的，已跳过。")
                return
            else:
                print(f"信息：网格 '{file_name}' 已成功修复为水密。")

        print(f"处理中: {file_name}...")

        # 3. 简化网格
        # 注意：如果原始面数小于目标面数，将不会进行简化
        if len(mesh.faces) > num_faces:
            mesh = mesh.simplify_quadratic_decimation(num_faces)
        
        mesh.export(simple_mesh_path)
        print(f"  -> 已保存简化网格到: {simple_mesh_path}")

        # 4. 计算骨架
        # GEM3D的函数需要一个细分后的网格来进行SDF计算
        subdivided_mesh = mesh.subdivide_to_size(max_edge=0.05, max_iter=50)
        
        skel_points, edges = get_full_min_sdf_skeleton(
            subdivided_mesh, 
            num_iter=skel_iter,
            init_type='random' # 'random' 对于非流形或复杂的网格通常更鲁棒
        )
        
        # 5. 保存骨架
        pts_save = skel_points.numpy().astype(np.float32)
        edges_save = edges.astype(np.int32)
        np.savez(skel_path, vertices=pts_save, edges=edges_save)
        print(f"  -> 已保存骨架到: {skel_path}")

    except Exception as e:
        print(f"处理 '{file_name}' 时发生错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="为ABC数据集中的OBJ文件预处理并生成骨架。")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='包含.obj文件的输入目录 (例如: data/abc/abc_obj)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='保存预处理结果的输出目录 (例如: data/abc/preprocessed)')
    parser.add_argument('--num_faces', type=int, default=100000,
                        help='网格简化后的目标面数。')
    parser.add_argument('--skel_iter', type=int, default=50,
                        help='骨架计算的迭代次数。')
    args = parser.parse_args()

    # 获取所有 .obj 文件
    obj_files = [f for f in os.listdir(args.input_dir) if f.endswith('.obj')]
    if not obj_files:
        print(f"错误：在 '{args.input_dir}' 中没有找到 .obj 文件。")
        return

    print(f"找到 {len(obj_files)} 个 .obj 文件。")

    for file_name in tqdm(obj_files, desc="整体进度"):
        obj_path = os.path.join(args.input_dir, file_name)
        process_mesh(obj_path, args.output_dir, args.num_faces, args.skel_iter)

if __name__ == '__main__':
    main()