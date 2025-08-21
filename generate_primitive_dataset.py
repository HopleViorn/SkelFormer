import trimesh
import numpy as np
import os

# --- 配置参数 ---
# 数据集根目录
DATA_DIR = "data/watertight_cad"
# 每种类别生成多少个样本
SAMPLES_PER_CLASS = 1000
# 是否在生成时打印信息
VERBOSE = True

def create_cuboid():
    """生成随机尺寸的长方体"""
    dims = np.random.rand(3) * 2 + 0.5 # 尺寸范围 [0.5, 2.5]
    mesh = trimesh.creation.box(extents=dims)
    return mesh

def create_sphere():
    """生成随机半径的球体"""
    radius = np.random.rand() * 1.0 + 0.5 # 半径范围 [0.5, 1.5]
    # subdivisions 增加可获得更平滑的球面
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    return mesh

def create_cylinder():
    """生成随机尺寸的圆柱体"""
    radius = np.random.rand() * 1.0 + 0.5 # 半径范围 [0.5, 1.5]
    height = np.random.rand() * 2.0 + 0.5 # 高度范围 [0.5, 2.5]
    # sections 增加可获得更平滑的侧面
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
    return mesh

def create_cone():
    """生成随机尺寸的圆锥体"""
    radius = np.random.rand() * 1.0 + 0.5 # 半径范围 [0.5, 1.5]
    height = np.random.rand() * 2.0 + 0.5 # 高度范围 [0.5, 2.5]
    mesh = trimesh.creation.cone(radius=radius, height=height, sections=64)
    return mesh

def create_torus():
    """生成随机尺寸的圆环体"""
    major_radius = np.random.rand() * 1.0 + 0.8 # 主半径
    minor_radius = np.random.rand() * 0.4 + 0.1 # 次半径 (确保 major > minor)
    mesh = trimesh.creation.torus(major_radius=major_radius, minor_radius=minor_radius)
    return mesh

def create_capsule():
    """生成随机尺寸的胶囊体"""
    height = np.random.rand() * 2.0 + 0.5
    radius = np.random.rand() * 1.0 + 0.5
    mesh = trimesh.creation.capsule(height=height, radius=radius)
    return mesh

def create_pyramid():
    """通过创建一个4个截面的圆锥来模拟棱锥"""
    radius = np.random.rand() * 1.0 + 0.5
    height = np.random.rand() * 2.0 + 0.5
    mesh = trimesh.creation.cone(radius=radius, height=height, sections=4)
    return mesh

def create_icosahedron():
    """生成随机大小的正二十面体"""
    radius = np.random.rand() * 1.0 + 0.5
    mesh = trimesh.creation.icosahedron(radius=radius)
    return mesh

def create_tube():
    """生成随机尺寸的管状体"""
    outer_radius = np.random.rand() * 1.0 + 0.8
    inner_radius = np.random.rand() * 0.6 + 0.1
    # 保证外径大于内径
    if inner_radius >= outer_radius:
        inner_radius = outer_radius - 0.2
    height = np.random.rand() * 2.0 + 0.5
    
    # 创建一个外圆柱体
    outer_cylinder = trimesh.creation.cylinder(radius=outer_radius, height=height, sections=64)
    # 创建一个内圆柱体
    inner_cylinder = trimesh.creation.cylinder(radius=inner_radius, height=height, sections=64)
    
    # 使用布尔差集运算得到管状体
    mesh = outer_cylinder.difference(inner_cylinder)
    return mesh

def create_hemisphere():
    """生成随机半径的半球体"""
    radius = np.random.rand() * 1.0 + 0.5
    # 创建一个球体
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    # 使用一个平面进行切割
    plane_normal = [0, 0, 1]
    plane_origin = [0, 0, 0]
    # slice_mesh_plane 会返回一个 watertight 的网格
    mesh = trimesh.intersections.slice_mesh_plane(
        mesh=sphere,
        plane_normal=plane_normal,
        plane_origin=plane_origin,
        cap=True # 保证切割后是封闭的
    )
    return mesh


# 将类别名称和生成函数关联起来
GEOMETRY_GENERATORS = {
    "cuboid": create_cuboid,
    "sphere": create_sphere,
    "cylinder": create_cylinder,
    "cone": create_cone,
    "torus": create_torus,
    "capsule": create_capsule,
    "pyramid": create_pyramid,
    "icosahedron": create_icosahedron,
    "tube": create_tube,
    "hemisphere": create_hemisphere,
}

def main():
    """主函数，用于生成整个数据集"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        if VERBOSE:
            print(f"Created base directory: {DATA_DIR}")

    for class_name, generator_func in GEOMETRY_GENERATORS.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        if VERBOSE:
            print(f"\nGenerating {SAMPLES_PER_CLASS} samples for class: '{class_name}'...")

        for i in range(SAMPLES_PER_CLASS):
            # 生成几何体
            mesh = generator_func()
            
            # 随机旋转和平移，增加数据多样性
            # 随机旋转
            angle = np.random.rand(3) * 2 * np.pi
            rotation_matrix = trimesh.transformations.euler_matrix(angle[0], angle[1], angle[2])
            # 随机平移
            translation_vector = (np.random.rand(3) - 0.5) * 0.1
            transform_matrix = rotation_matrix
            transform_matrix[:3, 3] = translation_vector
            mesh.apply_transform(transform_matrix)

            # 导出为 .obj 文件
            file_name = f"{class_name}_{i:04d}.obj"
            output_path = os.path.join(class_dir, file_name)
            mesh.export(output_path)

            if VERBOSE and (i + 1) % 10 == 0:
                print(f"  ... saved {i + 1}/{SAMPLES_PER_CLASS} samples.")
    
    print(f"\nDataset generation complete. Data saved in '{DATA_DIR}'.")


if __name__ == "__main__":
    main()
