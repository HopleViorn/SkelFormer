import sys
import os
# 将项目根目录添加到 sys.path
# 这样可以直接从 hy3dshape 导入模块，而无需相对导入
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 从当前目录导入 SkelFormer 模型和新的数据集
from skelformer import SkelFormer
from sdf_dataset import SDFDataset, GeometricSDFDataset

# --- 1. 训练主函数 ---

def train_sdf_model(
    data_dir="data/abc/abc_obj",
    epochs=5,
    batch_size=8, # 增加批量大小以提高效率
    learning_rate=1e-4,
    save_path="skelformer_sdf.pth",
    points_per_sample=4096*8,
    pc_size=81920,
    kl_loss_weight=1e-4, # KL散度损失的权重 (gamma)
    max_files_for_training=None # 限制训练文件数量，用于快速测试
):
    """
    训练SkelFormer模型以预测SDF，遵循Hunyuan3D论文中的策略。

    训练损失 Lr = E[MSE(Ds(x|Zs), SDF(x))] + γ * LKL
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # --- a. 初始化模型、优化器和损失函数 ---
    model = SkelFormer(pc_size = pc_size).to(device)

    # 加载预训练的ShapeVAE权重
    try:
        from hy3dshape.hy3dshape.models.autoencoders import ShapeVAE
        print("正在加载预训练的 ShapeVAE 模型...")
        shapevae = ShapeVAE.from_pretrained(
            'tencent/Hunyuan3D-2.1',
            subfolder='hunyuan3d-vae-v2-1',
            device=device,
            dtype=torch.float32  # 使用float32以匹配SkelFormer
        )
        model.load_weights_from_shapevae(shapevae.state_dict())
        del shapevae
        print("ShapeVAE 权重加载成功。")
    except Exception as e:
        print(f"加载预训练权重失败: {e}")

    optimizer = optim.Adam(model.geo_decoder.parameters(), lr=learning_rate)
    
    # SDF损失使用MSE (L2)
    sdf_loss_fn = nn.L1Loss() 

    # --- b. 准备数据加载器 ---
    print("--- 准备数据集 ---")
    # dataset = SDFDataset(
    #     data_dir=data_dir,
    #     points_per_sample=points_per_sample,
    #     pc_size=pc_size,
    #     max_files=max_files_for_training
    # )
    dataset = GeometricSDFDataset(
        num_samples = 10000,
        points_per_sample=points_per_sample,
        pc_size=pc_size,
    )
    
    if len(dataset) == 0:
        raise ValueError("数据集中没有有效的训练样本。请检查数据路径和文件。")
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print("--- 开始训练 ---")
    model.train() # 设置模型为训练模式
    for epoch in range(epochs):
        total_sdf_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        for i, (point_cloud, query_points, true_sdf) in enumerate(tqdm(dataloader)):
            point_cloud = point_cloud.to(device, non_blocking=True)
            query_points = query_points.to(device, non_blocking=True)
            true_sdf = true_sdf.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # 更高效的梯度清零

            # --- c. 前向传播 (使用 SkelFormer 的 encode 方法) ---
            # encode 方法现在会返回 DiagonalGaussianDistribution 对象
            # 其中包含均值和对数方差，用于计算KL散度
            posterior = model.encode(point_cloud, sample_posterior=True)
            
            # 从后验分布中采样潜在向量
            latents = posterior.sample() 
            
            # 使用潜在向量和查询点解码出SDF值
            predicted_sdf = model.decode_sdf(latents, query_points)

            # --- d. 计算损失 ---
            # 1. SDF重建损失 (MSE)
            sdf_loss = sdf_loss_fn(predicted_sdf, true_sdf)
            
            # 2. KL散度损失 (与标准正态分布)
            kl_loss = posterior.kl(dims=(1, 2)).mean() # 计算KL散度并取平均
            
            # 3. 总损失
            total_loss = sdf_loss

            # --- e. 反向传播和优化 ---
            total_loss.backward()
            # 梯度裁剪，防止梯度爆炸 (可选，但推荐)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()

            total_sdf_loss += sdf_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

            # 每隔一定步数打印一次损失
            if (i + 1) % 50 == 0:
                 avg_sdf_loss = total_sdf_loss / num_batches
                 avg_kl_loss = total_kl_loss / num_batches
                 print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}], "
                       f"Avg SDF Loss: {avg_sdf_loss:.6f}, Avg KL Loss: {avg_kl_loss:.6f}")

        # --- f. Epoch结束后的处理 ---
        avg_sdf_loss = total_sdf_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        print(f"Epoch [{epoch+1}/{epochs}] 完成, "
              f"平均 SDF 损失: {avg_sdf_loss:.6f}, 平均 KL 损失: {avg_kl_loss:.6f}")

        # 定期保存模型
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存到: {save_path}")

    print("--- 训练完成 ---")
    torch.save(model.state_dict(), save_path)
    print(f"最终模型已保存到: {save_path}")


if __name__ == "__main__":
    # 为了快速测试，可以限制训练文件数量
    # train_sdf_model(max_files_for_training=10, epochs=5)
    
    # 进行完整训练
    train_sdf_model()