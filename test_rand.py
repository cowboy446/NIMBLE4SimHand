import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bn = 100  # 每次处理的批次大小
delta = 0.1
pose_param = torch.zeros(bn, 30, device=device)
# pose_param[0] = torch.rand(30, device=device) * 2 * 1 - 1  # 第一个样本随机生成

for i in range(1, bn):
    # 生成随机变化，确保正负平衡
    change = torch.randn(30, device=device) * 2 * delta - delta
    positive_mask = torch.rand(30, device=device) > 0.5
    change[positive_mask] = torch.abs(change[positive_mask])  # 保证部分变化为正数
    change[~positive_mask] = -torch.abs(change[~positive_mask])  # 保证部分变化为负数
    
    # 更新 pose_param
    pose_param[i] = pose_param[i-1] + change
    
    # 限制范围
    pose_param[i] = torch.clamp(pose_param[i], -1, 1)
    print(f"pose_param[{i}][0]: {pose_param[i][0].cpu().numpy()}")
# 统计正负数个数
positive_count = (pose_param[:, 0] > 0).sum().item()
print(f"正数个数: {positive_count}, 负数个数个数: {bn - positive_count}")
# 画图, 绘制pose_param的第一个参数变化
import matplotlib.pyplot as plt
plt.plot(pose_param[:, 0].cpu().numpy())
plt.title("Pose Parameter Variation")
plt.xlabel("Sample Index")
plt.ylabel("Pose Param[0]")
plt.grid()
plt.show()