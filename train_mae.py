import torch
from torch.utils.data import DataLoader
import yaml
import pickle
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.transformer_encoder import TabularTransformerEncoder
from models.decoder import MAEDecoder
from datasets.geochem_dataset import GeochemMAEDataset
from utils.visualization import plot_loss_curve, plot_feature_reconstruction
from utils.logger import get_logger
from utils.seed import set_seed
from torch.utils.data import Subset

# ======== 读取配置 ========
with open("configs/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)
    SEED = cfg['training'].get('seed', 42)

set_seed(SEED)
os.makedirs(os.path.dirname(cfg['output']['checkpoint_path']), exist_ok=True)
os.makedirs(os.path.dirname(cfg['output']['log_path']), exist_ok=True)
os.makedirs(os.path.dirname(cfg['output']['figure_path']), exist_ok=True)

# ======== 加载预处理数据 ========
with open("preprocessed_geochem.pkl", "rb") as f:
    data = pickle.load(f)

X_scaled = data['X_scaled']
mask_probs = data['correlation_probs']
n_features = X_scaled.shape[1]
feature_names = data['feature_names']

# 选择 side-task 目标元素
side_targets = cfg['training'].get('side_targets', [])
side_indices = [feature_names.index(f) for f in side_targets]

# ======== 构建 Dataset 与 Dataloader ========
dataset = GeochemMAEDataset(
    X_scaled,
    mask_ratio=cfg['data']['mask_ratio'],
    mask_probs=mask_probs,
    mode=cfg['data']['mask_mode'],
    return_side_values=True,
    side_indices=side_indices,
    force_mask_side_features=True  # 👈 强制遮盖 side-task 特征
)

dataloader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)

# ======== 初始化模型 ========
# Transformer Encoder
encoder = TabularTransformerEncoder(
    num_tokens=n_features,
    embed_dim=cfg['model']['hidden_dim'],
    depth=cfg['model']['num_layers'],
    heads=cfg['model']['num_heads'],
    pooling=cfg['model']['pooling']
)

# 若 pooling 为 mean+max，输出为 2x hidden_dim
pooling_out_dim = {
    "mean": cfg['model']['hidden_dim'],
    "cls": cfg['model']['hidden_dim'],
    "max": cfg['model']['hidden_dim'],
    "flatten": cfg['model']['hidden_dim'] * n_features,
    "mean+max": cfg['model']['hidden_dim'] * 2
}[cfg['model']['pooling']]

decoder = MAEDecoder(
    latent_dim=pooling_out_dim,
    output_dim=n_features,
    hidden_dim=cfg['model']['hidden_dim']
)

# Side-task head（用于预测指定元素）
side_head = None
if side_targets:
    side_head = torch.nn.Linear(pooling_out_dim, len(side_targets))

# ======== 优化器 ========
params = list(encoder.parameters()) + list(decoder.parameters())
if side_head:
    params += list(side_head.parameters())

optimizer = torch.optim.Adam(params, lr=float(cfg['training']['learning_rate']))

# ======== 日志器 ========
logger = get_logger(cfg['output']['log_path'])
logger.info("🚀 Start Transformer-based MAE training...")

# ======== 训练循环 ========
loss_fn = torch.nn.MSELoss()
loss_history = []
best_loss = float('inf')
patience = cfg['training'].get('early_stop_patience', 10)
wait = 0
early_stop_path = cfg['output']['checkpoint_path'].replace(".pt", "_best.pt")

for epoch in range(cfg['training']['epochs']):
    encoder.train()
    decoder.train()
    if side_head:
        side_head.train()

    total_loss = 0.0
    for x_masked, x_true, mask, side_vals in dataloader:
        z = encoder(x_masked)
        x_recon = decoder(z)

        # 主任务重建 loss
        loss_main = loss_fn(x_recon[mask], x_true[mask])

        # 辅助任务 loss（如果设置）
        if side_head:
            side_pred = side_head(z)
            loss_side = F.mse_loss(side_pred, side_vals)
            loss = loss_main + cfg['training']['side_loss_weight'] * loss_side
        else:
            loss = loss_main

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    loss_history.append(total_loss)
    if (epoch + 1) % cfg['training']['log_interval'] == 0:
        logger.info(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Loss: {total_loss:.4f}")

    # === 早停判断逻辑 ===
    if total_loss < best_loss - 1e-4:
        best_loss = total_loss
        wait = 0
        # 保存当前最优模型
        torch.save(encoder.state_dict(), early_stop_path)
        logger.info(f"💾 New best model saved at epoch {epoch+1}, loss = {total_loss:.4f}")
    else:
        wait += 1
        if wait >= patience:
            logger.info(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break

# ======== 保存模型 & 可视化 ========
torch.save(encoder.state_dict(), cfg['output']['checkpoint_path'])
logger.info(f"✅ Transformer encoder saved to {cfg['output']['checkpoint_path']}")

plot_loss_curve(loss_history, cfg['output']['figure_path'])
logger.info(f"📉 Loss curve saved to {cfg['output']['figure_path']}")

# ======== 重建特征可视化（如 Au, Hg, As） ========
selected_features = ["Au",  "As","Sb","Bi","Hg","Ba","Co","Cu","Pb","Zn","Ag","W","Mo"]
feature_indices = [feature_names.index(f) for f in selected_features]

encoder.eval()
decoder.eval()
X_tensor = torch.tensor(X_scaled[:100], dtype=torch.float32)
with torch.no_grad():
    z = encoder(X_tensor)
    X_recon = decoder(z).numpy()

os.makedirs("outputs/figures/features/", exist_ok=True)
for i, feature_idx in enumerate(feature_indices):
    true_vals = X_scaled[:100, feature_idx]
    pred_vals = X_recon[:, feature_idx]
    fname = f"outputs/figures/features/recon_{selected_features[i]}.png"
    plot_feature_reconstruction(true_vals, pred_vals, selected_features[i], fname)
    logger.info(f"📊 Saved reconstruction plot for {selected_features[i]} to {fname}")

# ======== 热力图展示原始/掩码/重建样本 ========
encoder.eval()
decoder.eval()

num_samples = 10
sample_indices = np.random.choice(len(dataset), size=num_samples, replace=False)
subset = Subset(dataset, sample_indices)

original, mask_mat, masked_input, recon = [], [], [], []

with torch.no_grad():
    for i in range(num_samples):
        batch = subset[i]
        x_masked, x_true, mask = batch[:3]
        x_true = x_true.numpy()
        mask = mask.numpy()
        x_masked = x_masked.numpy()
        x_recon = decoder(encoder(torch.tensor(x_masked).unsqueeze(0))).squeeze(0).numpy()

        original.append(x_true)
        mask_mat.append(mask.astype(float))
        masked_input.append(x_masked)
        recon.append(x_recon)

# 拼接矩阵
original = np.array(original)
mask_mat = np.array(mask_mat)
masked_input = np.array(masked_input)
recon = np.array(recon)

heatmap_data = np.concatenate([
    original,
    mask_mat,
    masked_input,
    recon
], axis=0)

# 生成行标签
row_labels = (
    [f"ori-{i}" for i in range(num_samples)] +
    [f"mask-{i}" for i in range(num_samples)] +
    [f"inpt-{i}" for i in range(num_samples)] +
    [f"reco-{i}" for i in range(num_samples)]
)

# 绘图
plt.figure(figsize=(13, 6))
sns.heatmap(heatmap_data, cmap="coolwarm", cbar=True, xticklabels=feature_names)
plt.yticks(np.arange(0.5, 4 * num_samples + 0.5, 1), row_labels, rotation=0, fontsize=6)
plt.title("Original / Mask / Masked Input / Reconstruction")
plt.tight_layout()
os.makedirs("outputs/figures/heatmap/", exist_ok=True)
plt.savefig("outputs/figures/heatmap/improved_heatmap.png")
plt.close()

logger.info("📊 Saved improved heatmap with original, mask, masked input and reconstruction.")