import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from datasets import AnimeDataset
from model import create_unet
from scheduler import get_scheduler
from utils import show_images

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/data2/detatt/csh/data/anime"
IMAGE_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
MAX_SAMPLES = 5000

# 数据加载
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
dataset = AnimeDataset(DATA_PATH, transform, max_samples=MAX_SAMPLES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 模型与优化器
model = create_unet(IMAGE_SIZE).to(DEVICE)
scheduler = get_scheduler()
optimizer = AdamW(model.parameters(), lr=LR)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# 创建检查点目录
os.makedirs("checkpoints", exist_ok=True)

# 训练循环
losses = []
fixed_noise = torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_losses = []
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}"):
        clean = batch.to(DEVICE)
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (clean.size(0),), device=DEVICE)
        noisy = scheduler.add_noise(clean, noise, timesteps)
        noise_pred = model(noisy, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_losses.append(loss.item())
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(avg_loss)
    lr_scheduler.step(avg_loss)
    print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pth")

# 保存最终模型
torch.save(model.state_dict(), "checkpoints/final.pth")
