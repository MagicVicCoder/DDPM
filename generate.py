import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusers import DDPMPipeline

from model import create_unet
from scheduler import get_scheduler
from utils import show_images

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 64

# 加载模型和调度器
model = create_unet(IMAGE_SIZE).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/final.pth", map_location=DEVICE))
scheduler = get_scheduler()
pipe = DDPMPipeline(unet=model, scheduler=scheduler)

# 生成并保存
os.makedirs("final_generated", exist_ok=True)
for i in range(3):
    imgs = pipe(batch_size=8).images
    for j, img in enumerate(imgs):
        img.save(f"final_generated/batch{i+1}_{j+1}.png")
    # 可视化示例
    grid = show_images(torch.stack([transforms.ToTensor()(img) for img in imgs]))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
