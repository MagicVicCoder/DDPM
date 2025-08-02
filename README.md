
markdown

# DDPM Anime Generator

基于 Denoising Diffusion Probabilistic Models (DDPM) 的动漫图像生成项目。

## 项目结构
DDPM-Anime/
├── datasets.py
├── model.py
├── scheduler.py
├── utils.py
├── train.py
├── generate.py
├── requirements.txt
└── README.md

perl
复制
编辑

## 安装依赖
```bash
pip install -r requirements.txt
数据准备
将你的动漫图像放入 data/anime 目录，支持 .jpg、.png 格式。

训练
bash
复制
编辑
python train.py --data_path data/anime --epochs 30
生成
bash
复制
编辑
python generate.py
参数说明
IMAGE_SIZE: 输入图像大小，默认 64×64

BATCH_SIZE: 每批训练样本数量，默认 16

EPOCHS: 训练轮数，默认 30

LR: 学习率，默认 1e-4

许可证
MIT License
