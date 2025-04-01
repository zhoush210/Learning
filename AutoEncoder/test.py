import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from ae import AutoEncoder
from vae import VAE
from cvae import CVAE
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

def load_auto_encoder_model(model_path='autoencoder.pth', latent_dim=128):
    """加载预训练的自编码器模型"""
    model = AutoEncoder(latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_vae_model(model_path='vae.pth', latent_dim=128):
    """加载预训练的VAE模型"""
    model = VAE(latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_cvae_model(model_path='cvae.pth', latent_dim=128):
    """加载预训练的CVAE模型"""
    model = CVAE(latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def one_hot_encode(labels, num_classes=10):
    """将标签转换为one-hot编码"""
    return F.one_hot(labels, num_classes).float()

def generate_from_random(model, num_samples=5, latent_dim=128, labels=None):
    """从随机潜在向量生成图像"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 生成随机潜在向量
    z = torch.randn(num_samples, latent_dim).to(device)
    
    # 处理条件信息（如果是CVAE）
    if isinstance(model, CVAE):
        if labels is None:
            labels = torch.randint(0, 10, (num_samples,)).to(device)
        c = one_hot_encode(labels).to(device)
        generated = model.decoder(z, c)
    else:
        generated = model.decoder(z)
    
    # 显示生成的图像
    plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        # plt.imshow(generated[i].cpu().squeeze(), cmap='gray')
        plt.imshow(generated[i].detach().cpu().squeeze().numpy(), cmap='gray')

        if isinstance(model, CVAE):
            plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('random_generated.png')
    plt.show()
    
    return generated

def generate_from_custom_latent(model, latent_vector, latent_dim=128, label=None):
    """从自定义潜在向量生成图像"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 确保潜在向量维度正确
    if isinstance(latent_vector, list):
        latent_vector = torch.tensor([latent_vector], dtype=torch.float32)
    elif isinstance(latent_vector, np.ndarray):
        latent_vector = torch.from_numpy(latent_vector).float()
        
    if latent_vector.dim() == 1:
        latent_vector = latent_vector.unsqueeze(0)
    
    latent_vector = latent_vector.to(device)
    
    # 处理条件信息（如果是CVAE）
    if isinstance(model, CVAE):
        if label is None:
            label = torch.tensor([0], device=device)
        c = one_hot_encode(label).to(device)
        generated = model.decoder(latent_vector, c)
    else:
        generated = model.decoder(latent_vector)
    
    # 显示生成的图像
    plt.figure(figsize=(6, 6))
    # plt.imshow(generated[0].cpu().squeeze(), cmap='gray')
    plt.imshow(generated[0].detach().cpu().squeeze().numpy(), cmap='gray')

    if isinstance(model, CVAE):
        plt.title(f'Label: {label.item()}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('custom_generated.png')
    plt.show()
    
    return generated

def interpolate_latent(model, num_steps=10, latent_dim=128, label=None):
    """在潜在空间中进行插值，生成中间图像"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 生成两个随机潜在向量
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)
    
    # 计算插值系数
    alphas = torch.linspace(0, 1, num_steps).to(device)
    
    # 处理条件信息（如果是CVAE）
    if isinstance(model, CVAE):
        if label is None:
            label = torch.tensor([0], device=device)
        c = one_hot_encode(label).to(device)
    
    # 进行插值并生成中间图像
    interpolated_images = []
    for alpha in alphas:
        z_interp = alpha * z1 + (1 - alpha) * z2
        with torch.no_grad():
            if isinstance(model, CVAE):
                generated = model.decoder(z_interp, c)
            else:
                generated = model.decoder(z_interp)
            interpolated_images.append(generated.cpu().squeeze())
    
    # 显示插值结果
    plt.figure(figsize=(15, 3))
    for i, img in enumerate(interpolated_images):
        plt.subplot(1, num_steps, i + 1)
        plt.imshow(img, cmap='gray')
        if isinstance(model, CVAE) and i == 0:
            plt.title(f'Label: {label.item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('interpolated.png')
    plt.show()
    
    return interpolated_images

def generate_from_mean_latent(model, latent_dim=128):
    """使用训练集的平均潜在向量生成图像"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 加载训练数据
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        transform=transform, 
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256, 
        shuffle=False
    )
    
    # 计算所有潜在向量的均值
    latent_vectors = []
    with torch.no_grad():
        for data, labels in train_loader:
            data = data.to(device)
            if isinstance(model, CVAE):
                c = one_hot_encode(labels).to(device)
                latent, _ = model.encoder(data, c)
            elif isinstance(model, VAE):
                latent, _ = model.encoder(data)
            else:
                latent = model.encoder(data)
            latent_vectors.append(latent)
    
    # 计算均值
    latent_mean = torch.cat(latent_vectors).mean(dim=0, keepdim=True)
    print(f"潜在向量均值形状: {latent_mean.shape}")
    
    # 使用平均潜在向量生成图像
    with torch.no_grad():
        if isinstance(model, CVAE):
            # 使用最常见的类别（0）作为条件
            c = one_hot_encode(torch.tensor([0], device=device))
            generated = model.decoder(latent_mean, c)
        else:
            generated = model.decoder(latent_mean)
    
    # 显示生成的图像
    plt.figure(figsize=(6, 6))
    plt.imshow(generated[0].cpu().squeeze(), cmap='gray')
    if isinstance(model, CVAE):
        plt.title("mean_latent_generated (Label: 0)")
    else:
        plt.title("mean_latent_generated")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('mean_latent_generated.png')
    plt.show()
    
    return generated, latent_mean

if __name__ == "__main__":
    # 加载模型
    model = load_auto_encoder_model()
    # model = load_vae_model()
    # model = load_cvae_model()

    print("生成随机图像...")
    # 为CVAE指定标签
    labels = torch.tensor([0, 1, 2, 3, 4]) if isinstance(model, CVAE) else None
    generate_from_random(model, num_samples=5, labels=labels)

    print("在潜在空间中插值...")
    interpolate_latent(model, num_steps=5, label=torch.tensor([0]) if isinstance(model, CVAE) else None)
    
    print("使用训练集的平均潜在向量生成图像...")
    generated_img, mean_latent = generate_from_mean_latent(model)
    print(mean_latent.shape)

    print("从自定义潜在向量生成图像...")
    custom_vector = torch.zeros(128)  # 创建一个全零的潜在向量
    print(custom_vector.shape)
    for i in range(128):
        custom_vector[i] = 1.0  # 修改特定位置的值
    generate_from_custom_latent(model, custom_vector, label=torch.tensor([0]) if isinstance(model, CVAE) else None)
    
    print("所有图像已保存到当前目录!") 