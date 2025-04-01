import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义变分自编码器模型
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        
        # 编码器 - 输出均值和对数方差
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 潜在空间的维度是latent_dim
        self.fc_mu = nn.Linear(64 * 3 * 3, latent_dim)
        self.fc_logvar = nn.Linear(64 * 3 * 3, latent_dim)
        
        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, 64 * 3 * 3)
        
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (64, 3, 3)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Upsample(size=(28, 28), mode='bilinear', align_corners=True),
            nn.Sigmoid()
        )
    
    def encoder(self, x):
        """编码器：计算潜在空间的均值和对数方差"""
        x = self.encoder_conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧：从概率分布中采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decoder(self, z):
        """解码器：从潜在向量重建图像"""
        z = self.decoder_fc(z)
        x_recon = self.decoder_conv(z)
        return x_recon
    
    def forward(self, x):
        """前向传播"""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# 数据加载和预处理
def load_data():
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
        batch_size=128, 
        shuffle=True
    )
    
    return train_loader

# VAE损失函数
def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE损失函数 = 重建损失 + beta * KL散度
    """
    # 重建损失
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

# 训练函数
def train(model, train_loader, num_epochs=10, beta=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            # 计算损失
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, data, mu, logvar, beta)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item() / len(data):.4f}, '
                      f'Recon: {recon_loss.item() / len(data):.4f}, '
                      f'KL: {kl_loss.item() / len(data):.4f}')
        
        avg_loss = train_loss / len(train_loader.dataset)
        avg_recon = recon_loss_total / len(train_loader.dataset)
        avg_kl = kl_loss_total / len(train_loader.dataset)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Average Loss: {avg_loss:.4f}, '
              f'Recon: {avg_recon:.4f}, '
              f'KL: {avg_kl:.4f}')
    
    return model

# 可视化结果
def visualize_results(model, test_loader, num_images=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        # 获取一些测试图像
        data, _ = next(iter(test_loader))
        data = data[:num_images].to(device)
        
        # 重建图像
        recon_batch, _, _ = model(data)
        
        # 从随机潜在向量生成图像
        z = torch.randn(num_images, 128).to(device)
        gen_imgs = model.decoder(z)
        
        # 显示原始图像、重建图像和生成图像
        plt.figure(figsize=(15, 9))
        
        # 原始图像
        for i in range(num_images):
            plt.subplot(3, num_images, i + 1)
            plt.imshow(data[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Original')
        
        # 重建图像
        for i in range(num_images):
            plt.subplot(3, num_images, i + 1 + num_images)
            plt.imshow(recon_batch[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Reconstructed')
                
        # 生成图像
        for i in range(num_images):
            plt.subplot(3, num_images, i + 1 + 2*num_images)
            plt.imshow(gen_imgs[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Generated')
        
        plt.tight_layout()
        plt.savefig('vae_results.png')
        plt.close()

def main():
    # 创建模型
    model = VAE()
    
    # 加载数据
    train_loader = load_data()
    
    # 训练模型
    model = train(model, train_loader, num_epochs=10, beta=10)
    
    # 保存模型
    torch.save(model.state_dict(), 'vae.pth')
    
    # 加载测试数据并可视化结果
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    visualize_results(model, test_loader)

if __name__ == '__main__':
    main() 