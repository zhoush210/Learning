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

class CVAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(CVAE, self).__init__()
        
        # 编码器
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 64 * 3 * 3)
        )
        
        # 潜在空间
        self.fc_mu = nn.Linear(64 * 3 * 3 * 2, latent_dim)  # *2是因为要拼接图像和条件信息
        self.fc_logvar = nn.Linear(64 * 3 * 3 * 2, latent_dim)
        
        # 解码器
        self.decoder_fc = nn.Linear(latent_dim + num_classes, 64 * 3 * 3)  # 加入条件信息
        
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
    
    def encoder(self, x, c):
        """编码器：计算潜在空间的均值和对数方差"""
        # 处理图像
        x = self.encoder_conv(x)
        
        # 处理条件信息
        c = self.condition_encoder(c)
        c = c.view(-1, 64, 3, 3)
        c = c.flatten(1)
        
        # 拼接图像和条件信息
        combined = torch.cat([x, c], dim=1)
        
        # 计算均值和方差
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decoder(self, z, c):
        """解码器：从潜在向量和条件重建图像"""
        # 拼接潜在向量和条件信息
        combined = torch.cat([z, c], dim=1)
        z = self.decoder_fc(combined)
        x_recon = self.decoder_conv(z)
        return x_recon
    
    def forward(self, x, c):
        """前向传播"""
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar

def one_hot_encode(labels, num_classes=10):
    """将标签转换为one-hot编码"""
    return F.one_hot(labels, num_classes).float()

def cvae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """CVAE损失函数"""
    # 重建损失
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

def train(model, train_loader, num_epochs=10, beta=1.0):
    """训练CVAE模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            c = one_hot_encode(labels).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, c)
            
            # 计算损失
            loss, recon_loss, kl_loss = cvae_loss_function(recon_batch, data, mu, logvar, beta)
            
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

def visualize_results(model, test_loader, num_images=5):
    """可视化CVAE的结果"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        # 获取一些测试图像
        data, labels = next(iter(test_loader))
        data = data[:num_images].to(device)
        labels = labels[:num_images].to(device)
        c = one_hot_encode(labels).to(device)
        
        # 重建图像
        recon_batch, _, _ = model(data, c)
        
        # 从随机潜在向量生成图像（使用相同的条件）
        z = torch.randn(num_images, 128).to(device)
        gen_imgs = model.decoder(z, c)
        
        # 显示原始图像、重建图像和生成图像
        plt.figure(figsize=(15, 9))
        
        # 原始图像
        for i in range(num_images):
            plt.subplot(3, num_images, i + 1)
            plt.imshow(data[i].cpu().squeeze(), cmap='gray')
            plt.title(f'Label: {labels[i].item()}')
            plt.axis('off')
        
        # 重建图像
        for i in range(num_images):
            plt.subplot(3, num_images, i + 1 + num_images)
            plt.imshow(recon_batch[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
                
        # 生成图像
        for i in range(num_images):
            plt.subplot(3, num_images, i + 1 + 2*num_images)
            plt.imshow(gen_imgs[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('cvae_results.png')
        plt.close()

def main():
    # 创建模型
    model = CVAE()
    
    # 加载数据
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
    
    # 训练模型
    model = train(model, train_loader, num_epochs=10, beta=1.0)
    
    # 保存模型
    torch.save(model.state_dict(), 'cvae.pth')
    
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