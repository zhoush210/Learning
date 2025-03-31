import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义自编码器模型
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(AutoEncoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, latent_dim)  # 28x28 -> 14x14 -> 7x7 -> 3x3
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 3 * 3),
            nn.Unflatten(1, (64, 3, 3)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Upsample(size=(28, 28), mode='bilinear', align_corners=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output

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

# 训练函数
def train(model, train_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, data)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
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
        reconstructed = model(data)
        
        # 显示原始图像和重建图像
        plt.figure(figsize=(12, 4))
        for i in range(num_images):
            # 原始图像
            plt.subplot(2, num_images, i + 1)
            plt.imshow(data[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Original')
            
            # 重建图像
            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Reconstructed')
        
        plt.tight_layout()
        plt.savefig('reconstruction_results.png')
        plt.close()

def main():
    # 创建模型
    model = AutoEncoder()
    
    # 加载数据
    train_loader = load_data()
    
    # 训练模型
    model = train(model, train_loader)
    
    # 保存模型
    torch.save(model.state_dict(), 'autoencoder.pth')
    
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