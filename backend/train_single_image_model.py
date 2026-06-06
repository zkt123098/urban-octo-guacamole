# train_single_image_model.py
import os
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ------------------ 数据集类 ------------------
class SingleImageDataset(Dataset):
    def __init__(self, image_dir, metadata_root, transform=None):
        """
        image_dir: 存放图像的目录，如 D:/archive/128_x_128/128_x_128
        metadata_root: metadata 根目录，如 D:/archive/metadata/metadata
        """
        self.image_dir = image_dir
        self.metadata_root = metadata_root
        self.transform = transform
        self.samples = []

        # 遍历所有图像文件
        for fname in os.listdir(image_dir):
            if not fname.endswith('.gif'):
                continue
            # 提取台风ID，如 '197901_128_x_128.gif' -> '197901'
            typhoon_id = fname.split('_')[0]
            csv_path = os.path.join(metadata_root, f"{typhoon_id}.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                # 计算最大风速作为标签
                if 'wind' not in df.columns:
                    continue
                max_wind = df['wind'].max()
                if pd.isna(max_wind):
                    continue
                self.samples.append({
                    'image_path': os.path.join(image_dir, fname),
                    'typhoon_id': typhoon_id,
                    'max_wind': float(max_wind)
                })
            except Exception as e:
                print(f"处理 {typhoon_id} 失败: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(sample['image_path'])
        img = cv2.resize(img, (128, 128)) / 255.0          # 归一化
        img_tensor = torch.FloatTensor(img).unsqueeze(0)   # (1, 128, 128)
        label = torch.FloatTensor([sample['max_wind']])
        return img_tensor, label

# ------------------ CNN 回归模型 ------------------
class TyphoonCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        feats = self.features(x)          # (batch, 128, 1, 1)
        feats = feats.view(feats.size(0), -1)
        out = self.regressor(feats)
        return out

# ------------------ 训练函数 ------------------
def train():
    IMAGE_DIR = "D:/archive/128_x_128/128_x_128"
    METADATA_ROOT = "D:/archive/metadata/metadata"
    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    BEST_MODEL_PATH = "model_cache/typhoon_cnn.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    dataset = SingleImageDataset(IMAGE_DIR, METADATA_ROOT)
    print(f"可用样本总数: {len(dataset)}")
    if len(dataset) == 0:
        print("❌ 没有可用样本！请检查路径或CSV文件。")
        return

    # 划分训练/验证集（按样本随机划分，不考虑台风ID泄露，简单处理）
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = TyphoonCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()

        train_avg = train_loss / len(train_loader)
        val_avg = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {train_avg:.4f} | Val Loss: {val_avg:.4f}")

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   🏆 保存最佳模型")

    print(f"🎉 训练完成！模型保存在 {BEST_MODEL_PATH}")

if __name__ == "__main__":
    train()