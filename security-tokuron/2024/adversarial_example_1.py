import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 1. シンプルなニューラルネットワークの定義
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        x = self.fc(x)
        return x

# 2. データの準備
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. モデルと最適化関数の初期化
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. モデルの訓練（簡単な1エポックのみ）
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    break  # 1バッチのみで学習（例示のため）

# 5. 敵対的サンプルを生成する関数（FGSM）
def generate_adversarial_example(model, image, label, epsilon=0.1):
    image.requires_grad = True  # 勾配計算を有効化
    output = model(image)
    loss = criterion(output, label)
    model.zero_grad()
    loss.backward()  # 勾配の計算

    # FGSMによるノイズの追加
    perturbation = epsilon * image.grad.sign()
    adversarial_image = image + perturbation
    adversarial_image = torch.clamp(adversarial_image, 0, 1)  # [0, 1]範囲にクリップ

    return adversarial_image

# 6. 敵対的サンプルの生成と表示
import matplotlib.pyplot as plt

images, labels = next(iter(train_loader))
image, label = images[0:1], labels[0:1]  # 1枚の画像を使用

# 敵対的サンプル生成
adversarial_image = generate_adversarial_example(model, image, label)

# 元の画像と敵対的サンプルの比較表示
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image.squeeze().detach().numpy(), cmap='gray')
ax[0].set_title("Original Image")
ax[1].imshow(adversarial_image.squeeze().detach().numpy(), cmap='gray')
ax[1].set_title("Adversarial Image")
plt.show()
