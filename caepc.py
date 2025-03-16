# 概念自编码器训练（L1+L2+正交）
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset  # 添加Dataset导入
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd  # 添加pandas导入
import os
from sklearn.model_selection import StratifiedKFold
import json
import random


# =======================
# 设置随机种子以确保结果可重复
# =======================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 当输入数据大小变化不大时关闭该选项，以减少CUDA实现的非确定性


set_seed(37)  # 设置一个固定的随机种子


class MultiModalDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        # 修改这里的特征提取，因为新数据的特征全部来自basefeature的输出
        self.features = data.iloc[:, 2:].values  # 从第3列开始都是特征
        self.labels = data.iloc[:, 1].values  # 第2列是标签(DX_bl)
        self.patient_ids = data.iloc[:, 0].values  # 第1列是病人ID(PTID)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])
        return features, label


# =======================
# 2. 定义概念自编码器类
# =======================
# 修改ConceptAutoEncoder类
class ConceptAutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_layers):
        super(ConceptAutoEncoder, self).__init__()

        # 编码器
        encoder_layers = []
        in_features = input_dim
        for out_features in encoding_layers:
            encoder_layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_features = out_features
        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器
        decoder_layers = []
        for out_features in reversed(encoding_layers[:-1]):
            decoder_layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_features = out_features
        decoder_layers.append(nn.Linear(in_features, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def analyze_activation_patterns(self, concept_features, device="cpu"):
        concept_features = concept_features.to(device)
        self.encoder.to(device)

        with torch.no_grad():
            activations = self.encoder(concept_features)
            activations = activations.cpu().numpy()

        if activations.shape[1] > 0:
            print("First latent dimension range:", activations[:, 0].min(), activations[:, 0].max())
        if activations.shape[1] > 1:
            print("Second latent dimension range:", activations[:, 1].min(), activations[:, 1].max())

        if activations.shape[0] <= 1:
            print("Warning: Only one sample in the activation patterns.")
            return

        plt.figure(figsize=(10, 6))
        sns.heatmap(activations, cmap="viridis", cbar=True)
        plt.xlabel("Latent Dimensions")
        plt.ylabel("Samples")
        plt.title("Activation Patterns in Latent Space")
        plt.show()

        if activations.shape[1] >= 2:
            activations_to_plot = activations[:, :2]
            noisy_activations = activations_to_plot + np.random.normal(0, 0.01, activations_to_plot.shape)

            plt.figure(figsize=(8, 6))
            plt.scatter(noisy_activations[:, 0], noisy_activations[:, 1], alpha=0.6, c="blue")
            plt.xlabel("Latent Dimension 1")
            plt.ylabel("Latent Dimension 2")
            plt.title("Scatter Plot of First Two Latent Dimensions")
            plt.show()


# =======================
# 3. 定义概念自编码器训练类
# =======================
class ConceptAutoEncoderTrainer:
    def __init__(self, autoencoder, model_dir="./saved_models"):
        self.autoencoder = autoencoder
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train(self, train_loader, val_loader, num_epochs=50, optimizer_choice='adam', learning_rate=1e-3,
              regularization_type=None, regularization_strength_l1=0.0, regularization_strength_l2=0.0,
              orthogonal_constraint=False, lambda_orthogonality=1e-5, device="cpu",
              best_model_filename="best_concept_autoencoder.pth"):
        self.autoencoder.to(device)

        # 选择优化器
        if optimizer_choice.lower() == 'adam':
            optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate,
                                   weight_decay=regularization_strength_l2)
        elif optimizer_choice.lower() == 'sgd':
            optimizer = optim.SGD(self.autoencoder.parameters(), lr=learning_rate,
                                  weight_decay=regularization_strength_l2)
        elif optimizer_choice.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.autoencoder.parameters(), lr=learning_rate,
                                      weight_decay=regularization_strength_l2)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_choice}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )

        criterion = nn.SmoothL1Loss()
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            self.autoencoder.train()
            total_loss = 0
            for batch in train_loader:
                inputs = batch[0].to(device)
                optimizer.zero_grad()
                encoded, outputs = self.autoencoder(inputs)
                loss = criterion(outputs, inputs)

                if regularization_type in ['L1', 'L1_L2'] and regularization_strength_l1 > 0.0:
                    l1_loss = 0.0
                    for param in self.autoencoder.parameters():
                        l1_loss += torch.sum(torch.abs(param))
                    loss += regularization_strength_l1 * l1_loss

                if orthogonal_constraint:
                    W = self.autoencoder.encoder[0].weight
                    orthogonality_loss = torch.norm(torch.mm(W, W.t()) - torch.eye(W.size(0)).to(device))
                    loss += lambda_orthogonality * orthogonality_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            val_loss = self.evaluate(val_loader, criterion, regularization_type, regularization_strength_l1,
                                     orthogonal_constraint, lambda_orthogonality, device)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.autoencoder.state_dict(), os.path.join(self.model_dir, best_model_filename))

        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss over Epochs")
        plt.legend()
        plt.show()

    def evaluate(self, data_loader, criterion, regularization_type=None, regularization_strength_l1=0.0,
                 orthogonal_constraint=False, lambda_orthogonality=1e-5, device="cpu"):
        self.autoencoder.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0].to(device)
                _, outputs = self.autoencoder(inputs)
                loss = criterion(outputs, inputs)

                # 添加L1正则化
                if regularization_type in ['L1', 'L1_L2'] and regularization_strength_l1 > 0.0:
                    l1_loss = 0.0
                    for param in self.autoencoder.parameters():
                        l1_loss += torch.sum(torch.abs(param))
                    loss += regularization_strength_l1 * l1_loss

                # 添加正交约束
                if orthogonal_constraint:
                    W = self.autoencoder.encoder[0].weight
                    orthogonality_loss = torch.norm(torch.mm(W, W.t()) - torch.eye(W.size(0)).to(device))
                    loss += lambda_orthogonality * orthogonality_loss

                total_loss += loss.item()
        return total_loss / len(data_loader)


# =======================
# 5. 主程序
# =======================
def main():
    save_dir = "./saved_models"  # 模型保存目录
    # 修改encoding_layers以适应新的特征维度
    encoding_layers = [256, 64, 16]  # 根据basefeature提取的特征维度调整第一层
    # 或者
    # encoding_layers = [96, 32, 16]# 编码器的层，最后一个元素是概念空间的维度
    batch_size = 32
    num_epochs_ce = 100  # 概念自编码器训练的epoch数
    concept_num_folds = 5  # 概念自编码器的交叉验证折数
    optimizer_choice = 'adam'
    learning_rate_ce = 5e-4
    regularization_type = 'L2'
    regularization_strength_l1 = 1e-5
    regularization_strength_l2 = 1e-5
    orthogonal_constraint = False
    lambda_orthogonality = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    dataset = MultiModalDataset('fold_4_features.csv')
    labels = dataset.labels

    # 使用分层K折交叉验证
    skf = StratifiedKFold(n_splits=concept_num_folds, shuffle=True, random_state=37)

    best_concept_val_loss = float('inf')
    best_concept_model_path = os.path.join(save_dir, "best_concept_autoencoder_final.pth")

    # 开始交叉验证训练
    for fold_ce, (train_idx, val_idx) in enumerate(skf.split(dataset.features, labels), 1):
        print(f"\n开始概念自编码器的第 {fold_ce}/{concept_num_folds} 折交叉验证")

        # 划分训练集和验证集
        train_features = dataset.features[train_idx]
        val_features = dataset.features[val_idx]
        train_features = torch.FloatTensor(train_features)
        val_features = torch.FloatTensor(val_features)

        # 创建 DataLoaders
        train_dataset = TensorDataset(train_features)
        val_dataset = TensorDataset(val_features)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 初始化概念自编码器
        input_dim = train_features.shape[1]  # 所有特征的总维度
        concept_autoencoder = ConceptAutoEncoder(input_dim, encoding_layers).to(device)
        trainer = ConceptAutoEncoderTrainer(concept_autoencoder, model_dir=save_dir)

        # 定义每折的最优模型文件名
        best_model_filename = f"best_concept_autoencoder_fold{fold_ce}.pth"

        # 训练概念自编码器
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=num_epochs_ce,
            optimizer_choice=optimizer_choice,
            learning_rate=learning_rate_ce,
            regularization_type=regularization_type,
            regularization_strength_l1=regularization_strength_l1,
            regularization_strength_l2=regularization_strength_l2,
            orthogonal_constraint=orthogonal_constraint,
            lambda_orthogonality=lambda_orthogonality,
            device=device,
            best_model_filename=best_model_filename
        )

        # 评估当前折的模型
        current_val_loss = trainer.evaluate(
            val_loader,
            nn.SmoothL1Loss(),
            regularization_type=regularization_type,
            regularization_strength_l1=regularization_strength_l1,
            orthogonal_constraint=orthogonal_constraint,
            lambda_orthogonality=lambda_orthogonality,
            device=device
        )
        print(f"第 {fold_ce} 折的验证损失: {current_val_loss:.4f}")

        # 保存最佳模型
        if current_val_loss < best_concept_val_loss:
            best_concept_val_loss = current_val_loss
            torch.save(concept_autoencoder.state_dict(), best_concept_model_path)
            print(f"第 {fold_ce} 折的模型表现最好，已保存为 {best_concept_model_path}")

    print(f"\n最好的概念自编码器验证损失: {best_concept_val_loss:.4f}")
    print(f"最好的概念自编码器模型已保存为: {best_concept_model_path}")

    # 加载最佳模型进行激活模式分析
    best_concept_autoencoder = ConceptAutoEncoder(input_dim, encoding_layers).to(device)
    best_concept_autoencoder.load_state_dict(torch.load(best_concept_model_path, map_location=device))
    best_concept_autoencoder.eval()

    # 分析激活模式
    features_tensor = torch.FloatTensor(dataset.features).to(device)
    best_concept_autoencoder.analyze_activation_patterns(features_tensor)


# =======================
# 6. 运行主程序
# =======================
if __name__ == "__main__":
    main()
