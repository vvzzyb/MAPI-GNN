import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
import random
import pandas as pd

# 配置参数
CONFIG = {
    'model_name': 'resnet18',  # 可选: cnn, densenet, senet, resnet18, resnet50, resnet152
    'modalities': ['adc', 'hbv', 't2w'],  # 可选: ['adc'], ['hbv'], ['t2w'] 或 ['adc', 'hbv', 't2w']
    'num_epochs': 20,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_workers': 4,
    'outer_folds': 5,  # 外层交叉验证折数
    'inner_folds': 4,  # 内层交叉验证折数
    'save_features': True,  # 是否保存特征
    'feature_save_dir': 'features'  # 特征保存目录
}


class MedicalVolumeDataset(Dataset):
    def __init__(self, root_dir, modalities, transform=None):
        self.root_dir = root_dir
        self.modalities = modalities
        self.transform = transform
        self.samples = []
        self.load_samples()

    def load_samples(self):
        for label in ['0', '1']:
            label_dir = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_dir):
                continue

            patients = os.listdir(label_dir)
            for patient in patients:
                patient_dir = os.path.join(label_dir, patient)
                if not os.path.isdir(patient_dir):
                    continue

                patient_files = {mod: [] for mod in self.modalities}
                valid_sample = True

                for mod in self.modalities:
                    modality_dir = [d for d in os.listdir(patient_dir)
                                    if mod in d and os.path.isdir(os.path.join(patient_dir, d))]
                    if not modality_dir:
                        valid_sample = False
                        break
                    modality_dir = os.path.join(patient_dir, modality_dir[0])

                    files = [os.path.join(modality_dir, f)
                             for f in sorted(os.listdir(modality_dir))
                             if os.path.isfile(os.path.join(modality_dir, f))]
                    if not files:
                        valid_sample = False
                        break
                    patient_files[mod] = files

                if valid_sample:
                    # 保存病人ID（从路径中提取）
                    self.samples.append((patient_files, int(label), patient))

        print(f"Total number of samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_files, label, patient_id = self.samples[idx]
        volumes = {}

        for mod in self.modalities:
            volume = self.load_and_preprocess(patient_files[mod])
            volumes[mod] = volume

        sample = {**volumes, 'label': label, 'patient_id': patient_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_and_preprocess(self, file_paths):
        images = []
        for file_path in file_paths:
            img = Image.open(file_path).resize((224, 224)).convert('RGB')
            img = np.array(img) / 255.0
            images.append(img)

        if len(images) > 25:
            images = images[:25]
        elif len(images) < 25:
            # 使用已有切片的平均值进行填充
            mean_image = np.mean(images, axis=0)
            while len(images) < 25:
                images.append(mean_image)

        return np.stack(images)


class ToTensor(object):
    def __call__(self, sample):
        tensor_sample = {}
        for key in sample:
            if key == 'label':
                tensor_sample[key] = torch.tensor(sample[key]).long()
            elif key == 'patient_id':
                tensor_sample[key] = sample[key]
            else:
                tensor_sample[key] = torch.from_numpy(sample[key]).permute(3, 0, 1, 2).float()
        return tensor_sample


class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        # 移除原始模型的最后一层
        if isinstance(original_model, nn.Sequential):
            self.features = nn.Sequential(*list(original_model.children())[:-1])
        else:
            # 对于ResNet等预训练模型
            self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        return self.features(x)


def get_model(model_name, num_modalities):
    in_channels = 75 * num_modalities  # 25 slices * 3 channels * num_modalities

    if model_name == 'cnn':
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 2)
        )

    elif model_name == 'densenet':
        model = models.densenet121(pretrained=True)
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model

    elif model_name == 'senet':
        model = models.resnet50(pretrained=True)
        model.layer1 = nn.Sequential(SEBlock(256), *list(model.layer1.children()))
        model.layer2 = nn.Sequential(SEBlock(512), *list(model.layer2.children()))
        model.layer3 = nn.Sequential(SEBlock(1024), *list(model.layer3.children()))
        model.layer4 = nn.Sequential(SEBlock(2048), *list(model.layer4.children()))
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    elif model_name in ['resnet18', 'resnet50', 'resnet152']:
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet152(pretrained=True)

        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    else:
        raise ValueError(f"Unknown model name: {model_name}")


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def extract_features(model, feature_extractor, data_loader, device):
    model.eval()
    feature_extractor.eval()
    features_list = []
    labels_list = []
    patient_ids_list = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = torch.cat([batch[mod] for mod in CONFIG['modalities']], dim=1)
            B, C, D, H, W = inputs.shape
            inputs = inputs.transpose(1, 2).reshape(B, C * D, H, W)
            inputs = inputs.to(device)
            labels = batch['label']
            patient_ids = batch['patient_id']

            # 提取特征
            features = feature_extractor(inputs)
            features = features.view(features.size(0), -1)

            features_list.append(features.cpu().numpy())
            labels_list.extend(labels.numpy())
            patient_ids_list.extend(patient_ids)

    features = np.concatenate(features_list, axis=0)
    return features, labels_list, patient_ids_list


def save_features(features, labels, patient_ids, fold_idx, save_dir=CONFIG['feature_save_dir']):
    """
    保存特征到CSV文件，格式为：
    PTID, label, feature1, feature2, ...
    """
    os.makedirs(save_dir, exist_ok=True)

    # 创建特征列名
    feature_cols = [f'Frontal_{i + 1}' for i in range(features.shape[1])]

    # 创建DataFrame
    data = {
        'PTID': patient_ids,
        'DX_bl': labels
    }
    # 添加特征列
    for i, col in enumerate(feature_cols):
        data[col] = features[:, i]

    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    save_path = os.path.join(save_dir, f'fold_{fold_idx}_features.csv')
    df.to_csv(save_path, index=False)
    print(f"特征已保存到: {save_path}")


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []

    scaler = torch.cuda.amp.GradScaler()

    for batch in train_loader:
        inputs = torch.cat([batch[mod] for mod in CONFIG['modalities']], dim=1)
        B, C, D, H, W = inputs.shape
        inputs = inputs.transpose(1, 2).reshape(B, C * D, H, W)
        inputs = inputs.to(device)
        labels = batch['label'].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = torch.cat([batch[mod] for mod in CONFIG['modalities']], dim=1)
            B, C, D, H, W = inputs.shape
            inputs = inputs.transpose(1, 2).reshape(B, C * D, H, W)
            inputs = inputs.to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    metrics = {
        'train_loss': train_loss / len(train_loader),
        'val_loss': val_loss / len(val_loader),
        'train_acc': accuracy_score(train_labels, train_preds),
        'val_acc': accuracy_score(val_labels, val_preds),
        'train_auc': roc_auc_score(train_labels, train_preds),
        'val_auc': roc_auc_score(val_labels, val_preds),
        'current_lr': optimizer.param_groups[0]['lr']
    }

    return metrics


def main():
    # 设置随机种子
    torch.manual_seed(37)
    random.seed(37)
    np.random.seed(37)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集和转换
    transform = transforms.Compose([ToTensor()])
    dataset = MedicalVolumeDataset(
        root_dir='picai',
        modalities=CONFIG['modalities'],
        transform=transform
    )

    # 设置5折交叉验证
    n_splits = 5
    indices = np.arange(len(dataset))
    fold_size = len(dataset) // n_splits

    # 打乱数据集索引
    np.random.seed(42)
    np.random.shuffle(indices)

    # 将索引分成5份
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(n_splits)]
    fold_results = []

    # 对每一折进行训练和测试
    for test_fold_idx in range(n_splits):
        print(f'\nTest Fold: {test_fold_idx + 1}')
        print('=' * 50)

        # 确定验证集的索引（下一折，如果是最后一折则使用第一折）
        val_fold_idx = (test_fold_idx + 1) % n_splits

        # 获取测试集和验证集的索引
        test_indices = folds[test_fold_idx]
        val_indices = folds[val_fold_idx]

        # 获取训练集的索引（除了测试集和验证集的所有数据）
        train_indices = []
        for i in range(n_splits):
            if i != test_fold_idx and i != val_fold_idx:
                train_indices.extend(folds[i])

        # 创建数据加载器
        train_loader = DataLoader(
            dataset,
            batch_size=CONFIG['batch_size'],
            sampler=SubsetRandomSampler(train_indices),
            num_workers=CONFIG['num_workers'],
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset,
            batch_size=CONFIG['batch_size'],
            sampler=SubsetRandomSampler(val_indices),
            num_workers=CONFIG['num_workers'],
            pin_memory=True
        )
        test_loader = DataLoader(
            dataset,
            batch_size=CONFIG['batch_size'],
            sampler=SubsetRandomSampler(test_indices),
            num_workers=CONFIG['num_workers'],
            pin_memory=True
        )

        # 输出每个集合的大小
        print(f'训练集样本数: {len(train_indices)} (60%)')
        print(f'验证集样本数: {len(val_indices)} (20%)')
        print(f'测试集样本数: {len(test_indices)} (20%)')
        print(f'训练集折: {[i + 1 for i in range(n_splits) if i != test_fold_idx and i != val_fold_idx]}')
        print(f'验证集折: {val_fold_idx + 1}')
        print(f'测试集折: {test_fold_idx + 1}')

        # 获取模型和特征提取器
        model = get_model(CONFIG['model_name'], len(CONFIG['modalities']))
        feature_extractor = FeatureExtractor(model)
        model = model.to(device)
        feature_extractor = feature_extractor.to(device)

        # 模型、损失函数、优化器和学习率调度器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3, verbose=True
        )

        # 训练
        best_val_auc = 0
        best_metrics = None
        patience = 5
        patience_counter = 0

        for epoch in range(CONFIG['num_epochs']):
            metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

            print(f'Epoch {epoch + 1}/{CONFIG["num_epochs"]}:')
            print(f'Train Loss: {metrics["train_loss"]:.4f}, Val Loss: {metrics["val_loss"]:.4f}')
            print(f'Train Acc: {metrics["train_acc"]:.4f}, Val Acc: {metrics["val_acc"]:.4f}')
            print(f'Train AUC: {metrics["train_auc"]:.4f}, Val AUC: {metrics["val_auc"]:.4f}')

            if metrics['val_auc'] > best_val_auc:
                best_val_auc = metrics['val_auc']
                best_metrics = metrics
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'feature_extractor_state_dict': feature_extractor.state_dict()
                }, f'best_model_fold_{test_fold_idx + 1}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping triggered')
                    break

        # 加载最佳模型
        checkpoint = torch.load(f'best_model_fold_{test_fold_idx + 1}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])

        # 如果需要保存特征
        if CONFIG['save_features']:
            print(f'\n正在提取并保存第 {test_fold_idx + 1} 折的特征...')

            # 收集所有样本的特征
            all_features = []
            all_labels = []
            all_ids = []

            # 收集训练集特征
            train_features, train_labels, train_ids = extract_features(
                model, feature_extractor, train_loader, device)
            all_features.append(train_features)
            all_labels.extend(train_labels)
            all_ids.extend(train_ids)

            # 收集验证集特征
            val_features, val_labels, val_ids = extract_features(
                model, feature_extractor, val_loader, device)
            all_features.append(val_features)
            all_labels.extend(val_labels)
            all_ids.extend(val_ids)

            # 收集测试集特征
            test_features, test_labels, test_ids = extract_features(
                model, feature_extractor, test_loader, device)
            all_features.append(test_features)
            all_labels.extend(test_labels)
            all_ids.extend(test_ids)

            # 合并所有特征
            all_features = np.concatenate(all_features, axis=0)

            # 保存这一折的所有特征
            save_features(all_features, all_labels, all_ids, test_fold_idx + 1)

            print(f'特征维度: {all_features.shape[1]}')

        # 在测试集上评估
        model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = torch.cat([batch[mod] for mod in CONFIG['modalities']], dim=1)
                B, C, D, H, W = inputs.shape
                inputs = inputs.transpose(1, 2).reshape(B, C * D, H, W)
                inputs = inputs.to(device)
                labels = batch['label'].to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # 计算测试集指标
        test_acc = accuracy_score(test_labels, test_preds)
        test_auc = roc_auc_score(test_labels, test_preds)

        fold_results.append({
            'test_fold': test_fold_idx + 1,
            'val_fold': val_fold_idx + 1,
            'val_acc': best_metrics['val_acc'],
            'val_auc': best_metrics['val_auc'],
            'test_acc': test_acc,
            'test_auc': test_auc
        })

        print(f'\nFold {test_fold_idx + 1} Results:')
        print(f'Validation Fold: {val_fold_idx + 1}')
        print(f'Best Val Acc: {best_metrics["val_acc"]:.4f}, Test Acc: {test_acc:.4f}')
        print(f'Best Val AUC: {best_metrics["val_auc"]:.4f}, Test AUC: {test_auc:.4f}')

    # 打印总体结果
    print('\nOverall Cross-validation Results:')
    metrics_names = ['val_acc', 'val_auc', 'test_acc', 'test_auc']
    for metric in metrics_names:
        values = [fold[metric] for fold in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f'Mean {metric}: {mean:.4f} ± {std:.4f}')


if __name__ == '__main__':
    main()