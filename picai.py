# ***端到端（可选择样本平衡）+更多指标+嵌套交叉验证+mask优化
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import dgl
from dgl.nn import GraphConv, GATConv
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix


# =======================
# 设置随机种子
# =======================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =======================
# 配置类
# =======================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据相关
    graphs_dir = "./saved_models"
    csv_path = "fold_4_features.csv"  # basefeature生成的特征文件
    random_state = 42
    n_splits = 5
    use_balanced_sampling = False  # 新增：是否使用样本平衡

    # 模型架构参数
    plane_in_dim = 1
    plane_hidden_dim = 64  # 128
    plane_out_dim = 32  # 64
    num_planes = 16  # 潜在空间维度
    num_heads = 2  # 4
    node_hidden_dim = 128  # 256
    num_classes = 2

    # 训练参数
    '''num_epochs = 20
    batch_size = 32
    patience = 10
    learning_rate = 0.001
    weight_decay = 5e-4
    feat_dropout = 0.3
    gnn_dropout = 0.3

    # 损失权重
    lambda_reconstruction = 0.1  # 重构损失的权重
    lambda_classification = 1.0'''  # 分类损失的权重

    # 图构建参数
    k_neighbors = 5

    # 训练参数
    num_epochs = 30
    batch_size = 4  # 32
    patience = 10
    learning_rate = 0.001
    weight_decay = 1e-3  # 原来是5e-4
    feat_dropout = 0.5  # 原来是0.3
    gnn_dropout = 0.5  # 原来是0.3

    # 损失权重
    lambda_reconstruction = 0.3  # 重构损失的权重，原来是0.1
    lambda_classification = 1.0  # 分类损失的权重


# 设置随机种子
set_seed(Config.random_state)


# =======================
# 平面图编码器
# =======================
class PlaneGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=Config.num_heads):
        super(PlaneGraphEncoder, self).__init__()

        # 第一层GAT
        self.gat1 = GATConv(in_dim, hidden_dim, num_heads,
                            feat_drop=Config.feat_dropout,
                            attn_drop=Config.feat_dropout,
                            residual=True,
                            allow_zero_in_degree=True)

        # 第二层GAT
        self.gat2 = GATConv(hidden_dim * num_heads, out_dim, 1,
                            feat_drop=Config.feat_dropout,
                            attn_drop=Config.feat_dropout,
                            residual=True,
                            allow_zero_in_degree=True)

        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(out_dim)

        # 重构器
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim * num_heads),
            nn.BatchNorm1d(hidden_dim * num_heads),
            nn.ReLU(),
            nn.Dropout(Config.feat_dropout),
            nn.Linear(hidden_dim * num_heads, in_dim)
        )

    def forward(self, g, return_reconstruction=True):
        h = g.ndata['feat']
        if h.dim() == 3:
            h = h.squeeze(1)

        # 第一层GAT
        h = self.gat1(g, h)
        h = h.reshape(h.shape[0], -1)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=Config.feat_dropout, training=self.training)

        # 第二层GAT
        h = self.gat2(g, h)
        h = h.squeeze(1)
        h = self.bn2(h)
        h = F.relu(h)

        # 计算图级别表示
        g.ndata['h'] = h
        graph_rep = dgl.mean_nodes(g, 'h')

        if return_reconstruction:
            # 重构原始特征
            reconstruction = self.decoder(h)
            return graph_rep, reconstruction
        return graph_rep


class MaskedGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(MaskedGraphConv, self).__init__()
        self.conv = GraphConv(in_feats, out_feats, norm='both', allow_zero_in_degree=True)

    def forward(self, g, h, mask):
        with g.local_scope():
            # 获取图的设备
            device = g.device

            # 获取边的源节点和目标节点
            src, dst = g.edges()
            # 确保mask在正确的设备上
            mask = mask.to(device)
            # 创建边掩码并确保在正确的设备上
            edge_mask = (mask[src] & mask[dst]).to(device)

            # 存储原始边权重
            original_weights = None
            if 'weight' in g.edata:
                original_weights = g.edata['weight']

            # 应用边掩码，确保在正确的设备上
            g.edata['_masked_weight'] = edge_mask.float().unsqueeze(-1).to(device)
            if original_weights is not None:
                g.edata['_masked_weight'] = g.edata['_masked_weight'] * original_weights

            # 执行带掩码的图卷积
            return self.conv(g, h, edge_weight=g.edata['_masked_weight'])


# =======================
# 端到端模型
# =======================
class End2EndModel(nn.Module):
    def __init__(self, input_dim, original_feature_dim):
        super(End2EndModel, self).__init__()

        # 平面图编码器
        self.plane_encoders = nn.ModuleList([
            PlaneGraphEncoder(
                in_dim=Config.plane_in_dim,
                hidden_dim=Config.plane_hidden_dim,
                out_dim=Config.plane_out_dim
            ) for _ in range(Config.num_planes)
        ])

        # 计算融合后的特征维度
        self.fused_dim = original_feature_dim + Config.num_planes * Config.plane_out_dim

        # 特征转换层
        self.feature_transform = nn.Sequential(
            nn.Linear(self.fused_dim, Config.node_hidden_dim),
            nn.BatchNorm1d(Config.node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.gnn_dropout)
        )

        # 节点分类GNN
        self.node_gnn = nn.ModuleList([
            MaskedGraphConv(Config.node_hidden_dim, Config.node_hidden_dim),
            MaskedGraphConv(Config.node_hidden_dim, Config.node_hidden_dim),
            MaskedGraphConv(Config.node_hidden_dim, Config.node_hidden_dim)
        ])

        # 批归一化层
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(Config.node_hidden_dim) for _ in range(3)
        ])

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(Config.node_hidden_dim, Config.node_hidden_dim // 2),
            nn.LayerNorm(Config.node_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.gnn_dropout),
            nn.Linear(Config.node_hidden_dim // 2, Config.num_classes)
        )

    def forward(self, patient_graphs_list, patient_graph, original_features, mask):
        batch_size = len(patient_graphs_list)  # 添加batch_size定义
        total_reconstruction_loss = 0  # 初始化重构损失
        all_patient_features = []

        # 处理每个病人的平面图
        for patient_graphs in patient_graphs_list:
            patient_plane_features = []

            # 处理当前病人的每个平面图
            for plane_idx, plane_graph in enumerate(patient_graphs):
                # 获取平面图表示和重构
                graph_rep, reconstruction = self.plane_encoders[plane_idx](plane_graph)

                # 计算重构损失
                original = plane_graph.ndata['feat']
                if original.dim() == 3:
                    original = original.squeeze(1)
                if reconstruction.dim() == 3:
                    reconstruction = reconstruction.squeeze(1)
                reconstruction_loss = F.mse_loss(reconstruction, original)
                total_reconstruction_loss += reconstruction_loss

                # 收集平面图表示
                patient_plane_features.append(graph_rep)

            # 拼接当前病人的所有平面图特征
            patient_all_planes = torch.cat(patient_plane_features, dim=0)
            all_patient_features.append(patient_all_planes)

        # 将所有病人的平面图特征堆叠起来
        all_plane_features = torch.stack(all_patient_features)
        # 将平面图特征调整为2D: [num_patients, num_planes * plane_out_dim]
        all_plane_features = all_plane_features.view(all_plane_features.size(0), -1)

        # 将平面图特征与原始特征拼接
        node_features = torch.cat([original_features, all_plane_features], dim=1)

        # 节点分类GNN前向传播
        h = self.feature_transform(node_features)
        h_list = [h]

        # 使用mask进行消息传递
        for gnn, bn in zip(self.node_gnn, self.bn_layers):
            h_new = gnn(patient_graph, h, mask)  # 传入mask参数
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=Config.gnn_dropout, training=self.training)
            h = h_new + h  # 残差连接
            h_list.append(h)

        # 取所有层输出的平均
        h = torch.stack(h_list, dim=0).mean(dim=0)

        # 分类
        logits = self.classifier(h)

        # 确保返回正确的重构损失值
        avg_recon_loss = total_reconstruction_loss / (batch_size * Config.num_planes)

        return logits, avg_recon_loss


# =======================
# 数据集类
# =======================
class PatientDataset:
    def __init__(self, csv_path):  # 参数名是 csv_path
        print("\n==== 加载数据集 ====")
        # 基础数据加载
        self.df = pd.read_csv(csv_path)  # 这里应该用 csv_path 而不是 csv_file
        self.patient_ids = self.df.iloc[:, 0].values
        self.labels = self.df.iloc[:, 1].values
        self.original_features = self.df.iloc[:, 2:].values

        # 维度计算
        self.original_feature_dim = self.original_features.shape[1]
        self.plane_feature_dim = Config.num_planes * Config.plane_out_dim
        self.fused_dim = self.original_feature_dim + self.plane_feature_dim

        print(f"原始特征维度: {self.original_features.shape}")

        # 验证图文件是否存在
        self._verify_graph_files()

        # 加载保存的平面图
        print("加载已保存的平面图...")
        self.patient_graphs = self.load_saved_graphs()
        print(f"成功加载了 {len(self.patient_graphs)} 个病人的平面图")
        print(f"每个病人有 {len(self.patient_graphs[0])} 个平面图")

        # 构建病人关系图
        print("构建病人关系图...")
        self.relation_graph = self.build_relation_graph()

        print(f"融合特征维度: ({len(self.patient_ids)}, {self.fused_dim})")
        print(f"标签分布: {np.bincount(self.labels)}")

    def _verify_graph_files(self):
        """验证所有需要的图文件是否存在"""
        missing_files = []
        for patient_id, label in zip(self.patient_ids, self.labels):
            graph_path = os.path.join(Config.graphs_dir, "graphs",
                                      f"{int(label)}", f"{patient_id}", "graphs.pt")
            if not os.path.exists(graph_path):
                missing_files.append(graph_path)

        if missing_files:
            raise FileNotFoundError(
                f"找不到以下图文件，请确保已运行 graph.py:\n" +
                "\n".join(missing_files)
            )

    def load_saved_graphs(self):
        """加载已保存的平面图"""
        patient_graphs = []

        for patient_id, label in zip(self.patient_ids, self.labels):
            # 构建图文件路径
            graph_path = os.path.join(Config.graphs_dir, "graphs",
                                      f"{int(label)}", f"{patient_id}", "graphs.pt")

            # 加载图
            try:
                graphs = torch.load(graph_path)
                # 确保所有图都是 DGL 图对象
                if not all(isinstance(g, dgl.DGLGraph) for g in graphs):
                    raise TypeError(f"病人 {patient_id} 的某些图不是 DGL 图对象")
                patient_graphs.append(graphs)
            except Exception as e:
                raise RuntimeError(f"加载病人 {patient_id} 的图时出错: {str(e)}")

        return patient_graphs

    def build_relation_graph(self):
        """构建病人关系图"""
        features_tensor = torch.FloatTensor(self.original_features)
        sim_matrix = torch.cdist(features_tensor, features_tensor)

        _, indices = torch.topk(sim_matrix, k=Config.k_neighbors + 1, dim=1, largest=False)

        src = []
        dst = []
        for i in range(len(self.patient_ids)):
            for j in indices[i][1:]:
                src.extend([i, j])
                dst.extend([j, i])

        g = dgl.graph((torch.tensor(src), torch.tensor(dst)))
        g.ndata['feat'] = features_tensor
        g.ndata['label'] = torch.LongTensor(self.labels)

        return g


# =======================
# 训练函数
# =======================
def train_model(model, dataset, train_mask, val_mask, device, fold_dir):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=Config.learning_rate,
                                 weight_decay=Config.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                  patience=5, verbose=True)

    criterion = nn.CrossEntropyLoss()

    # 确保所有数据都在正确的设备上
    patient_graph = dataset.relation_graph.to(device)
    original_features = torch.FloatTensor(dataset.original_features).to(device)
    labels = torch.LongTensor(dataset.labels).to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    # 将平面图移动到设备上
    patient_graphs_list = [
        [g.to(device) for g in patient_graphs]
        for patient_graphs in dataset.patient_graphs
    ]

    best_val_acc = 0
    no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(Config.num_epochs):
        # 训练
        model.train()
        optimizer.zero_grad()

        # 前向传播
        logits, recon_loss = model(patient_graphs_list, patient_graph, original_features, train_mask)
        clf_loss = criterion(logits[train_mask], labels[train_mask])

        # 总损失
        loss = (Config.lambda_classification * clf_loss +
                Config.lambda_reconstruction * recon_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 计算训练指标
        _, train_pred = torch.max(logits[train_mask], dim=1)
        train_acc = (train_pred == labels[train_mask]).float().mean()

        # 验证
        model.eval()
        with torch.no_grad():
            val_logits, val_recon_loss = model(patient_graphs_list, patient_graph, original_features, val_mask)
            val_clf_loss = criterion(val_logits[val_mask], labels[val_mask])
            val_loss = (Config.lambda_classification * val_clf_loss +
                        Config.lambda_reconstruction * val_recon_loss)

            _, val_pred = torch.max(val_logits[val_mask], dim=1)
            val_acc = (val_pred == labels[val_mask]).float().mean()

        # 更新学习率
        scheduler.step(val_acc)

        # 记录历史
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss.item())
        history['val_acc'].append(val_acc.item())

        print(f'Epoch {epoch:03d}, '
              f'Train Loss: {loss.item():.4f}, '
              f'Train Acc: {train_acc.item():.4f}, '
              f'Val Loss: {val_loss.item():.4f}, '
              f'Val Acc: {val_acc.item():.4f}, '
              f'Recon Loss: {recon_loss.item():.4f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            best_model_path = os.path.join(fold_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"模型更新已保存到: {best_model_path}")
        else:
            no_improve += 1
            if no_improve >= Config.patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

    return history, best_val_acc.item()


def evaluate(model, dataset, mask, device, save_predictions=False, fold=None, save_dir=None):
    """评估函数"""
    model.eval()
    with torch.no_grad():
        patient_graph = dataset.relation_graph.to(device)
        original_features = torch.FloatTensor(dataset.original_features).to(device)
        labels = torch.LongTensor(dataset.labels).to(device)
        mask = mask.to(device)

        patient_graphs_list = [
            [g.to(device) for g in patient_graphs]
            for patient_graphs in dataset.patient_graphs
        ]

        logits, _ = model(patient_graphs_list, patient_graph, original_features, mask)

        # 获取预测概率和预测标签
        probs = torch.softmax(logits[mask], dim=1)
        _, preds = torch.max(logits[mask], dim=1)

        # 将张量移到CPU并转换为numpy数组
        y_true = labels[mask].cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_prob = probs.cpu().numpy()

        # 如果需要保存预测结果
        if save_predictions and fold is not None and save_dir is not None:
            # 获取被mask选中的病人ID
            mask_cpu = mask.cpu()  # 先将mask移到CPU
            patient_ids = dataset.patient_ids[mask_cpu.numpy()]  # 使用numpy索引

            # 创建预测结果DataFrame
            results_df = pd.DataFrame({
                'patient_id': patient_ids,
                'true_label': y_true,
                'predicted_label': y_pred,
                'prob_class_0': y_prob[:, 0],
                'prob_class_1': y_prob[:, 1]
            })

            # 保存预测结果
            pred_save_path = os.path.join(save_dir, f'fold_{fold}_predictions.csv')
            results_df.to_csv(pred_save_path, index=False)
            print(f"预测结果已保存到: {pred_save_path}")

        # 计算评估指标
        metrics = {
            'acc': (y_pred == y_true).mean(),
            'auc': roc_auc_score(y_true, y_prob[:, 1]),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp)
        metrics['sensitivity'] = metrics['recall']

    return metrics


def plot_learning_curves(history, fold, save_dir):
    """绘制学习曲线"""
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Fold {fold} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'Fold {fold} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'learning_curves_fold_{fold}.png'))
    plt.close()


def save_experiment_results(config, cv_results):
    """保存实验配置和结果"""
    results = {
        'config': {k: v for k, v in config.__dict__.items()
                   if not k.startswith('__')},
        'cv_results': cv_results
    }

    # 保存路径
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config.graphs_dir,
                             f'experiment_results_{timestamp}.json')

    # 转换为可序列化的格式
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, torch.device):
            return str(obj)
        return obj

    serializable_results = json.loads(
        json.dumps(results, default=convert_to_serializable)
    )

    # 保存结果
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n实验配置和结果已保存到: {save_path}")


def create_nested_cv_folds(dataset, n_splits=5, random_state=37, use_balanced=True):
    """
    创建嵌套交叉验证的索引

    Args:
        dataset: PatientDataset 实例
        n_splits: 交叉验证折数
        random_state: 随机种子
        use_balanced: 是否使用样本平衡策略

    Returns:
        list of tuples: 每个元素包含 (train_idx, val_idx, test_idx)
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from collections import Counter

    labels = dataset.labels
    num_nodes = len(dataset.patient_ids)

    # 创建两个分层K折交叉验证器
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_folds = []
    fold_assignments = np.zeros(num_nodes)  # 用于追踪每个样本被分配到哪个fold

    # 首先将所有数据划分为5个fold
    for fold_idx, (_, fold_idx_test) in enumerate(outer_cv.split(np.zeros(num_nodes), labels), 1):
        fold_assignments[fold_idx_test] = fold_idx

    # 对每个测试fold，创建训练-验证划分
    for test_fold in range(1, n_splits + 1):
        # 获取当前test fold的索引
        test_idx = np.where(fold_assignments == test_fold)[0]
        # 获取其他fold的索引
        other_idx = np.where(fold_assignments != test_fold)[0]

        # 从其他fold中选择一个作为验证集
        val_fold = test_fold % n_splits + 1
        if val_fold == test_fold:
            val_fold = val_fold % n_splits + 1
        val_idx = np.where(fold_assignments == val_fold)[0]

        # 剩余的fold作为训练集
        train_idx = np.where((fold_assignments != test_fold) &
                             (fold_assignments != val_fold))[0]

        if use_balanced:
            # 对训练集进行平衡采样
            train_labels = labels[train_idx]
            class_counts = Counter(train_labels)
            min_class_count = min(class_counts.values())

            balanced_train_indices = []
            for label in np.unique(labels):
                label_indices = train_idx[train_labels == label]
                selected_indices = np.random.RandomState(random_state).choice(
                    label_indices,
                    size=min_class_count,
                    replace=False
                )
                balanced_train_indices.extend(selected_indices)

            # 随机打乱训练集
            balanced_train_indices = np.array(balanced_train_indices)
            np.random.RandomState(random_state).shuffle(balanced_train_indices)
            train_idx = balanced_train_indices

        all_folds.append((train_idx, val_idx, test_idx))

        # 打印每个fold的分配情况
        print(f"\nFold {test_fold} 分配:")
        print(f"训练集大小: {len(train_idx)} ({len(train_idx) / num_nodes * 100:.1f}%)")
        print(f"验证集大小: {len(val_idx)} ({len(val_idx) / num_nodes * 100:.1f}%)")
        print(f"测试集大小: {len(test_idx)} ({len(test_idx) / num_nodes * 100:.1f}%)")
        print(f"训练集类别分布: {np.bincount(labels[train_idx])}")
        print(f"验证集类别分布: {np.bincount(labels[val_idx])}")
        print(f"测试集类别分布: {np.bincount(labels[test_idx])}")

    return all_folds


# =======================
# 主函数
# =======================
def main():
    print(f"使用设备: {Config.device}")
    print(f"样本平衡策略: {'启用' if Config.use_balanced_sampling else '禁用'}")

    # 加载数据集
    print("\n==== 加载数据集 ====")
    dataset = PatientDataset(Config.csv_path)

    # 打印原始数据集的类别分布
    print("原始数据集类别分布:", np.bincount(dataset.labels))

    # 创建保存目录
    results_dir = os.path.join(Config.graphs_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 使用嵌套交叉验证
    print("\n==== 创建嵌套交叉验证折 ====")
    folds = create_nested_cv_folds(
        dataset,
        n_splits=Config.n_splits,
        random_state=Config.random_state,
        use_balanced=Config.use_balanced_sampling
    )

    # 记录交叉验证结果
    cv_results = []

    # 开始交叉验证
    for fold, (train_idx, val_idx, test_idx) in enumerate(folds, 1):
        print(f"\n==== 开始第 {fold}/{Config.n_splits} 折交叉验证 ====")

        # 创建每个fold的保存目录
        fold_dir = os.path.join(results_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        # 创建掩码
        train_mask = torch.zeros(len(dataset.patient_ids), dtype=torch.bool).to(Config.device)
        val_mask = torch.zeros(len(dataset.patient_ids), dtype=torch.bool).to(Config.device)
        test_mask = torch.zeros(len(dataset.patient_ids), dtype=torch.bool).to(Config.device)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # 打印每个集合的大小和类别分布
        print("\n数据集分布情况:")
        print(f"训练集大小: {train_mask.sum().item()}")
        print(f"训练集类别分布: {np.bincount(dataset.labels[train_idx])}")
        print(f"验证集大小: {val_mask.sum().item()}")
        print(f"验证集类别分布: {np.bincount(dataset.labels[val_idx])}")
        print(f"测试集大小: {test_mask.sum().item()}")
        print(f"测试集类别分布: {np.bincount(dataset.labels[test_idx])}")

        # 创建模型
        model = End2EndModel(
            input_dim=dataset.original_features.shape[1],
            original_feature_dim=dataset.original_features.shape[1]
        ).to(Config.device)

        # 训练模型
        history, best_val_acc = train_model(
            model=model,
            dataset=dataset,
            train_mask=train_mask,
            val_mask=val_mask,
            device=Config.device,
            fold_dir=fold_dir
        )

        # 加载最佳模型并进行测试集评估
        best_model_path = os.path.join(fold_dir, 'best_model.pt')
        model.load_state_dict(torch.load(best_model_path))

        # 测试集评估并保存预测结果
        test_metrics = evaluate(
            model,
            dataset,
            test_mask,
            Config.device,
            save_predictions=True,
            fold=fold,
            save_dir=fold_dir
        )

        print("\n测试集评估结果:")
        print(f"准确率 (ACC): {test_metrics['acc']:.4f}")
        print(f"AUC: {test_metrics['auc']:.4f}")
        print(f"精确率 (Precision): {test_metrics['precision']:.4f}")
        print(f"召回率 (Recall/Sensitivity): {test_metrics['recall']:.4f}")
        print(f"F1分数: {test_metrics['f1']:.4f}")
        print(f"特异性 (Specificity): {test_metrics['specificity']:.4f}")

        # 记录结果
        cv_results.append({
            'fold': fold,
            'history': history,
            'best_val_acc': best_val_acc,
            'test_metrics': test_metrics
        })

        # 绘制学习曲线
        plot_learning_curves(history, fold, fold_dir)

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 输出总体结果
    print("\n==== 交叉验证总结 ====")
    metrics_names = ['acc', 'auc', 'precision', 'recall', 'f1', 'specificity', 'sensitivity']
    for metric in metrics_names:
        values = [result['test_metrics'][metric] for result in cv_results]
        print(f"平均 {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # 保存实验结果
    save_experiment_results(Config, cv_results)


if __name__ == "__main__":
    main()