import torch
import torch.nn as nn
import numpy as np
import dgl
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# 加载数据
def load_features(save_dir):
    """加载概念自编码器的编码特征"""
    features = []
    labels = []

    model_path = os.path.join(save_dir, "final_features.pt")
    if os.path.exists(model_path):
        data = torch.load(model_path)
        features = data['features']
        labels = data['labels']
    else:
        raise FileNotFoundError(f"Features file not found: {model_path}")

    return features, labels


def analyze_significant_features(concept_autoencoder, feature, device, percentage=0.05):
    """分析显著特征并返回归一化的影响值"""
    # 确保feature是二维的
    feature = torch.FloatTensor(feature).reshape(1, -1).to(device)  # 修改这里
    feature_dim = feature.shape[1]
    latent_dim = None
    for module in concept_autoencoder.encoder:
        if isinstance(module, nn.Linear):
            latent_dim = module.out_features

    significant_features = [[] for _ in range(latent_dim)]
    feature_impacts = [[] for _ in range(latent_dim)]

    # 获取原始潜在表示
    with torch.no_grad():
        original_latent = concept_autoencoder.encoder(feature)  # 已经是二维的了

    # 计算每个潜在维度的显著特征
    for dim in range(latent_dim):
        impacts = []
        for feature_idx in range(feature_dim):
            perturbed = feature.clone()
            perturbed[0, feature_idx] = 0

            with torch.no_grad():
                perturbed_latent = concept_autoencoder.encoder(perturbed)

            impact = torch.abs(perturbed_latent[0, dim] - original_latent[0, dim]).item()
            impacts.append((impact, feature_idx))

        # 选择影响最大的特征
        sorted_features = sorted(impacts, key=lambda x: x[0], reverse=True)
        num_significant = max(1, int(feature_dim * percentage))
        significant_features[dim] = [f[1] for f in sorted_features[:num_significant]]

        # 归一化影响值
        max_impact = sorted_features[0][0]
        min_impact = sorted_features[-1][0]
        if max_impact > min_impact:
            norm_impacts = [(impact - min_impact) / (max_impact - min_impact)
                            for impact, _ in sorted_features[:num_significant]]
        else:
            norm_impacts = [1.0] * num_significant
        feature_impacts[dim] = norm_impacts

        print(f"潜在维度 {dim + 1} 的显著特征索引: {significant_features[dim]}")

    return significant_features, feature_impacts


def build_graphs(feature, significant_features, feature_impacts, latent_dim, k_neighbors=5, similarity_threshold=0.8):
    """构建带权图，使用余弦相似度 + K近邻"""
    num_features = len(feature)
    graphs = []

    for dim in range(latent_dim):
        g = dgl.graph([])  # 创建一个空图
        g.add_nodes(num_features)  # 为特征添加节点

        g.ndata['feat'] = torch.FloatTensor(feature).reshape(-1, 1)

        sig_features = significant_features[dim]
        impacts = feature_impacts[dim]

        # 计算显著特征之间的余弦相似度
        feature_matrix = np.array([feature[sig] for sig in sig_features]).reshape(-1, 1)
        similarity_matrix = cosine_similarity(feature_matrix)

        # 使用K近邻算法选择最相似的K个特征
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
        nbrs.fit(feature_matrix)
        distances, indices = nbrs.kneighbors(feature_matrix)

        src = []
        dst = []
        weights = []

        for i in range(len(sig_features)):
            for j in range(k_neighbors):
                neighbor_idx = indices[i, j]
                if neighbor_idx != i:  # 排除自己
                    src.append(sig_features[i])
                    dst.append(sig_features[neighbor_idx])
                    weight = (impacts[i] + impacts[neighbor_idx]) / 2  # 权重为影响值的平均值
                    weights.append(weight)

        # 添加边和边的权重
        if src and dst:
            g.add_edges(src, dst)
            g.edata['weight'] = torch.FloatTensor(weights).reshape(-1, 1)

        graphs.append(g)

        print(f"\n潜在维度 {dim + 1} 图的统计信息:")
        print(f"节点数: {g.num_nodes()}")
        print(f"边数: {len(g.edges()[0]) // 2}")  # 除以2因为是双向边
        if 'weight' in g.edata:
            weights = g.edata['weight']
            print(f"边权重范围: [{weights.min().item():.4f}, {weights.max().item():.4f}]")

    return graphs

def main():
    # 配置参数
    save_dir = "./saved_models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    data = pd.read_csv('fold_4_features.csv')  # 加载basefeature生成的特征文件
    features = data.iloc[:, 2:].values  # 特征
    labels = data.iloc[:, 1].values    # 标签
    patient_ids = data.iloc[:, 0].values  # 病人ID

    # 加载概念自编码器
    concept_model_path = os.path.join(save_dir, "best_concept_autoencoder_final.pth")
    if os.path.exists(concept_model_path):
        input_dim = features.shape[1]
        encoding_layers = [256, 64, 16]
        concept_autoencoder = ConceptAutoEncoder(input_dim=input_dim,
                                                 encoding_layers=encoding_layers).to(device)
        concept_autoencoder.load_state_dict(torch.load(concept_model_path, map_location=device))
        concept_autoencoder.eval()
        print(f"已加载概念自编码器模型: {concept_model_path}")
    else:
        raise FileNotFoundError(f"概念自编码器模型未找到: {concept_model_path}")

    # 为每个样本构建图
    os.makedirs(os.path.join(save_dir, "graphs"), exist_ok=True)

    for idx in tqdm(range(len(features)), desc="构建图"):
        feature = features[idx]
        label = labels[idx]
        patient_id = patient_ids[idx]  # 获取当前病人ID

        # 分析显著特征
        significant_features, feature_impacts = analyze_significant_features(
            concept_autoencoder, feature, device, percentage=0.1
        )

        # 构建多个图
        graphs = build_graphs(feature, significant_features, feature_impacts, latent_dim=16, k_neighbors=5)

        # 按病人ID保存图
        save_path = os.path.join(save_dir, "graphs", f"{int(label)}", f"{patient_id}", "graphs.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(graphs, save_path)

    print("已完成所有样本的图构建和保存")


if __name__ == "__main__":
    main()
