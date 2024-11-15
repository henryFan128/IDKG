import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from models import GCN, GCN_GAT

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, average_precision_score, precision_recall_curve
from neo4j import GraphDatabase
import networkx as nx
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import random

uri = "neo4j://219.228.149.80:7687"  
username = "neo4j"              
password = "xxxxxxxxxxx"     
driver = GraphDatabase.driver(uri, auth=(username, password))

def get_graph_data():
    with driver.session() as session:
        nodes_result = session.run("""
            MATCH (n)
            WHERE n:Drug OR n:Disease OR n:Protein
            RETURN id(n) AS node_id, labels(n) AS labels, 
                   CASE 
                       WHEN n:Drug THEN n.Drug_name 
                       WHEN n:Disease THEN n.Disease_name
                       WHEN n:Protein THEN n.Protein_name
                   END AS name
        """)
        nodes = []
        node_labels = {}
        node_names = {}
        drug_nodes = []
        disease_nodes = []
        protein_nodes = []
        for record in nodes_result:
            node_id = record['node_id']
            labels = record['labels']
            name = record['name']
            nodes.append(node_id)
            node_labels[node_id] = labels
            node_names[node_id] = name
            if 'Drug' in labels:
                drug_nodes.append(node_id)
            elif 'Disease' in labels:
                disease_nodes.append(node_id)
            elif 'Protein' in labels:
                protein_nodes.append(node_id)

        edges_result = session.run("""
            MATCH (a:Drug)-[r]->(b)
            WHERE b:Disease OR b:Protein
            RETURN id(a) AS source, id(b) AS target
        """)
        edges = []
        positive_edges = set()
        for record in edges_result:
            source = record['source']
            target = record['target']
            edges.append((source, target))
            positive_edges.add((source, target))

    return nodes, edges, positive_edges, node_labels, node_names, drug_nodes, disease_nodes, protein_nodes

def train_and_evaluate(model, model_name, train_data, val_data, test_data):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00005)  
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    losses = []

    def train():
        model.train()
        optimizer.zero_grad()
        z = model(train_data)
        pos_edge_index = train_data.edge_label_index[:, train_data.edge_label == 1]
        neg_edge_index = train_data.edge_label_index[:, train_data.edge_label == 0]
        loss = model.loss(z, pos_edge_index, neg_edge_index)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        optimizer.step()
        scheduler.step() 
        losses.append(loss.item())
        return loss.item()

    def test(data):
        model.eval()
        z = model(data)
        pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
        neg_edge_index = data.edge_label_index[:, data.edge_label == 0]
        auc, ap = model.test(z, pos_edge_index, neg_edge_index)
        return auc, ap

    for epoch in range(1, 151):
        loss = train()
        val_auc, val_ap = test(val_data)
        test_auc, test_ap = test(test_data)
        print(f'{model_name} - Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')

    model.eval()

    z = model(test_data)
    pos_edge_index = test_data.edge_label_index[:, test_data.edge_label == 1]
    neg_edge_index = test_data.edge_label_index[:, test_data.edge_label == 0]
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    y_pred = model.predict(z, edge_index)

    test_labels = torch.cat([
        torch.ones(pos_edge_index.size(1), device=device),
        torch.zeros(neg_edge_index.size(1), device=device)
    ])
    accuracy = accuracy_score(test_labels.cpu().numpy(), y_pred.cpu().detach().numpy().round())
    roc_auc = roc_auc_score(test_labels.cpu().numpy(), y_pred.cpu().detach().numpy())
    prc_auc = average_precision_score(test_labels.cpu().numpy(), y_pred.cpu().detach().numpy())

    Accuracy.append(accuracy)
    ROC.append(roc_auc)
    PRC.append(prc_auc)

    fpr, tpr, roc_thresholds = roc_curve(test_labels.cpu().numpy(), y_pred.cpu().detach().numpy())
    roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': roc_thresholds})
    roc_data.to_csv(f'roc_data_{model_name.lower()}.csv', index=False)

    precision, recall, prc_thresholds = precision_recall_curve(test_labels.cpu().numpy(), y_pred.cpu().detach().numpy())
    prc_data = pd.DataFrame({'Precision': precision, 'Recall': recall, 'Threshold': np.append(prc_thresholds, 1)})
    prc_data.to_csv(f'prc_data_{model_name.lower()}.csv', index=False)

    return Accuracy, ROC, PRC

def cross_validate(model_class, model_name, data, k=10):
    accuracies = []
    rocs = []
    prcs = []

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(data.edge_index.t().numpy()):
        train_data, val_data, test_data = transform(data)
        model = model_class(in_channels=data.num_features, out_channels=32)
        accuracy, roc_auc, prc_auc = train_and_evaluate(model, model_name, train_data, val_data, test_data)
        accuracies.append(accuracy)
        rocs.append(roc_auc)
        prcs.append(prc_auc)

    accuracy_mean = np.mean(accuracies)
    accuracy_std = np.std(accuracies)
    roc_mean = np.mean(rocs)
    roc_std = np.std(rocs)
    prc_mean = np.mean(prcs)
    prc_std = np.std(prcs)

    print(f'{model_name} - Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}')
    print(f'{model_name} - ROC AUC: {roc_mean:.4f} ± {roc_std:.4f}')
    print(f'{model_name} - PRC AUC: {prc_mean:.4f} ± {prc_std:.4f}')

    return accuracy_mean, accuracy_std, roc_mean, roc_std, prc_mean, prc_std

nodes, edges, positive_edges, node_labels, node_names, drug_nodes, disease_nodes, protein_nodes = get_graph_data()
node_id_map = {node_id: i for i, node_id in enumerate(nodes)}
reverse_node_id_map = {i: node_id for node_id, i in node_id_map.items()}

edges = [(node_id_map[source], node_id_map[target]) for source, target in edges]

G_full = nx.Graph()
G_full.add_nodes_from(range(len(nodes)))
G_full.add_edges_from(edges)

# change NetworkX Graph into PyTorch Geometric 
edge_index = torch.tensor(list(G_full.edges)).t().contiguous()
x = torch.eye(len(G_full.nodes))
data = Data(x=x, edge_index=edge_index)

# Use RandomLinkSplit to split the data and add negative samples
transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=True)
train_data, val_data, test_data = transform(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Accuracy = []
ROC = []
PRC = []

gcn_model = GCN(in_channels=data.num_features, out_channels=32)
train_and_evaluate(gcn_model, "GCN", train_data, val_data, test_data)

gat_model = GCN_GAT(in_channels=data.num_features, out_channels=32)
train_and_evaluate(gat_model, "GAT", train_data, val_data, test_data)

cross_validate(GCN, "GCN", data)
cross_validate(GCN_GAT, "GAT", data)

torch.save(gcn_model, 'M1.pth')
torch.save(gat_model, 'M2.pth')