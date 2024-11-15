import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np  

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.bn1 = BatchNorm(128)
        self.conv2 = GCNConv(128, 64)
        self.bn2 = BatchNorm(64)
        self.conv3 = GCNConv(64, out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(out_channels * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        z = self.conv3(x, edge_index)
        return z

    def decode(self, z, edge_index):
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        z_concat = torch.cat([z_src, z_dst], dim=1)  
        return self.decoder(z_concat).view(-1)

    def loss(self, z, pos_edge_index, neg_edge_index):
        pos_pred = self.decode(z, pos_edge_index)
        neg_pred = self.decode(z, neg_edge_index)
        pos_label = torch.ones(pos_pred.size(0), device=pos_pred.device)
        neg_label = torch.zeros(neg_pred.size(0), device=neg_pred.device)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([pos_label, neg_label], dim=0)
        return F.binary_cross_entropy_with_logits(pred, labels)

    def test(self, z, pos_edge_index, neg_edge_index):
        pos_pred = torch.sigmoid(self.decode(z, pos_edge_index))
        neg_pred = torch.sigmoid(self.decode(z, neg_edge_index))
        pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().detach().numpy()
        pos_y = np.ones(pos_pred.size(0))
        neg_y = np.zeros(neg_pred.size(0))
        y = np.hstack([pos_y, neg_y])
        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)
        return auc, ap

    def predict(self, z, edge_index):
        logits = self.decode(z, edge_index)
        pred = torch.sigmoid(logits)
        return pred

class GCN_GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN_GAT, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.bn1 = BatchNorm(128)
        self.conv2 = GATConv(128, 64, heads=4, concat=True)
        self.bn2 = BatchNorm(256)  # 64 * 4
        self.conv3 = GATConv(256, out_channels, heads=1, concat=False)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(out_channels * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        z = self.conv3(x, edge_index)
        return z

    def decode(self, z, edge_index):
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        z_concat = torch.cat([z_src, z_dst], dim=1)
        return self.decoder(z_concat).view(-1)

    def loss(self, z, pos_edge_index, neg_edge_index):
        pos_pred = self.decode(z, pos_edge_index)
        neg_pred = self.decode(z, neg_edge_index)
        pos_label = torch.ones(pos_pred.size(0), device=pos_pred.device)
        neg_label = torch.zeros(neg_pred.size(0), device=neg_pred.device)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([pos_label, neg_label], dim=0)
        return F.binary_cross_entropy_with_logits(pred, labels)

    def test(self, z, pos_edge_index, neg_edge_index):
        pos_pred = torch.sigmoid(self.decode(z, pos_edge_index))
        neg_pred = torch.sigmoid(self.decode(z, neg_edge_index))
        pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().detach().numpy()
        pos_y = np.ones(pos_pred.size(0))
        neg_y = np.zeros(neg_pred.size(0))
        y = np.hstack([pos_y, neg_y])
        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)
        return auc, ap

    def predict(self, z, edge_index):
        logits = self.decode(z, edge_index)
        pred = torch.sigmoid(logits)
        return pred