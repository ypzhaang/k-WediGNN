from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn.functional as F
from graph_tool.all import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.datasets import WebKB
from torch_geometric.nn import GCNConv
from torch_scatter import scatter
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler,Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
# from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset
import multi_graph
import weisfeiler_leman
import processing_data
import multi_al


device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
graph = torch.load(r'/mnt/5468e/daihuan/Q/SpeqNets-pycharm/graph_data_ucd_pre_correct.pt')
num_nodes = graph.x.shape[0]
edges = graph.edge_index
# edges = edges.numpy()
edges = np.array(edges)
total_data = []
label = []

edge_index = np.array(positive_edges)
edge_index = edge_index.T
x = graph.x
x = x.cpu().detach().numpy()

out_multi_index , in_multi_index = multi_graph.get_multigraph(torch.tensor(edge_index.T), num_nodes)
data = weisfeiler_leman.k_tuple_graph(graph.x, edge_index)
data_out = weisfeiler_leman.k_tuple_graph(graph.x, np.array(out_multi_index))
data_in = weisfeiler_leman.k_tuple_graph(graph.x, np.array(in_multi_index))

data = data.to(device)
data_out = data_out.to(device)
data_in = data_in.to(device)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dim = 256
        self.conv_1_1 = GCNConv(2050, dim)
        self.conv_1_2 = GCNConv(2050, dim)
        self.mlp_1 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))
        self.conv_2_1 = GCNConv(dim, dim)
        self.conv_2_2 = GCNConv(dim, dim)
        self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))
        self.mlp = Sequential(Linear(3 * dim, 2 * dim), ReLU(), Linear(2 * dim, 256))


    def forward(self):
        x_original, edge_index_1, edge_index_2 = data.x, data.edge_index_1, data.edge_index_2
        index_1, index_2 = data.index_1, data.index_2

        x_out, edge_index_out_1, edge_index_out_2 = data_out.x, data_out.edge_index_1, data_out.edge_index_2
        index_out_1, index_out_2 = data_out.index_1, data_out.index_2

        x_in, edge_index_in_1, edge_index_in_2 = data_in.x, data_in.edge_index_1, data_in.edge_index_2
        index_in_1, index_in_2 = data_in.index_1, data_in.index_2

        x_original_1 = F.relu(self.conv_1_1(x_original, edge_index_1))
        x_original_2 = F.relu(self.conv_1_2(x_original, edge_index_2))
        x_original = self.mlp_1(torch.cat([x_original_1, x_original_2], dim=-1))

        x_out_1 = F.relu(self.conv_1_1(x_out, edge_index_out_1))
        x_out_2 = F.relu(self.conv_1_2(x_out, edge_index_out_2))
        x_out = self.mlp_1(torch.cat([x_out_1, x_out_2], dim=-1))

        x_in_1 = F.relu(self.conv_1_1(x_in, edge_index_in_1))
        x_in_2 = F.relu(self.conv_1_2(x_in, edge_index_in_2))
        x_in = self.mlp_1(torch.cat([x_in_1, x_in_2], dim=-1))

        x_1 = F.relu(self.conv_2_1(x_original, edge_index_1))
        x_2 = F.relu(self.conv_2_2(x_original, edge_index_2))
        x_original = self.mlp_2(torch.cat([x_1, x_2], dim=-1))

        x_out_1 = F.relu(self.conv_2_1(x_out, edge_index_out_1))
        x_out_2 = F.relu(self.conv_2_2(x_out, edge_index_out_2))
        x_out = self.mlp_2(torch.cat([x_out_1, x_out_2], dim=-1))

        x_in_1 = F.relu(self.conv_2_1(x_in, edge_index_in_1))
        x_in_2 = F.relu(self.conv_2_2(x_in, edge_index_in_2))
        x_in = self.mlp_2(torch.cat([x_in_1, x_in_2], dim=-1))

        index_1 = index_1.to(torch.int64)
        index_2 = index_2.to(torch.int64)
        index_out_1 = index_out_1.to(torch.int64)
        index_out_2 = index_out_2.to(torch.int64)
        index_in_1 = index_in_1.to(torch.int64)
        index_in_2 = index_in_2.to(torch.int64)
        x_1 = scatter(x_original, index_1, dim=0, reduce="mean")
        x_2 = scatter(x_original, index_2, dim=0, reduce="mean")
        x_out_1 = scatter(x_out, index_out_1, dim=0, reduce="mean")
        x_out_2 = scatter(x_out, index_out_2, dim=0, reduce="mean")
        x_in_1 = scatter(x_in, index_in_1, dim=0, reduce="mean")
        x_in_2 = scatter(x_in, index_in_2, dim=0, reduce="mean")
        c1 = torch.tensor(range(x_1.shape[0]))
        c2 = torch.tensor(range(x_1.shape[0]))
        a1 = torch.unique(index_in_1)
        a2 = torch.unique(index_out_1)
        x_ = self.mlp_3(torch.cat([x_1, x_2], dim=-1))
        x_in = self.mlp_3(torch.cat([x_in_1, x_in_2], dim=-1))
        x_out = self.mlp_3(torch.cat([x_out_1, x_out_2], dim=-1))
        c1 = c1[torch.tensor([i for i in range(len(c1)) if i not in a1], dtype=torch.long)]
        c2 = c2[torch.tensor([i for i in range(len(c2)) if i not in a2], dtype=torch.long)]
        x_in[c1] = x_[c1]
        x_out[c2] = x_[c2]
        x = self.mlp(torch.cat([x_, x_in, x_out], dim=1))

        return F.log_softmax(x, dim=1)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.embedding = Net()
        # 定义共享的神经网络层
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.x = nn.Linear(128,1)
        # self.y1 = nn.Linear(128, 64)
        # self.y2 = nn.Linear(64, 16)
        # self.y3 = nn.Linear(16, 4)
        # self.y4 = nn.Linear(4, 1)

    def forward(self, i, j):
        x1 = self.embedding()[i]
        x2 = self.embedding()[j]
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x2 = F.relu(self.fc1(x2))
        x2 = F.relu(self.fc2(x2))
        x2 = F.relu(self.fc3(x2))
        concatenated = torch.cat((x1, x2, x1-x2, torch.mul(x1, x2)), dim=1)
        final = self.x(concatenated)
        final = torch.sigmoid(final)
        return final
