import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpatialConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialConvLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        batch_size, num_nodes, in_channels = x.size()
        x = x.reshape(batch_size * num_nodes, in_channels)
        x = self.gcn(x, edge_index)
        x = x.reshape(batch_size, num_nodes, -1)
        return x

class LSTMLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_channels, batch_first=True)

    def forward(self, x_T1, x_T2):
        batch_size, num_nodes, in_channels = x_T1.size()
        x = torch.stack([x_T1, x_T2], dim=2)
        x = x.permute(0, 1, 3, 2).reshape(batch_size * num_nodes, 2, in_channels)
        x, (h_n, c_n) = self.lstm(x)

        return h_n.reshape(batch_size, num_nodes, -1)

class STDFG(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, similarity_threshold=0):
        super(STDFG, self).__init__()
        self.spatial_T1 = SpatialConvLayer(in_channels, spatial_channels)
        self.spatial_T2 = SpatialConvLayer(spatial_channels, spatial_channels)
        self.temporal = LSTMLayer(spatial_channels, spatial_channels)
        self.channel_adjust = nn.Linear(in_channels, spatial_channels)
        self.fc = nn.Linear(spatial_channels, out_channels)
        self.similarity_threshold = similarity_threshold

    def build_edge_index_from_similarity(self, x, height, width):
        batch_size, num_nodes, num_features = x.size()
        similarity_matrix = F.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)
        edge_index_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if similarity_matrix[:, i, j].mean() > self.similarity_threshold and i != j:
                    edge_index_list.append([i, j])

        if len(edge_index_list) == 0:
            edge_index_list = []
            for i in range(height):
                for j in range(width):
                    node = i * width + j
                    if i > 0:
                        edge_index_list.append([node, node - width])
                    if i < height - 1:
                        edge_index_list.append([node, node + width])
                    if j > 0:
                        edge_index_list.append([node, node - 1])
                    if j < width - 1:
                        edge_index_list.append([node, node + 1])

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        return edge_index

    def forward(self, x_T1, x_T2):
        batch_size, channels, height, width = x_T1.size()
        x_T1 = x_T1.reshape(batch_size, height * width, channels)
        x_T2 = x_T2.reshape(batch_size, height * width, channels)
        edge_index = self.build_edge_index_from_similarity(x_T1, height, width).to(device)
        if edge_index.size(1) == 0:
            return torch.zeros_like(x_T1).to(device).reshape(batch_size, channels, height, width)

        x_T1 = self.spatial_T1(x_T1, edge_index)
        x_T2 = self.channel_adjust(x_T2)

        delta_weights = self.temporal(x_T1, x_T2)
        delta_weights = delta_weights.mean(dim=0)
        delta_weights = delta_weights.mean(dim=0)
        delta_weights = delta_weights.unsqueeze(0)

        with torch.no_grad():
            original_weights = self.spatial_T2.gcn.lin.weight
            delta_weights = delta_weights.expand_as(original_weights)
            updated_weights = original_weights + delta_weights

        self.spatial_T2.gcn.lin.weight = nn.Parameter(updated_weights)
        x_T2 = self.spatial_T2(x_T2, edge_index)
        x = F.relu(x_T2)
        x = self.fc(x)
        x = x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)
        return x
