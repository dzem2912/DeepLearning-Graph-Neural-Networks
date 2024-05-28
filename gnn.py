import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout, Sigmoid, Sequential
from torch_geometric.nn import GraphConv, NNConv, global_add_pool, global_max_pool


class simpleGNN(torch.nn.Module):
    def __init__(self, dim: int, num_features: int, num_classes: int, dropout_probability: float = 0.2):
        super().__init__()

        self.num_features: int = num_features
        self.dim: int = dim
        self.relu = ReLU()
        self.conv1 = GraphConv(num_features, dim)
        self.conv2 = GraphConv(dim, dim)
        self.conv3 = GraphConv(dim, dim)
        self.conv4 = GraphConv(dim, dim)
        self.conv5 = GraphConv(dim, dim)
        self.lin1 = Linear(dim, num_classes)
        self.sigmoid = Sigmoid()
        
    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.relu(self.conv1(x, edge_index, edge_weight))
        x = self.relu(self.conv2(x, edge_index, edge_weight))
        x = global_max_pool(x, batch)
        x = self.lin1(x)

        x = self.sigmoid(x)
        return x


class GNN1(torch.nn.Module):
    def __init__(self, dim: int, num_features: int, num_classes: int, dropout_probability: float = 0.2):
        super().__init__()

        self.num_features = num_features
        self.dim = dim
        self.dropout_probability: float = dropout_probability

        self.conv1 = GraphConv(num_features, dim)
        self.conv2 = GraphConv(dim, dim)
        self.conv3 = GraphConv(dim, dim)
        self.conv4 = GraphConv(dim, dim)
        self.conv5 = GraphConv(dim, dim)

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, num_classes)
        self.sigmoid = Sigmoid()
    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin2(x)
        return self.sigmoid(x)
    

class GNN2(torch.nn.Module):
    def __init__(self, dim: int, num_features: int, num_classes: int, dropout_probability: float = 0.2):
        super().__init__()

        self.dim = dim
        self.num_features = num_features
        self.num_classes = num_classes

        self.conv1 = GraphConv(num_features, dim)
        self.conv2 = GraphConv(dim, dim)
        self.conv3 = GraphConv(dim, dim)
        self.conv4 = GraphConv(dim, dim)
        self.conv5 = GraphConv(dim, dim)

        self.bn1 = BatchNorm1d(dim)
        self.bn2 = BatchNorm1d(dim)
        self.bn3 = BatchNorm1d(dim)
        self.bn4 = BatchNorm1d(dim)
        self.bn5 = BatchNorm1d(dim)

        # Linear == Dense in tensorflow
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, dim)

        self.lin3 = Linear(dim, num_classes)

        self.relu = ReLU()
        self.dropout = Dropout(p=dropout_probability)
        self.sigmoid = Sigmoid()

    def getNumberOfClasses(self):
        return self.num_classes
    
    def getNumberOfFeatures(self):
        return self.num_features

    def forward(self, x, edge_index, batch, edge_weight=None):
        x1 = self.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))

        # Residual connection -> Vanishing gradient mitigation
        x2 = self.relu(self.bn2(self.conv2(x1, edge_index, edge_weight))) + x1
        x3 = self.relu(self.bn3(self.conv3(x2, edge_index, edge_weight))) + x2
        x4 = self.relu(self.bn4(self.conv4(x3, edge_index, edge_weight))) + x3
        x5 = self.relu(self.bn5(self.conv5(x4, edge_index, edge_weight))) + x4
        
        # Pooling layer
        x = global_max_pool(x5, batch)

        # Dense equivalent, to learned the extracted features
        x = self.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.lin3(x)
        
        x = torch.sigmoid(x)

        return x
    

class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, dim: int, num_features: int, num_classes: int, edge_attr_dim: int, dropout_probability: float = 0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dim = dim
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout_probability = dropout_probability

        self.relu = ReLU()

        # Linear layer to transform edge_attr to consistent dimension
        self.edge_transform = Linear(edge_attr_dim, dim)

        nn1 = Sequential(Linear(dim, 128), ReLU(), Linear(128, num_features * dim))
        self.conv_in = NNConv(num_features, dim, nn1)

        self.conv_layers = torch.nn.ModuleList()
        for _ in range(4):
            nn_conv = Sequential(Linear(dim, 128), ReLU(), Linear(128, dim * dim))
            self.conv_layers.append(NNConv(dim, dim, nn_conv))

        self.bn_layers = torch.nn.ModuleList()
        for _ in range(5):
            self.bn_layers.append(BatchNorm1d(dim))

        self.dropout = Dropout(p=dropout_probability)

        self.lin_layers = torch.nn.ModuleList()
        for _ in range(2):
            self.lin_layers.append(Linear(dim, dim))

        self.lin_out = Linear(dim, num_classes)

        self.sigmoid = Sigmoid()

    def getNumClasses(self):
        return self.num_classes

    def getNumFeatures(self):
        return self.num_features

    def forward(self, x, edge_index, edge_attr, batch, edge_weight=None):
        # Transform edge_attr to consistent dimension
        edge_attr = self.edge_transform(edge_attr)

        x1 = self.relu(self.bn_layers[0](self.conv_in(x, edge_index, edge_attr)))

        x2 = self.relu(self.bn_layers[1](self.conv_layers[0](x1, edge_index, edge_attr, edge_weight))) + x1
        x3 = self.relu(self.bn_layers[2](self.conv_layers[1](x2, edge_index, edge_attr, edge_weight))) + x2
        x4 = self.relu(self.bn_layers[3](self.conv_layers[2](x3, edge_index, edge_attr, edge_weight))) + x3
        x5 = self.relu(self.bn_layers[4](self.conv_layers[3](x4, edge_index, edge_attr, edge_weight))) + x4

        x = global_max_pool(x5, batch)

        x = self.relu(self.lin_layers[0](x))
        x = self.dropout(x)
        x = self.relu(self.lin_layers[1](x))
        x = self.dropout(x)
        x = self.lin_out(x)

        x = torch.sigmoid(x)

        return x
        
