from captum.attr import Saliency
import torch
import numpy as np
from collections import defaultdict

def single_graph_forward_pass(edge_mask, data):
    """
    This function loads the trained model and computes a forward pass on a graph.
    This function is used only for explainability with captum Saliency.

    """
    print(data)
    model = torch.load("model.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #batch = torch.zeros(data.x.shape[0], dtype=int).to(device)
    batch = torch.zeros(data.x.shape[0], dtype=int).to(device)

    return model(data.x, data.edge_index, data.edge_attr, batch, edge_mask)


def explain(data, target=0):
    """
    This method will be used together with the captum's Saliency method to generate an edge mask, 
    which is essentially a numpy array containing the importance value of each edge of the graph. 
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    
    saliency = Saliency(single_graph_forward_pass) # single forward pass of the model
    mask = saliency.attribute(input_mask, target=target, additional_forward_args=(data, ))
    
    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0: # to avoid division with 0
        edge_mask /= edge_mask.max()

    return edge_mask


def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for value, edge1, edge2 in list(zip(edge_mask, *data.edge_index)):
        edge1, edge2 = edge1.item(), edge2.item()

        if edge1 > edge2:
            edge1, edge2 = edge2, edge1
        edge_mask_dict[(edge1, edge2)] += value

    return edge_mask_dict
