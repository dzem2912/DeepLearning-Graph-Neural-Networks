from captum.attr import Saliency
import torch
import numpy as np
from collections import defaultdict

def single_graph_forward_pass(input_mask=None, data=None):
    """
    This function loads the trained model and computes a forward pass on a graph.
    This function is used only for explainability with captum Saliency.
    """
    model = torch.load("model_gnn.pth")
    model.eval()  # Ensure the model is in evaluation mode

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(data)
    batch = torch.zeros(data.x.shape[0], dtype=int).to(device)

    data = data.to(device)  # Ensure the data is on the correct device

    if input_mask is not None:
        # Apply the mask if provided
        edge_attr = data.edge_attr * input_mask.view(-1, 1)
    else:
        edge_attr = data.edge_attr

    return model(data.x, data.edge_index, edge_attr, batch)

def explain(data, target=0):
    """
    This method will be used together with the captum's Saliency method to generate an edge mask,
    which is essentially a numpy array containing the importance value of each edge of the graph.
    """
    print(f"Explain: {data}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)  # Ensure the data is on the correct device
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)

    saliency = Saliency(single_graph_forward_pass)  # single forward pass of the model
    mask = saliency.attribute(input_mask, target=target, additional_forward_args=(data,))

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:  # to avoid division with 0
        edge_mask /= edge_mask.max()

    return edge_mask

# mask = saliency.attribute(inputs=input_mask, target=data.y)
    
def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for value, edge1, edge2 in list(zip(edge_mask, *data.edge_index)):
        edge1, edge2 = edge1.item(), edge2.item()

        if edge1 > edge2:
            edge1, edge2 = edge2, edge1
        edge_mask_dict[(edge1, edge2)] += value

    return edge_mask_dict
