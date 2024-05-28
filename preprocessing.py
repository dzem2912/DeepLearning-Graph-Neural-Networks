import matplotlib.pyplot as plt
import random
from torch_geometric.datasets import TUDataset
from utils import visualize_molecule, to_molecule


def check_dataset(dataset: TUDataset):
    """
    This function takes as an argument the dataset. 
    Draws a random graph from the dataset and converts it to a molecule-looking graph.
    The Molecule is visualized with the function,
    and then some properties of the randomly drawn graph are printed in the terminal.
    """
    data = random.choice([t for t in dataset])
    mol = to_molecule(data)
    plt.figure(figsize=(10, 5))
    visualize_molecule(mol)
    
    print(f'Number of Nodes: {data.num_nodes}')
    print(f'Number of Edges: {data.num_edges}')
    print(f'Number of Node Features: {data.num_node_features}')
    print(f'Number of Edge Features: {data.num_edge_features if data.edge_attr is not None else 0}')
    print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
    print(f'Contains self-loops: {data.has_self_loops()}')
    print(f'Is directed: {data.is_directed()}')

    print(f'Node features (x): {data.x}')
    print(f'Edge indices (edge_index): {data.edge_index}')
    print(f'Graph label (y): {data.y}')

    if data.edge_attr is not None:
        print(f'Edge features (edge_attr): {data.edge_attr}')

    if hasattr(data, 'train_mask'):
        print(f'Training mask: {data.train_mask}')

    for key, item in data:
        if key not in ['edge_index', 'x', 'y', 'edge_attr', 'train_mask']:
            print(f'{key}: {item}')