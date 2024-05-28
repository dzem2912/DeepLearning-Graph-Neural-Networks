from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt


ATOM_MAP = ['C', 'O', 'Cl', 'H', 'N', 'F',
                'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']


def visualize_molecule(graph, edge_mask=None, draw_edge_labels=None):
    """
    This method takes the graph which was previously converted into a molecule graph and 
    is further processed to look like a molecule e.g., making it planar, such that the edges do not intersect.
    Furthermore, it generates the connections between the nodes described by the input graph and adds the necessary colors.
    Finally, the plt.show() method call is done outside of the function call for easier visualization of the
    saliency map visualization.
    """
    graph = graph.copy().to_undirected()
    node_labels = {}
    for node, node_attributes in graph.nodes(data=True):
        node_labels[node] = node_attributes['element']
    position = nx.planar_layout(graph)
    position = nx.spring_layout(graph, pos=position)

    if edge_mask is None:
        edge_color = 'black'
        widths = None
    
    else:
        edge_color = [edge_mask[(edge1, edge2)] for edge1, edge2 in graph.edges()]
        widths = [x * 10 for x in edge_color]
    
    nx.draw(graph, pos=position, labels=node_labels, width=widths,
            edge_color=edge_color, edge_cmap=plt.cm.YlGnBu,
            node_color='white')

    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}
        nx.draw_networkx_edge_labels(graph, position, 
                                     edge_labels=edge_labels, font_color='red')

    # Call plt.show() outside of the method call, would be nice for explainbility :)

def to_molecule(graph_data):
    """
    Converts a graph dataset sample into a networkx suitable for visualization as a molecule.
    Nodes get exchanged by their respective elements.
    """
    graph = to_networkx(graph_data, node_attrs=['x'], edge_attrs=['edge_attr'])

    for _, node_attributes in graph.nodes(data=True):
        node_attributes['element'] = ATOM_MAP[node_attributes['x'].index(0.0)]
        #del graph_data['x']
    return graph