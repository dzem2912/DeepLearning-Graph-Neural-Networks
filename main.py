import torch
import random
import numpy as np
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from gnn import GNN1, GNN2
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
from gnn import simpleGNN, GraphNeuralNetwork
from utils import to_molecule, visualize_molecule
from preprocessing import check_dataset
from explanations import explain, aggregate_edge_directions
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
from torchinfo import summary
import seaborn as sns


np.random.seed(42)
torch.manual_seed(42)

def test_model(name: str, lr:float, dropout_prob: float, train_loader:DataLoader, 
               val_loader: DataLoader, test_loader: DataLoader, num_features: int, num_classes: int):
    """
    This function does the evaluation of the model defined by the 'name' argument.
    It implements the train and test function in the pyTorch way.
    """
    if name == 'GNN1':
        model = GNN1(dim=32, num_features=num_features, num_classes=num_classes, dropout_probability=dropout_prob)
        print(summary(model=model))
    elif name == 'GNN2':
        model = GNN2(dim=1024, num_features=num_features, num_classes=num_classes, dropout_probability=dropout_prob)
        print(summary(model))
    elif name == 'simple':
        model = simpleGNN(dim=32, num_features=num_features, num_classes=num_classes, dropout_probability=dropout_prob)
        print(summary(model))
    elif name == 'GraphNeuralNetwork':
        model = GraphNeuralNetwork(dim=256, num_features=num_features, num_classes=num_classes, edge_attr_dim=3, dropout_probability=dropout_prob).cuda()

    optimizer: Adam = Adam(model.parameters(), lr=lr)

    if torch.cuda.is_available():
        device: str = 'cuda:0'
    else:
        device: str = 'cpu'

    def train(epoch: int):
        """
        This method defines the training step and is being called for each epoch. The model is doing a forward pass, computing the
        Binary cross entropy loss and then backpropagating to adjust the weights such that the loss is minimized.
        """
        print(f"Epoch: {epoch}/{100}")
        model.train()

        total_train_loss: float = 0.0
        total_val_loss: float = 0.0

        for data in train_loader:
            optimizer.zero_grad() # Sets the gradients to 0

            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            target = F.one_hot(data.y, num_classes=num_classes).float().cuda()
            
            loss_fn = torch.nn.BCELoss().cuda()
            loss = loss_fn(output, target)
            loss.backward()

            total_train_loss += loss.item() * data.num_graphs
            optimizer.step()

        # Validation loop 
        with torch.no_grad(): # disable gradient computation
            for data in val_loader:
                data = data.to(device)
                output = model(data.x, data.edge_index, data.edge_attr, data.batch)
                output = torch.Tensor(output).cuda()

                target = F.one_hot(data.y, num_classes=num_classes).float().cuda()
                
                loss_fn = torch.nn.BCELoss().cuda()
                val_loss = loss_fn(output, target)

                total_val_loss += val_loss.item() * data.num_graphs
        
        return total_train_loss / len(train_loader.dataset), total_val_loss / len(val_loader.dataset)
    
    def test(test_loader: DataLoader):
        """
        This function does a forward pass on the test dataset and compares the predicted labels with the true labels.
        It computes the classification metrics such as the balanced accuracy score, f1 score and the confusion matrix.
        """
        model.eval()

        correct: int = 0
        all_predictions = []
        all_lables = []
        
        for data in test_loader:
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.edge_attr, data.batch)
            predicted_class = []
            for output in outputs:
                if output[0] >= 0.5:
                     predicted_class.append(0)
                else:
                     predicted_class.append(1)
            
            predicted_class = torch.tensor(predicted_class).cpu()
            true_class = data.y.cpu()
            correct += predicted_class.eq(true_class).sum().item()

            all_predictions.extend(predicted_class.cpu().numpy())
            all_lables.extend(data.y.cpu().numpy())

        accuracy = balanced_accuracy_score(all_lables, all_predictions)
        f1 = f1_score(all_lables, all_predictions, average='binary')
        cm = confusion_matrix(all_lables, all_predictions)
        
        return accuracy, f1, cm
    
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0

    train_losses = []
    val_losses = []
    num_epochs: int = 100
    for epoch in range(1, num_epochs + 1):
        train_loss, val_loss = train(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    train_accuracy, _, _ = test(train_loader)
    val_accuracy, _, _ = test(val_loader)

    print(f"Train Accuracy: {(train_accuracy * 100):.4f}%")
    print(f"Validation Accuracy: {(val_accuracy * 100):.4f}%")

    test_accuracy, test_f1, test_cm = test(test_loader)
    print(f"Model\nName: {name}\n lr:{lr} \n dp: {dropout_prob}\n") 
    print(f"Test accuracy: {(test_accuracy * 100):.4f}%")
    print(f"Test F1 Score: {test_f1:.2f}")
    print(f"Test CM: \n {test_cm}")

    # Plotting using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    return model

if __name__ == "__main__":
    path = '.'
    dataset: TUDataset = TUDataset(path, name='Mutagenicity').shuffle()
    print(f"lenght of dataset: {len(dataset)}")
    # Get some general information about the dataset
    
    #check_dataset(dataset=dataset)

    import networkx as nx

    #data = random.choice([x for x in dataset if not x.y.item()])
    data = dataset[50]
    #print(data)
    #molecule_graph = to_molecule(data)
    #print("Molecule element:\n", molecule_graph)
    #visualize_molecule(molecule_graph)
    #plt.show()
    #exit(0)
    
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.25, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.25, random_state=42)

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=16)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=16)

    print(f"length of train: {len(train_loader.dataset)}")
    print(f"length of val: {len(val_loader.dataset)}")
    print(f"length of test: {len(test_loader.dataset)}")

    lr: float = 0.0001
    dp: float = 0.3

    model = test_model("GraphNeuralNetwork", lr, dp, train_loader, val_loader, test_loader, dataset.num_features, dataset.num_classes)

    # Explanations
    torch.save(model, "GNN_MK2.pth")
    
    data = random.choice([t for t in test_dataset])
    edge_mask = explain(data=data, target=0)
    edge_mask_dict = aggregate_edge_directions(edge_mask, data=data)
    plt.figure(figsize=(10, 5))
    plt.title('Salienicy')

    molecule = to_molecule(data)
    visualize_molecule(molecule, edge_mask_dict)
    visualize_molecule(molecule, None)
    plt.show()
    
    