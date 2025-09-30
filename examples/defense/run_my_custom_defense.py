import torch
import dgl
from dgl.data import CoraGraphDataset
from pygip.models.defense.my_custom_defense import MyCustomDefense

class Cora:
    def __init__(self, graph):
        self.graph_dataset = None
        self.graph_data = graph
        self.api_type = "dgl"
        self.num_nodes = graph.num_nodes()
        self.num_features = graph.ndata['feat'].shape[1]
        self.num_classes = len(torch.unique(graph.ndata['label']))
        self.graph_data.ndata['train_mask'] = graph.ndata.get('train_mask', torch.zeros(self.num_nodes, dtype=torch.bool))
        self.graph_data.ndata['val_mask'] = graph.ndata.get('val_mask', torch.zeros(self.num_nodes, dtype=torch.bool))
        self.graph_data.ndata['test_mask'] = graph.ndata.get('test_mask', torch.zeros(self.num_nodes, dtype=torch.bool))
        if self.graph_data.ndata['train_mask'].sum() == 0:
            self.graph_data.ndata['train_mask'][:int(0.6 * self.num_nodes)] = True
        if self.graph_data.ndata['val_mask'].sum() == 0:
            self.graph_data.ndata['val_mask'][int(0.6 * self.num_nodes):int(0.8 * self.num_nodes)] = True
        if self.graph_data.ndata['test_mask'].sum() == 0:
            self.graph_data.ndata['test_mask'][int(0.8 * self.num_nodes):] = True

def main():
    dataset = Cora(CoraGraphDataset()[0])
    defense = MyCustomDefense(dataset, attack_node_fraction=0.3, hidden_dim=64, epochs=50, lr=0.01)
    results = defense.defend()
    print("Results:", results)

if __name__ == "__main__":
    main()
