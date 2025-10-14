from torch import nn

class mlp_nn(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.0):
        super().__init__()
        layers = list()
        hidden_layer_num = len(hidden_layers)

        for i in range(hidden_layer_num):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_layers[i]))
            else:
                layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            
            layers.append(nn.ReLU())
            
            if (i+1) % 2 == 0 and i != (hidden_layer_num - 1) and dropout != 0.0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_layers[-1], 2))
        self.predict_layers = nn.Sequential(*layers)


    def forward(self, input):
        output = self.predict_layers(input)
        return output