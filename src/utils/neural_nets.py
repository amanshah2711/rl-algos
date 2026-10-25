
import torch.nn as nn

def create_mlp(input_dim, hidden_layer_dim, output_dim, num_hidden_layers):

    layers = []
    layer_input_dim = input_dim
    for _ in range(num_hidden_layers):
        hidden_layer = nn.Linear(layer_input_dim, hidden_layer_dim)
        layers.append(hidden_layer)
        layers.append(nn.ReLU()) #TODO make activations customizable for now relu default
        layer_input_dim = hidden_layer_dim

    final_layer = nn.Linear(layer_input_dim, output_dim)
    layers.append(final_layer)
    return nn.Sequential(*layers)
