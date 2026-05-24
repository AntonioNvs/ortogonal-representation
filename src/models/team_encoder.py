import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

class TeamGraphEncoder(nn.Module):
    def __init__(self, num_nodes_dict, hidden_dim=32, out_dim=8):
        super(TeamGraphEncoder, self).__init__()
        
        self.node_emb = nn.ModuleDict()
        for node_type, num_nodes in num_nodes_dict.items():
            self.node_emb[node_type] = nn.Embedding(num_nodes, hidden_dim)
        
        self.conv1 = HeteroConv({
            ('results', 'f2p_constructorId', 'constructors'): SAGEConv((-1, -1), hidden_dim),
            ('qualifying', 'f2p_constructorId', 'constructors'): SAGEConv((-1, -1), hidden_dim),
            ('constructor_standings', 'f2p_constructorId', 'constructors'): SAGEConv((-1, -1), hidden_dim)
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('results', 'f2p_constructorId', 'constructors'): SAGEConv((-1, -1), out_dim),
            ('qualifying', 'f2p_constructorId', 'constructors'): SAGEConv((-1, -1), out_dim),
            ('constructor_standings', 'f2p_constructorId', 'constructors'): SAGEConv((-1, -1), out_dim)
        }, aggr='max')
        
        self.ln = nn.LayerNorm(out_dim)
        
    def forward(self, x_dict, edge_index_dict, target_constructor_ids):
        h_dict = {}
        for node_type, x in x_dict.items():
            device = x.device
            num_nodes = self.node_emb[node_type].num_embeddings
            idx = torch.arange(num_nodes, device=device)
            h_dict[node_type] = self.node_emb[node_type](idx)
            
        out_dict = self.conv1(h_dict, edge_index_dict)
        out_dict = {key: x.relu() for key, x in out_dict.items()}
        
        for key, h in h_dict.items():
            if key not in out_dict:
                out_dict[key] = h
        
        out_dict = self.conv2(out_dict, edge_index_dict)
        
        all_constructor_embeddings = out_dict['constructors']
        v_equipe = all_constructor_embeddings[target_constructor_ids]
        
        return self.ln(v_equipe)