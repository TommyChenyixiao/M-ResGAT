import torch
import torch.nn.functional as F

def get_split_idx(num_nodes, train_ratio=0.6, val_ratio=0.2):
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    return train_idx, val_idx, test_idx

def train_model(model, data, optimizer, train_idx):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()