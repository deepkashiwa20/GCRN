import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(GCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k*dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, support):
        node_num = support.shape[0]
        support_set = [torch.eye(node_num).to(support.device), support]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * support, support_set[-1]) - support_set[-2]) 
        
        x_g = []
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1) # B, N, cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv
    
class GCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(GCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k)
        self.update = GCN(dim_in+self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, support):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, support))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, support))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
class GCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers):
        super(GCRNN, self).__init__()
        assert rnn_layers >= 1, 'At least one GCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.gcrnn_cells = nn.ModuleList()
        self.gcrnn_cells.append(GCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, rnn_layers):
            self.gcrnn_cells.append(GCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, support):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.rnn_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.gcrnn_cells[i](current_inputs[:, t, :, :], state, support)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.rnn_layers):
            init_states.append(self.gcrnn_cells[i].init_hidden_state(batch_size))
        return init_states


class GCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, rnn_layers, embed_dim, cheb_k):
        super(GCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.rnn_layers = rnn_layers
        self.embed_dim = embed_dim
        self.cheb_k = cheb_k
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.decoder_dim = self.rnn_units
        
        # graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        
        # encoder
        self.encoder = GCRNN(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.rnn_layers)
        
        # deocoder
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.rnn_units), bias=True)
    
    def forward(self, x):
        support = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, _ = self.encoder(x, init_state, support) # B, T, N, hidden      
        h_t = h_en[:, -1:, :, :]   # B, 1, N, hidden
        output = self.end_conv(h_t)  # (B, T, N, 1)
        return output

def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'In total: {param_count} trainable parameters.')
    return

def main():
    import sys
    import argparse
    from torchsummary import summary
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3, help="which GPU to use")
    parser.add_argument('--num_nodes', type=int, default=207, help='number of variables (e.g., 207 in METR-LA, 325 in PEMS-BAY)')
    parser.add_argument('--seq_len', type=int, default=12, help='sequence length of historical observation')
    parser.add_argument('--horizon', type=int, default=12, help='sequence length of prediction')
    parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
    parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
    parser.add_argument('--rnn_units', type=int, default=64, help='number of hidden units')
    parser.add_argument('--rnn_layers', type=int, default=2, help='number of rnn layers')
    parser.add_argument('--embed_dim', type=int, default=8, help='embedding dimension for adaptive graph')
    parser.add_argument('--cheb_k', type=int, default=2, help='max diffusion step or Cheb K')
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    model = GCRN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, 
                 horizon=args.horizon, rnn_units=args.rnn_units, rnn_layers=args.rnn_layers,
                 embed_dim=args.embed_dim, cheb_k=args.cheb_k).to(device)
    summary(model, (args.seq_len, args.num_nodes, args.input_dim), device=device)
    print_params(model)
        
if __name__ == '__main__':
    main()
