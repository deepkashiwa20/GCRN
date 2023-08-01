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
    
class GCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers):
        super(GCRNN_Encoder, self).__init__()
        assert rnn_layers >= 1, 'At least one GCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.gcrnn_cells = nn.ModuleList()
        self.gcrnn_cells.append(GCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, rnn_layers):
            self.gcrnn_cells.append(GCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, support):
        #shape of x: (B, T, N, D), shape of init_state: (rnn_layers, B, N, hidden_dim)
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
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (rnn_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        # return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.rnn_layers):
            init_states.append(self.gcrnn_cells[i].init_hidden_state(batch_size))
        return init_states

class GCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers):
        super(GCRNN_Decoder, self).__init__()
        assert rnn_layers >= 1, 'At least one GCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.gcrnn_cells = nn.ModuleList()
        self.gcrnn_cells.append(GCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, rnn_layers):
            self.gcrnn_cells.append(GCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, support):
        # xt: (B, N, D)
        # init_state: (rnn_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.rnn_layers):
            state = self.gcrnn_cells[i](current_inputs, init_state[i], support)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class GCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, rnn_layers=1, embed_dim=8, cheb_k=3, ycov_dim=1, cl_decay_steps=2000, use_curriculum_learning=True):
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
        self.ycov_dim = ycov_dim
        self.decoder_dim = self.rnn_units
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        
        # graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        
        # encoder
        self.encoder = GCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.rnn_layers)
        
        # deocoder
        self.decoder = GCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.rnn_layers)
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim))
    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
    
    def forward(self, x, y_cov, labels=None, batches_seen=None):
        support = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, support) # B, T, N, hidden      
        h_t = h_en[:, -1, :, :]   # B, N, hidden (last state)        
        ht_list = [h_t]*self.rnn_layers
            
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, support)
            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)
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
    # parser.add_argument('--rnn_layers', type=int, default=1, help='number of rnn layers')
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    model = GCRN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, 
                 horizon=args.horizon, rnn_units=args.rnn_units).to(device)
    summary(model, [(args.seq_len, args.num_nodes, args.input_dim), (args.horizon, args.num_nodes, args.output_dim)], device=device)
    print_params(model)
        
if __name__ == '__main__':
    main()
