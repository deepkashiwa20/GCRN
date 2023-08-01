import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, support_len):
        super(GCN, self).__init__()
        self.cheb_k = cheb_k
        self.support_len = support_len
        self.weights = nn.Parameter(torch.FloatTensor(support_len*cheb_k*dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, supports):
        support_set = []
        for support in supports:
            node_num = support.shape[0]
            support_set.extend([torch.eye(node_num).to(support.device), support])
            for k in range(2, self.cheb_k):
                support_set.append(torch.matmul(2 * support, support_set[-1]) - support_set[-2]) 
        
        x_g = []
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1) # B, N, support_len*cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv
    
class GCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, support_len):
        super(GCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, support_len)
        self.update = GCN(dim_in+self.hidden_dim, dim_out, cheb_k, support_len)

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
    
class DCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, support_len, num_layers):
        super(DCRNN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GCRNCell(node_num, dim_in, dim_out, cheb_k, support_len))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GCRNCell(node_num, dim_out, dim_out, cheb_k, support_len))

    def forward(self, x, init_state, support):
        #shape of x: (B, T, N, D), shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, support)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        # return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states

class DCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, support_len, num_layers):
        super(DCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GCRNCell(node_num, dim_in, dim_out, cheb_k, support_len))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GCRNCell(node_num, dim_out, dim_out, cheb_k, support_len))

    def forward(self, xt, init_state, support):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], support)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class MGCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, embed_dim=8, 
                 cheb_k=3, ycov_dim=1, adj_mx=None, adp_mx=True, cl_decay_steps=2000, use_curriculum_learning=True):
        super(MGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.cheb_k = cheb_k
        self.rnn_units = rnn_units
        self.num_layers = num_layers
        self.ycov_dim = ycov_dim
        self.decoder_dim = self.rnn_units
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        
        # graphs
        self.adp_mx = adp_mx
        self.adj_mx = adj_mx
        
        if self.adp_mx:
            self.node_embeddings1 = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
            self.node_embeddings2 = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
            self.support_len = 1
        else:
            self.support_len = 0
        if self.adj_mx is not None:
            self.support_len += len(adj_mx)
        else:
            assert False, 'When adp_mx=False, adj_mx should be given...'
            
        # encoder
        self.encoder = DCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, 
                                      self.support_len, self.num_layers)
        
        # deocoder
        self.decoder = DCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, 
                                      self.support_len, self.num_layers)
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim))
    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
    
    def forward(self, x, y_cov, labels=None, batches_seen=None):
        if self.adp_mx:
            supports = [F.softmax(F.relu(torch.mm(self.node_embeddings1, self.node_embeddings2.transpose(0, 1))), dim=1)]
        else:
            supports = []
        if self.adj_mx is not None:
            supports += self.adj_mx
        else:
            assert False, 'When adp_mx=False, adj_mx should be given...'
        
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports) # B, T, N, hidden      
        h_t = h_en[:, -1, :, :]   # B, N, hidden (last state)        
        ht_list = [h_t]*self.num_layers
            
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports)
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
    print(f'In total: {param_count} trainable parameters. \n')
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
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    model = MGCRN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, horizon=args.horizon, rnn_units=args.rnn_units).to(device)
    summary(model, [(args.seq_len, args.num_nodes, args.input_dim), (args.horizon, args.num_nodes, args.output_dim)], device=device)
    print_params(model)
        
if __name__ == '__main__':
    main()
