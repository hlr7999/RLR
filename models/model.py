import math
import torch
import torch.nn as nn
from .embedding import Embedding2D


class Mlp(nn.Module):
    def __init__(self, 
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(
            self,
            args,
            n_heads,
            d_model,
            d_feed_foward=None,
            dropout=0.1,
            seq_len=0,
            ref_len=1
        ):
        super(EncoderLayer, self).__init__()

        d_feed_foward = d_feed_foward or 4 * d_model

        self.msa, self.mca, self.ca, self.nsa, self.sa  = None, None, None, None, None

        if 'mca' in args.blocks:
            self.mca = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        if 'ca' in args.blocks:
            self.ca = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        if 'msa' in args.blocks:
            self.msa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        if 'nsa' in args.blocks:
            self.nsa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        if 'sa' in args.blocks:
            self.sa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = Mlp(d_model, d_feed_foward, drop=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.seq_len = seq_len
        self.ref_len = ref_len
        self.block_mode = args.block_mode
        if self.block_mode == 'serial':
            self.norm0 = nn.LayerNorm(d_model)

    def forward(self, x, ref_mca, gate):
        if self.block_mode == 'serial':
            pass
        else:
            new_x = torch.zeros_like(x)
            gi = 0
            if self.mca is not None:
                mask = self.generate_mask(reverse=True)
                mca_x = torch.zeros_like(x)
                for ref_mca in ref_mca.split(self.seq_len, dim=1): # refl x (N, L, D)
                    mca_x = mca_x + self.mca(
                        x, ref_mca, ref_mca, attn_mask=mask
                    )[0]
                mca_x = self.dropout(mca_x)
                if gate is not None:
                    g = gate[:, :, gi].unsqueeze(-1)
                    gi += 1
                else:
                    g = 2
                new_x = new_x + g * mca_x
            if self.ca is not None:
                ca_x = self.ca(
                    x, ref_mca, ref_mca
                )[0]
                ca_x = self.dropout(ca_x)
                if gate is not None:
                    g = gate[:, :, gi].unsqueeze(-1)
                    gi += 1
                else:
                    g = 2
                new_x = new_x + g * ca_x
            if self.msa is not None:
                mask = self.generate_mask()
                msa_x = self.msa(
                    x, x, x, attn_mask=mask
                )[0]
                msa_x = self.dropout(msa_x)
                if gate is not None:
                    g = gate[:, :, gi].unsqueeze(-1)
                    gi += 1
                else:
                    g = 1
                new_x = new_x + g * msa_x
            if self.nsa is not None:
                mask = self.generate_mask()
                nsa_x = self.nsa(
                    x, ref_mca, x, attn_mask=mask
                )[0]
                nsa_x = self.dropout(nsa_x)
                if gate is not None:
                    g = gate[:, :, gi].unsqueeze(-1)
                    gi += 1
                else:
                    g = 1
                new_x = new_x + g * nsa_x
            if self.sa is not None:
                sa_x = self.sa(
                    x, x, x
                )[0]
                sa_x = self.dropout(sa_x)
                if gate is not None:
                    g = gate[:, :, gi].unsqueeze(-1)
                    gi += 1
                else:
                    g = 1
                new_x = new_x + g * sa_x

            x = self.norm1(new_x)
            y = self.ffn(x)
            out = self.norm2(x + y)

        return out

    def generate_mask(self, reverse=False):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h = w = int(math.sqrt(self.seq_len))
        if h == 16:
            hm, wm = 5, 5
        elif h == 32:
            hm, wm = 7, 7
        elif h == 64:
            hm, wm = 11, 11
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if reverse:
            mask = (
                mask.float()
                .masked_fill(mask == 1, float("-inf"))
                .cuda()
            )
        else:
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
                .cuda()
            )
        return mask


class Encoder(nn.Module):
    def __init__(self, encode_layers, gate_layer=None, norm_layer=None):
        super(Encoder, self).__init__()
        
        self.encode_layers = nn.ModuleList(encode_layers)
        self.norm = norm_layer
        self.gate = gate_layer

    def forward(self, x, ref_mca):
        for i, layer in enumerate(self.encode_layers):
            if self.gate is not None:
                if isinstance(self.gate, nn.Parameter):
                    g = self.gate[i].unsqueeze(0).unsqueeze(0).repeat(x.shape[0], 1, 1) # (N, 1, K)
                else:
                    gx = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=-1)
                    g = self.gate(gx) # (N, 1, K)
            else:
                g = None
            x = layer(x, ref_mca, g)

        if self.norm is not None:
            x = self.norm(x)

        return x


class RLR(nn.Module):
    def __init__(
            self,
            seq_len,
            in_channels,
            out_channels,
            d_model=512,
            n_heads=4,
            d_feed_foward_scale=4,
            dropout=0.0,
            args=None
        ):
        super(RLR, self).__init__()
        
        d_feed_foward = d_model * d_feed_foward_scale
        n_layers = args.layers

        # embedding
        h = w = int(math.sqrt(seq_len))
        self.embedding = Embedding2D(in_channels, d_model, dropout, h=h, w=w)

        # reference
        self.ref_mca = None
        self.ref_len = args.ref_len
        if 'mca' in args.blocks or 'nsa' in args.blocks or 'ca' in args.blocks:
            self.ref_mca = nn.Parameter(torch.randn(1, seq_len * self.ref_len, in_channels))
            self.ref_mca_embedding = Embedding2D(in_channels, d_model, dropout, h=h, w=w, ref_len=self.ref_len)
        
        # gate
        if args.blocks_gate == 'net':
            gate_net = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, len(args.blocks)),
                nn.Softmax(dim=-1)
            )
        elif args.blocks_gate == 'gate':
            gate_net = nn.Parameter(torch.ones(n_layers, len(args.blocks)))
        else:
            gate_net = None
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    args,
                    n_heads,
                    d_model,
                    d_feed_foward,
                    dropout=dropout,
                    seq_len=seq_len,
                ) for _ in range(n_layers)
            ],
            gate_layer=gate_net,
            norm_layer=nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, out_channels, bias=True)

    def forward(self, x):
        # (N, L, dim) -> (N, L, d_model)
        emb = self.embedding(x)
        ref_mca_emb = None
        if self.ref_mca is not None:
            ref_mca_emb = self.ref_mca_embedding(self.ref_mca.repeat([x.shape[0], 1, 1]))
        emb = self.encoder(emb, ref_mca_emb)
        # (N, L, d_model) -> (N, L, dim)
        x_out = self.projection(emb)

        return x_out
