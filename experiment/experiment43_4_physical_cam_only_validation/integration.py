"""Experiment 43.4 isolated physically instantiated CAM-only model."""
from pathlib import Path
import torch
import torch.nn as nn

INPUT_DIMS=[48,6,10,256]

class PhysicalCAMOnly(nn.Module):
    """Original CAM/PLN path with every GFR and PEN object absent."""
    def __init__(self, layers, dims, hp):
        super().__init__()
        self.hyperpm=hp
        self.input_data_dims=list(dims)
        self.d_q=self.d_k=self.d_v=hp.n_hidden
        self.n_head=hp.n_head
        self.dropout=hp.dropout
        self.n_layer=hp.nlayer
        self.modal_num=hp.nmodal
        self.n_class=hp.nclass
        self.d_out=self.d_v*self.n_head*self.modal_num
        # Retained by the original CAM/PLN registration and unused by forward.
        self.fusion_weight=nn.Parameter(torch.ones([6,1]))
        self.InputLayer=layers.VariLengthInputLayer(dims,self.d_k,self.d_v,self.n_head,self.dropout)
        self.Encoder=[]; self.FeedForward=[]
        for index in range(self.n_layer):
            enc=layers.EncodeLayer(self.d_k*self.n_head,self.d_k,self.d_v,self.n_head,self.dropout)
            self.add_module(f"encode_{index}",enc); self.Encoder.append(enc)
            ff=layers.FeedForwardLayer(self.d_v*self.n_head,self.d_v*self.n_head,dropout=self.dropout)
            self.add_module(f"feed_{index}",ff); self.FeedForward.append(ff)
        self.Outputlayer=layers.OutputLayer(self.d_v*self.n_head*self.modal_num,self.d_v*self.n_head,self.n_class,self.modal_num,self.dropout)
    def forward(self,x):
        bs=x.size(0); x,attn0=self.InputLayer(x); attn=attn0.mean(dim=1)
        attn_map=[attn.detach().cpu().numpy()]
        for index in range(self.n_layer):
            x,attn_layer=self.Encoder[index](q=x,k=x,v=x,modal_num=self.modal_num)
            attn=attn_layer.mean(dim=1); x=self.FeedForward[index](x)
            attn_map.append(attn.detach().cpu().numpy())
        out,hidden=self.Outputlayer(x.view(bs,-1),attn.view(bs,-1))
        return out,hidden,attn_map

def build(layers,hp):
    m=PhysicalCAMOnly(layers,INPUT_DIMS,hp)
    forbidden=('gfr','graphlearn','gcn','messagepassing','pen','spgs','res','clip','avg_pool','prompt')
    names=[n.lower() for n,_ in m.named_parameters()]
    keys=[n.lower() for n in m.state_dict()]
    assert sum(p.numel() for p in m.parameters())==49544
    assert not any(any(x in n for x in forbidden) for n in names+keys)
    return m
