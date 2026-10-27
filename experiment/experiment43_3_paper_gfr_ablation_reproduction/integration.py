"""Minimal isolated A constructor-compatible PEN bypass and physical B PLN.
Functions ported once from audited E43.2; no runtime dependency on history.
"""
from pathlib import Path
import ast,difflib
from collections import Counter
INPUT_DIMS=[48,6,10,256]

def make_pln_without_pen(torch, layers, dims, hp):
    import torch.nn as nn

    class PLNNoPEN(nn.Module):
        def __init__(self):
            super().__init__()
            self.hyperpm = hp
            self.input_data_dims = dims
            self.d_q = self.d_k = self.d_v = hp.n_hidden
            self.n_head = hp.n_head
            self.dropout = hp.dropout
            self.n_layer = hp.nlayer
            self.modal_num = hp.nmodal
            self.n_class = hp.nclass
            self.d_out = self.d_v * self.n_head * self.modal_num
            self.fusion_weight = nn.Parameter(torch.ones([6, 1]))
            self.InputLayer = layers.VariLengthInputLayer(dims, self.d_k, self.d_v, self.n_head, self.dropout)
            self.Encoder = []
            self.FeedForward = []
            for index in range(self.n_layer):
                encoder = layers.EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v,
                                             self.n_head, self.dropout)
                self.add_module("encode_%d" % index, encoder)
                self.Encoder.append(encoder)
                feedforward = layers.FeedForwardLayer(self.d_v * self.n_head,
                                                       self.d_v * self.n_head, dropout=self.dropout)
                self.add_module("feed_%d" % index, feedforward)
                self.FeedForward.append(feedforward)
            self.Outputlayer = layers.OutputLayer(self.d_v * self.n_head * self.modal_num,
                                                   self.d_v * self.n_head, self.n_class,
                                                   self.modal_num, self.dropout)

        def forward(self, x):
            bs = x.size(0)
            attn_map = []
            x, input_attention = self.InputLayer(x)
            attn = input_attention.mean(dim=1)
            attn_map.append(attn.detach().cpu().numpy())
            for index in range(self.n_layer):
                x, layer_attention = self.Encoder[index](q=x, k=x, v=x, modal_num=self.modal_num)
                attn = layer_attention.mean(dim=1)
                x = self.FeedForward[index](x)
                attn_map.append(attn.detach().cpu().numpy())
            output, hidden = self.Outputlayer(x.view(bs, -1), attn.view(bs, -1))
            return output, hidden, attn_map

    module = PLNNoPEN()
    assert not hasattr(module, "MLP")
    assert not hasattr(module, "prompt_text_features")
    assert not hasattr(module, "avg_pool")
    return module

def make_agent(torch, np, layers, network, utils, hp, x, y, train, test, base):
    from torch import optim

    class Agent:
        pass

    agent = Agent()
    agent.dev = torch.device("cuda")
    agent.hyperpm = hp
    agent.th, agent.GC_mode = hp.th, hp.GC_mode
    agent.MP_mode, agent.MF_mode = hp.MP_mode, hp.MF_mode
    agent.d_v, agent.modal_num, agent.n_class = hp.n_hidden, hp.nmodal, hp.nclass
    agent.dropout, agent.alpha, agent.n_head = hp.dropout, hp.alpha, hp.n_head
    agent.feat = torch.from_numpy(x).float().to(agent.dev)
    agent.targ = torch.from_numpy(y).long().to(agent.dev)
    agent.trn_idx = np.asarray(train)
    agent.tst_idx = agent.val_idx = np.asarray(test)
    trn_label = agent.targ[agent.trn_idx].cpu().numpy()
    counter = Counter(trn_label)
    weights = len(trn_label) / np.asarray(list(counter.values())) / agent.n_class
    agent.weight = torch.from_numpy(weights).float().to(agent.dev)
    agent.dataLayer = layers.InitialDataLayer(INPUT_DIMS)
    agent.MessagePassing0 = network.GFR(INPUT_DIMS, INPUT_DIMS, th=agent.th, mode=agent.GC_mode).to(agent.dev)
    agent.ModalFusion = (network.PLN(INPUT_DIMS,hp) if base=="A" else make_pln_without_pen(torch,layers,INPUT_DIMS,hp)).to(agent.dev)
    assert (hasattr(agent.ModalFusion,"MLP")) == (base=="A")
    agent.optimizer_MP0 = optim.Adam(agent.MessagePassing0.parameters(), lr=hp.lr, weight_decay=hp.reg)
    agent.optimizer_MF = optim.Adam(agent.ModalFusion.parameters(), lr=hp.lr, weight_decay=hp.reg)
    agent.ModalFusion.apply(utils.my_weight_init)
    opt_ids = [id(p) for opt in (agent.optimizer_MP0, agent.optimizer_MF)
               for group in opt.param_groups for p in group["params"]]
    model_ids = [id(p) for module in (agent.MessagePassing0, agent.ModalFusion) for p in module.parameters()]
    assert len(opt_ids) == len(set(opt_ids)) and set(opt_ids) == set(model_ids)
    assert np.array_equal(agent.val_idx, agent.tst_idx)
    return agent

def prepare(root,exp):
    root,exp=Path(root),Path(exp);out=exp/'.runtime/sources';diff=[]
    for name in ['network.py','layers.py','utils.py']:
        orig=(root/name).read_text();new=orig
        if name=='network.py':
            a=new.index('from CLIP.clip import clip\n');b=new.index('class MLP',a)
            new=new[:a]+'device = "cuda" if torch.cuda.is_available() else "cpu"\n_prompt_text_features = None\n_legacy_loader = None\n\ndef get_prompt_text_features():\n    global _prompt_text_features\n    if _prompt_text_features is None:\n        assert _legacy_loader is not None\n        _prompt_text_features = _legacy_loader()\n    return _prompt_text_features\n\n'+new[b:]
            a=new.index('        text_features = self.prompt_text_features\n');b=new.index('        x = x.view(bs, -1)\n',a)
            new=new[:a]+new[b:]
            ast.parse(new)
        if (out/name).exists():assert (out/name).read_text()==new
        else:(out/name).write_text(new)
        diff.extend(difflib.unified_diff(orig.splitlines(True),new.splitlines(True),fromfile='user/'+name,tofile='isolated/'+name))
    (exp/'integration.patch').write_text(''.join(diff))
