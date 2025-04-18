import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class HyperLayers(nn.Module):
    def __init__(self, input_dim, hypernet_embed, n_agents, embed_dim):
        super(HyperLayers, self).__init__()
        self.w1 = nn.Sequential(
            nn.Linear(input_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, n_agents * embed_dim)
        )
        self.b1 = nn.Sequential(
            nn.Linear(input_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, embed_dim)
        )
        self.w2 = nn.Sequential(
            nn.Linear(input_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, embed_dim)
        )
        self.b2 = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        return x

class GroupMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(GroupMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_groups = 3
        self.embed_dim = args.mixing_embed_dim
        self.hypernet_embed = args.hypernet_embed
        self.state_ally_feats_size = args.state_ally_feats_size
        self.state_enemy_feats_size = args.state_enemy_feats_size
        self.input_dim = self.state_dim = int(np.prod(args.state_shape)) 

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        assert self.qmix_pos_func == "abs"

        self.groups = nn.ModuleList([
            HyperLayers(self.input_dim, self.hypernet_embed, self.n_agents, self.embed_dim) for _ in range(self.n_groups)
        ])
        self.hyper = HyperLayers(self.input_dim, self.hypernet_embed, self.n_groups, self.embed_dim)

        if getattr(args, "use_orthogonal", False):
            raise NotImplementedError
            for m in self.modules():
                orthogonal_init_(m)

    def get_masks(self, ally_states):
        # 只取初始时间的类型（无人死亡）
        type_matrix = ally_states[:, :, :, -3:]
        types = [th.where(type_matrix[:, 0, :, i] == 1) for i in range(self.n_groups)]
        masks = [th.zeros(ally_states.shape[0], ally_states.shape[1], self.n_agents) for _ in range(self.n_groups)]
        for i in range(self.n_groups):
            masks[i][types[i][0], :, types[i][1]] = 1
        return masks

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()

        split_indices = [self.n_agents * self.state_ally_feats_size, self.n_agents * self.state_ally_feats_size + self.n_enemies * self.state_enemy_feats_size]
        ally_states, enemy_states, last_actions = np.split(states, split_indices, axis=2)
        ally_states = ally_states.reshape(b, t, self.n_agents, self.state_ally_feats_size)
        
        masks = self.get_masks(ally_states)
        states = states.reshape(-1, self.state_dim) # b * t, state_dim

        group_qvals = []
        for i, group in enumerate(self.groups):
            w1 = group.w1(states).view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
            b1 = group.b1(states).view(-1, 1, self.embed_dim)
            w2 = group.w2(states).view(-1, self.embed_dim, 1) # b * t, emb, 1
            b2 = group.b2(states).view(-1, 1, 1)
            if self.abs:
                w1 = self.pos_func(w1)
                w2 = self.pos_func(w2)
            masked_qvals = (qvals * masks[i].to(qvals.device)).reshape(b * t, 1, self.n_agents)
            hidden = F.elu(th.matmul(masked_qvals, w1) + b1) # b * t, 1, emb
            output = th.matmul(hidden, w2) + b2 # b * t, 1, 1
            group_qvals.append(output)
        # Combine group Q-values
        group_qvals = th.cat(group_qvals, dim=2)  # Concatenate along the last dimension
        w1 = self.hyper.w1(states).view(-1, self.n_groups, self.embed_dim) # b * t, n_groups, emb
        b1 = self.hyper.b1(states).view(-1, 1, self.embed_dim)
        w2 = self.hyper.w2(states).view(-1, self.embed_dim, 1) # b * t, emb, 1
        b2 = self.hyper.b2(states).view(-1, 1, 1)
        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)
        hidden = F.elu(th.matmul(group_qvals, w1) + b1) # b * t, 1, emb
        qtot = th.matmul(hidden, w2) + b2 # b * t, 1, 1
        
        return qtot.view(b, t, -1)

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)
