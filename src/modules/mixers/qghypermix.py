import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_

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

class GHyperMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(GHyperMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_groups = 3
        self.embed_dim = args.mixing_embed_dim
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.hypernet_embed = args.hypernet_embed
        self.state_ally_feats_size = args.state_ally_feats_size
        self.state_enemy_feats_size = args.state_enemy_feats_size
        self.input_dim = self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = args.obs_shape
        
        self.hidden_states = None
        
        self.abs = abs
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        assert self.qmix_pos_func == "abs"

        self.groups = nn.ModuleList([
            HyperLayers(self.input_dim + self.embed_dim, self.hypernet_embed, self.n_agents, self.embed_dim) for _ in range(self.n_groups)
        ])

        self.hyper = HyperLayers(self.input_dim, self.hypernet_embed, self.n_groups, self.embed_dim)

        self.embedding_w1 = nn.Sequential(nn.Linear(self.input_dim, self.hypernet_embed),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(self.hypernet_embed, self.obs_dim * self.embed_dim))
        self.embedding_b1 = nn.Sequential(nn.Linear(self.input_dim, self.hypernet_embed),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(self.hypernet_embed, self.n_agents * self.embed_dim))
        
        self.rnn = nn.GRUCell(self.embed_dim, self.rnn_hidden_dim)
        
        self.embedding_w2 = nn.Sequential(nn.Linear(self.input_dim, self.hypernet_embed),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(self.hypernet_embed, self.rnn_hidden_dim * self.embed_dim))
        self.embedding_b2 = nn.Sequential(nn.Linear(self.input_dim, self.hypernet_embed),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(self.hypernet_embed, self.n_agents * self.embed_dim))

    def get_masks(self, ally_states):
        # 只取初始时间的类型（无人死亡）
        type_matrix = ally_states[:, :, :, -3:]
        types = [th.where(type_matrix[:, 0, :, i] == 1) for i in range(self.n_groups)]
        masks = [th.zeros(ally_states.shape[0], ally_states.shape[1], self.n_agents).to(ally_states.device) for _ in range(self.n_groups)]
        for i in range(self.n_groups):
            masks[i][types[i][0], :, types[i][1]] = 1
        return masks

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = th.zeros(1, self.rnn_hidden_dim, device=self.embedding_w1[0].weight.device).unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def forward(self, qvals, states, obs):
        b, t, a = qvals.size()
        
        split_indices = [self.n_agents * self.state_ally_feats_size, self.n_agents * self.state_ally_feats_size + self.n_enemies * self.state_enemy_feats_size]
        ally_states, enemy_states, last_actions = np.split(states, split_indices, axis=2)
        ally_states = ally_states.reshape(b, t, self.n_agents, self.state_ally_feats_size)
        
        masks = self.get_masks(ally_states)  # [b, t, n_agents]
        states = states.reshape(-1, self.state_dim)
        
        # 提取group特征
        w1 = self.embedding_w1(states).view(b, t, self.obs_dim, self.embed_dim)  # (b, t, obs_dim, emb)
        b1 = self.embedding_b1(states).view(b, t, self.n_agents, self.embed_dim)  # (b, t, n_agents, emb)
        w2 = self.embedding_w2(states).view(b, t, self.rnn_hidden_dim, self.embed_dim)  # (b, t, rnn_hidden_dim, emb)
        b2 = self.embedding_b2(states).view(b, t, self.n_agents, self.embed_dim)  # (b, t, n_agents, emb)
        # 不需要单调性
        # ally_embedding = F.elu(th.matmul(obs, w1) + b1).to(obs.device)  # (b, t, n_agents, emb)
        ally_embedding_1 = F.elu(th.matmul(obs, w1) + b1).reshape(-1, self.embed_dim).to(obs.device)  # (b * t * n_agents, emb)
        ally_embedding_h = self.hidden_states.reshape(-1, self.rnn_hidden_dim)
        ally_embedding_h = self.rnn(ally_embedding_1, ally_embedding_h)
        self.hidden_states = ally_embedding_h.view(b, t, a, -1)  # (b, t, n_agents, rnn_hidden_dim)
        ally_embedding = F.elu(th.matmul(self.hidden_states, w2) + b2).to(obs.device)  # (b, t, n_agents, emb)
        # 计算loss
        group_embeddings = []

        for i in range(self.n_groups):
            group_embedding = ally_embedding * masks[i].unsqueeze(-1)  # (b, t, n_agents, emb)
            group_embeddings.append(group_embedding)
        # 类内
        intra_class_similarity = [0 for _ in range(self.n_groups)]
        count_intra = [0 for _ in range(self.n_groups)]
        for i in range(self.n_groups):
            cosine_sim = F.cosine_similarity(group_embeddings[i].unsqueeze(2), group_embeddings[i].unsqueeze(3), dim=4)
            intra_class_similarity[i] += cosine_sim.sum()
            count_intra[i] += th.count_nonzero(cosine_sim)
        # 类间
        inter_class_similarity = [0 for _ in range(int(self.n_groups * (self.n_groups - 1) / 2))]
        count_inter = [0 for _ in range(int(self.n_groups * (self.n_groups - 1) / 2))]
        index = 0
        for i in range(self.n_groups):
            for j in range(i + 1, self.n_groups):
                cosine_sim = F.cosine_similarity(group_embeddings[i].unsqueeze(2), group_embeddings[j].unsqueeze(3), dim=4)
                inter_class_similarity[index] += cosine_sim.sum()
                count_inter[index] += th.count_nonzero(cosine_sim)
                index += 1
        # 计算最终损失
        group_loss = 0
        for i in range(self.n_groups):
            intra_class_similarity[i] /= count_intra[i]
            inter_class_similarity[i] /= count_inter[i]
            group_loss += -intra_class_similarity[i] + inter_class_similarity[i]  # 鼓励内部相似度高，类间相似度低
        
        # 拼接至states后
        states_enhanced = []
        for i in range(self.n_groups):
            masked_embedding = ally_embedding * masks[i].unsqueeze(-1)  # (b, t, n_agents, emb)
            mask_sum = masks[i].unsqueeze(-1).sum(dim=-2, keepdim=True)  # (b, t, 1, 1)
            group_embedding = masked_embedding.sum(dim=-2, keepdim=True) / (mask_sum + 1e-8)  # (b, t, 1, emb)
            state_enhanced = th.cat((states, group_embedding.reshape(-1, self.embed_dim)), dim=1)  # (b*t, state_dim+emb)
            states_enhanced.append(state_enhanced)
        group_qvals = []
        for i, group in enumerate(self.groups):
            w1 = group.w1(states_enhanced[i]).view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
            b1 = group.b1(states_enhanced[i]).view(-1, 1, self.embed_dim)
            w2 = group.w2(states_enhanced[i]).view(-1, self.embed_dim, 1) # b * t, emb, 1
            b2 = group.b2(states_enhanced[i]).view(-1, 1, 1)
            if self.abs:
                w1 = self.pos_func(w1)
                w2 = self.pos_func(w2)
            masked_qvals = (qvals * masks[i].to(qvals.device)).reshape(b * t, 1, self.n_agents)
            hidden = F.elu(th.matmul(masked_qvals, w1) + b1) # b * t, 1, emb
            output = th.matmul(hidden, w2) + b2 # b * t, 1, 1
            group_qvals.append(output)
        
        # Combine group Q-values
        group_qvals = th.cat(group_qvals, dim=2)  # Concatenate along the last dimension (b*t, 1, n_groups)

        w1 = self.hyper.w1(states).view(-1, self.n_groups, self.embed_dim) # (b*t, n_groups, emb)
        b1 = self.hyper.b1(states).view(-1, 1, self.embed_dim)
        w2 = self.hyper.w2(states).view(-1, self.embed_dim, 1) # (b*t, emb, 1)
        b2 = self.hyper.b2(states).view(-1, 1, 1)
        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)
        hidden = F.elu(th.matmul(group_qvals, w1) + b1) # (b*t, 1, emb)
        qtot = th.matmul(hidden, w2) + b2 # (b*t, 1, 1)
        
        return qtot.view(b, t, -1), group_loss