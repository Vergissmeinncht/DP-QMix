import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DeathPredictionModelNoLSTM(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_agents: int,
                 hidden_dim: int = 128,
                 dropout_rate: float = 0.2,
                 death_pred_state_loss_weight: float = 1.0,
                 death_pred_death_loss_weight: float = 1.0,
                 death_pred_lr: float = 1e-3):
        super(DeathPredictionModelNoLSTM, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim

        # Hyperparameters for auto-training
        self.state_loss_weight = death_pred_state_loss_weight
        self.death_loss_weight = death_pred_death_loss_weight
        self.grad_clip = 1.0
        self.lr = death_pred_lr

        # Initialize optimizer (will be called during setup)
        self.optimizer = None

        # 直接连接输入到预测头
        self.input_dim = state_dim + action_dim + 1
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 状态预测头（输出所有时间步的预测）
        self.state_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # 死亡概率预测头（输出所有时间步的预测）
        self.death_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 每个时间步预测1个值
            nn.Sigmoid()  # 确保输出在[0,1]之间
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        # Initialize optimizer
        self.setup_optimizer(self.lr)

    def setup_optimizer(self, lr=None):
        """Setup or reset optimizer with the given learning rate"""
        if lr is not None:
            self.lr = lr
        self.optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        return self.optimizer

    def forward(self,
                states: th.Tensor,
                actions: th.Tensor,
                masks: th.Tensor) -> tuple:
        """
        输入维度说明:
            states: (B, T, N, state_dim)
            actions: (B, T, N, action_dim)
            masks: (B, T, N)
        输出维度:
            next_states_pred: (B, T, N, state_dim)
            death_probs: (B, T, N)
        """
        B, T, N, _ = states.shape

        # 合并输入特征 (保留时间步信息)
        inputs = th.cat([
            states,
            actions,
            masks.unsqueeze(-1).float()  # (B, T, N) -> (B, T, N, 1)
        ], dim=-1)  # (B, T, N, state_dim + action_dim + 1)

        # 调整维度以适应全连接层
        inputs = inputs.reshape(B * T * N, -1)

        # 前向传播
        fc_out = self.fc(inputs)

        # 预测下一状态（每个时间步独立预测）
        state_pred = self.state_pred(fc_out)  # (B*T*N, state_dim)
        state_pred = state_pred.view(B, T, N, self.state_dim)  # (B, T, N, state_dim)

        # 预测死亡概率（每个时间步独立预测）
        death_probs = self.death_pred(fc_out)  # (B*T*N, 1)
        death_probs = death_probs.view(B, T, N)  # (B, T, N)

        return state_pred, death_probs

    def update(self,
               states: th.Tensor,
               actions: th.Tensor,
               masks: th.Tensor,
               next_states: th.Tensor,
               next_masks: th.Tensor) -> dict:
        """
        使用提供的一批数据自动更新模型

        Args:
            states: 当前状态 (B, T, N, state_dim)
            actions: 当前动作 (B, T, N, action_dim)
            masks: 当前死亡掩码 (B, T, N)
            next_states: 下一时刻状态 (B, T, N, state_dim)
            next_masks: 下一时刻死亡掩码 (B, T, N)

        Returns:
            包含损失信息的字典
        """

        # 前向传播获取预测
        state_pred, death_probs = self(states, actions, masks)

        # 计算状态预测损失
        state_loss = F.mse_loss(
            state_pred.reshape(-1, self.state_dim),
            next_states.reshape(-1, self.state_dim)
        )

        # 计算死亡预测损失
        death_loss = F.binary_cross_entropy(
            death_probs.reshape(-1),
            next_masks.float().reshape(-1)
        )

        # 总损失
        loss = self.state_loss_weight * state_loss + self.death_loss_weight * death_loss

        # 自动反向传播与参数更新
        if self.optimizer is None:
            self.setup_optimizer()

        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()
        return {
            'state_loss': state_loss.item(),
            'death_loss': death_loss.item(),
            'total_loss': loss.item()
        }
