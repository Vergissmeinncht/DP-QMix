import copy
import numpy as np
import torch.nn.functional as F
import random
import time

import torch as th
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qgroupmix import GroupMixer
from modules.mixers.qgroupmix_atten import GroupMixerAtten
from modules.mixers.qgattenmix import GAttenMixer
from modules.mixers.qghypermix import GHyperMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num
from modules.FOX.FNet import Encoder, Decoder, DecoderTau, VAEModel


def calculate_target_q(target_mac, batch, enable_parallel_computing=False, thread_num=4):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)
    with th.no_grad():
        # Set target mac to testing mode
        target_mac.set_evaluation_mode()
        target_mac_out = []
        target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
        return target_mac_out


def calculate_n_step_td_target(mixer, target_mixer, target_max_qvals, batch, rewards, terminated, mask, gamma,
                               td_lambda,
                               enable_parallel_computing=False, thread_num=4, q_lambda=False, target_mac_out=None):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)

    with th.no_grad():
        # Set target mixing net to testing mode
        target_mixer.eval()
        # Calculate n-step Q-Learning targets
        if mixer == "qgattenmix" or mixer == "qghypermix":
            target_mixer.init_hidden(batch.batch_size * batch.max_seq_length)
            target_max_qvals, _ = target_mixer(target_max_qvals, batch["state"], batch["obs"])
        elif mixer == "qgroupmix-atten":
            target_max_qvals = target_mixer(target_max_qvals, batch["state"], batch["obs"])
        else:
            target_max_qvals = target_mixer(target_max_qvals, batch["state"])

        if q_lambda:
            raise NotImplementedError
            qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
            qvals = target_mixer(qvals, batch["state"])
            targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals, gamma, td_lambda)
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, gamma, td_lambda)
        return targets.detach()


class FOX_NQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.n_agents = args.n_agents
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":  # 31.521K
            self.mixer = Mixer(args)
        elif args.mixer == "qgroupmix":
            self.mixer = GroupMixer(args)
        elif args.mixer == "qgroupmix-atten":
            self.mixer = GroupMixerAtten(args)
        elif args.mixer == "qgattenmix":
            self.mixer = GAttenMixer(args)
        elif args.mixer == "qghypermix":
            self.mixer = GHyperMixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        self.encoder = Encoder(args.rnn_hidden_dim, args.predict_net_dim, 64).to(self.device)
        self.decoder = Decoder(64 * 3, args.predict_net_dim, args.n_agents * 4).to(self.device)
        self.decoder_tau = DecoderTau(64, args.predict_net_dim, 64).to(self.device)
        self.VAEModel = VAEModel(Encoder=self.encoder, Decoder=self.decoder, DecoderTau=self.decoder_tau).to(
            self.device)
        self.diversity_mean = 0
        self.normalizer = 0

        self.iteration = 25

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0

        self.enable_parallel_computing = (not self.args.use_cuda) and getattr(self.args, 'enable_parallel_computing',
                                                                              False)
        # self.enable_parallel_computing = False
        if self.enable_parallel_computing:
            from multiprocessing import Pool
            # Multiprocessing pool for parallel computing.
            self.pool = Pool(1)
        self.list = [(np.arange(args.n_agents - i) + i).tolist() + np.arange(i).tolist()
                     for i in range(args.n_agents)]

    def _get_obs_component_dim(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (6, 5), (4, 5), 1]
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        return (move_feats_dim, enemy_feats_dim_flatten, ally_feats_dim_flatten, own_feats_dim), (
            enemy_feats_dim, ally_feats_dim)

    def _get_obs_component_dim(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (6, 5), (4, 5), 1]
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        return (move_feats_dim, enemy_feats_dim_flatten, ally_feats_dim_flatten, own_feats_dim), (
            enemy_feats_dim, ally_feats_dim)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        obs_component_dim, _ = self._get_obs_component_dim()
        t = t % batch.max_seq_length
        raw_obs_t = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        move_feats_t, enemy_feats_t, ally_feats_t, own_feats_t = th.split(raw_obs_t, obs_component_dim, dim=-1)
        enemy_feats_t = enemy_feats_t.reshape(bs * self.n_agents * self.n_enemies,
                                              -1)  # [bs * n_agents * n_enemies, fea_dim]
        ally_feats_t = ally_feats_t.reshape(bs * self.n_agents * self.n_allies,
                                            -1)  # [bs * n_agents * n_allies, a_fea_dim]
        # merge move features and own features to simplify computation.
        context_feats = [move_feats_t, own_feats_t]  # [batch, agent_num, own_dim]
        own_context = th.cat(context_feats, dim=2).reshape(bs * self.n_agents, -1)  # [bs * n_agents, own_dim]

        embedding_indices = []
        if self.args.obs_agent_id:
            # agent-id indices, [bs, n_agents]
            embedding_indices.append(th.arange(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1))
        if self.args.obs_last_action:
            # action-id indices, [bs, n_agents]
            if t == 0:
                embedding_indices.append(None)
            else:
                embedding_indices.append(batch["actions"][:, t - 1].squeeze(-1))

        return bs, own_context, enemy_feats_t, ally_feats_t, embedding_indices

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        start_time = time.time()
        th.set_printoptions(threshold=10_000)

        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        rewards = batch["reward"][:, :-1]  # [128, 81, 1]
        actions = batch["actions"][:, :-1]  # [128, 81, 8, 1]
        terminated = batch["terminated"][:, :-1].float()  # [128, 81, 1]
        mask = batch["filled"][:, :-1].float()  # [128, 81, 1]
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]  # [128, 82, 8, 15]
        actions_onehot = batch["actions_onehot"][:, :-1]  # [128, 81, 8, n_actions]
        last_actions_onehot = th.cat([th.zeros_like(actions_onehot[:, 0].unsqueeze(1)), actions_onehot],
                                     dim=1)  # [128, 82, 8, n_actions]

        self.mac.set_train_mode()
        mac_out = []
        hidden_store_list = []
        local_qs_list = []
        q_f_list = []
        self.mac.init_hidden(batch.batch_size)
        initial_hidden = self.mac.hidden_states.clone().detach().to(self.args.device)  # [128, 8, 64]

        for t in range(batch.max_seq_length):  # 循环 82 次
            inputs = self._build_inputs(batch, t)
            agent_outs, hidden_store, local_qs, q_f = self.mac.agent.forward(
                inputs, initial_hidden.clone().detach(), self.VAEModel.Encoder)
            mac_out.append(agent_outs)  # [128, 8, 15]
            hidden_store_list.append(hidden_store)  # [128, 8, 64]
            local_qs_list.append(local_qs)  # [128 * 8, 15]
            q_f_list.append(q_f)  # [128 * 8, 15]
            initial_hidden = hidden_store  # 更新隐藏状态

        # 堆叠时间维度
        mac_out = th.stack(mac_out, dim=1)  # [128, 82, 8, 15]
        hidden_store = th.stack(hidden_store_list, dim=1)  # [128, 82, 8, 64]
        local_qs = th.stack(local_qs_list, dim=1).view(batch.batch_size, batch.max_seq_length, self.n_agents,
                                                       -1)  # [128, 82, 8, 15]
        q_f = th.stack(q_f_list, dim=1).view(batch.batch_size, batch.max_seq_length, self.n_agents,
                                             -1)  # [128, 82, 8, 15]

        mac_out[avail_actions == 0] = -9999999  # 屏蔽不可用动作

        # 重塑 hidden_store 为 [bs, n_agents, seq_length, rnn_hidden_dim]
        hidden_store = hidden_store.permute(0, 2, 1, 3)  # [128, 8, 82, 64]

        formation = batch["formation"][:, 1:]  # [128, 81, ...]
        max_idx = batch["max_idx"][:, :-1]  # [128, 81, 8]
        min_idx = batch["min_idx"][:, :-1]  # [128, 81, 8]
        history = hidden_store[:, :, 1:].clone().detach()  # [128, 8, 81, 64]
        visit = batch["visit"][:, 1:]  # [128, 81, ...]

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # [128, 81, 8]

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        self.target_mac.init_hidden(batch.batch_size)
        initial_hidden_target = self.target_mac.hidden_states.clone().detach().to(self.args.device)
        target_mac_out = []
        target_hidden_store_list = []
        target_local_qs_list = []
        target_q_f_list = []

        for t in range(batch.max_seq_length):
            inputs = self._build_inputs(batch, t)
            agent_outs, hidden_store, local_qs1, q_f = self.target_mac.agent.forward(
                inputs, initial_hidden_target.clone().detach(), self.VAEModel.Encoder)
            target_mac_out.append(agent_outs)
            target_hidden_store_list.append(hidden_store)
            target_local_qs_list.append(local_qs1)
            target_q_f_list.append(q_f)
            initial_hidden_target = hidden_store

        target_mac_out = th.stack(target_mac_out, dim=1)[:, 1:]  # [128, 81, 8, 15]
        target_hidden_store = th.stack(target_hidden_store_list, dim=1)[:, 1:]  # [128, 81, 8, 64]
        target_local_qs = th.stack(target_local_qs_list, dim=1).view(batch.batch_size, batch.max_seq_length,
                                                                     self.n_agents, -1)[:, 1:]  # [128, 81, 8, 15]
        target_q_f = th.stack(target_q_f_list, dim=1).view(batch.batch_size, batch.max_seq_length, self.n_agents, -1)[:,
                     1:]  # [128, 81, 8, 15]

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)  # [128, 81, 8]
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]  # [128, 81, 8]

        with th.no_grad():
            visit = visit.float()
            visit_reward = th.where(visit != 0, 1 / th.sqrt(visit), th.zeros_like(visit))
            max_visit = th.max(visit_reward).item()
            normalized_visit = visit_reward - max_visit

            formation_repeat = formation.clone().detach().unsqueeze(2).repeat(1, 1, 20, 1)

            diversity_reward = 0
            batch_size = batch.batch_size
            max_seq_length = batch.max_seq_length - 1  # 调整为 81

            for agent in range(self.args.n_agents):
                tau = history[:, agent]  # [128, 81, 64]
                idx_max = max_idx[:, :, agent].detach().long()  # [128, 81]
                idx_min = min_idx[:, :, agent].detach().long()  # [128, 81]

                tau_max = history[th.arange(batch_size)[:, None], idx_max, agent]  # [128, 81, 64]
                tau_min = history[th.arange(batch_size)[:, None], idx_min, agent]  # [128, 81, 64]

                tau = tau.unsqueeze(2).expand(-1, -1, 20, -1)  # [128, 81, 20, 64]
                tau_max = tau_max.unsqueeze(2).expand(-1, -1, 20, -1)
                tau_min = tau_min.unsqueeze(2).expand(-1, -1, 20, -1)

                z_rand1 = th.randn(batch_size, max_seq_length, 20, 64, device=self.args.device)  # [128, 81, 20, 64]
                z_rand2 = th.randn(batch_size, max_seq_length, 20, 64, device=self.args.device)

                z_i, _, _ = self.VAEModel.Encoder(tau)
                z_max, _, _ = self.VAEModel.Encoder(tau_max)
                z_min, _, _ = self.VAEModel.Encoder(tau_min)

                f_prime_z = self.VAEModel(z_i, z_max, z_min)
                f_prime_i = self.VAEModel(z_i, z_rand1, z_rand2)
                f_prime_max = self.VAEModel(z_rand1, z_max, z_rand2)
                f_prime_min = self.VAEModel(z_rand1, z_rand2, z_min)

                log_q_o = self.VAEModel.get_log_pi(f_prime_z, formation_repeat)
                log_p_i_o = self.VAEModel.get_log_pi(f_prime_i, formation_repeat)
                log_p_max_o = self.VAEModel.get_log_pi(f_prime_max, formation_repeat)
                log_p_min_o = self.VAEModel.get_log_pi(f_prime_min, formation_repeat)

                nan_mask = th.isnan(log_q_o)
                log_q_o[nan_mask] = 0
                mean_log_q_o = log_q_o.sum() / nan_mask.size(0) - nan_mask.sum()
                self.diversity_mean = max(self.diversity_mean, abs(mean_log_q_o))
                self.normalizer = abs(mean_log_q_o) / self.diversity_mean

                ir = log_q_o - log_p_i_o / 3 - log_p_max_o / 3 - log_p_min_o / 3
                max_ir = th.max(ir).item()
                min_ir = th.min(ir).item()

                ir = ((ir - max_ir) / (max_ir - min_ir + 1e-6)) * self.normalizer
                ir = ir.mean(dim=2)  # [128, 81]

                diversity_reward += ir * self.args.beta2  # [128, 81]

        self.mixer.train()
        if self.args.mixer in ["qgattenmix", "qghypermix"]:
            self.mixer.init_hidden(batch.batch_size * (batch.max_seq_length - 1))
            chosen_action_qvals, group_loss = self.mixer(chosen_action_qvals, batch["state"][:, :-1],
                                                         batch["obs"][:, :-1])
            self.target_mixer.init_hidden(batch.batch_size * (batch.max_seq_length - 1))
            target_max_qvals, _ = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch["obs"][:, :-1])
        elif self.args.mixer == "qgroupmix-atten":
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch["obs"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch["obs"][:, :-1])
        else:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        targets = rewards + diversity_reward + self.args.beta1 * normalized_visit + self.args.gamma * (
                1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            if self.mixer is None:
                tot_q_data = np.mean(tot_q_data, axis=2)
                tot_target = np.mean(tot_target, axis=2)
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(tot_q_data[:, 0]),
                                 t_env)
            return

        if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
            targets = targets.get()

        td_error = (chosen_action_qvals - targets)
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        mask_elems = mask.sum()
        loss = masked_td_error.sum() / mask_elems
        norm_loss = F.l1_loss(local_qs[:, :-1], target=th.zeros_like(local_qs[:, :-1]), reduction='none')
        mask_expand = mask.unsqueeze(-1).expand_as(norm_loss)
        norm_loss = (norm_loss * mask_expand).sum() / mask_expand.sum()
        loss += 0.1 * norm_loss

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        if self.args.mixer in ["qgattenmix", "qghypermix"]:
            self.optimiser.zero_grad()
            (loss + self.args.alpha * group_loss).backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()
        else:
            self.optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        print("Avg cost {} seconds".format(self.avg_time))

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            with th.no_grad():
                mask_elems = mask_elems.item()
                td_error_abs = masked_td_error.abs().sum().item() / mask_elems
                q_taken_mean = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
                target_mean = (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
            if self.args.mixer in ["qgattenmix", "qghypermix"]:
                self.logger.log_stat("group_loss", group_loss.item(), t_env)
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("q_taken_mean", q_taken_mean, t_env)
            self.logger.log_stat("target_mean", target_mean, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def __del__(self):
        if self.enable_parallel_computing:
            self.pool.close()
