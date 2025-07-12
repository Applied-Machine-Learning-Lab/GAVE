import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from gin import current_scope_str


def getScore(budget, cpa_cons, states, all_reward):
    beta = 2
    curr_cost = budget * (1 - states[1])
    curr_all_reward = all_reward
    curr_cpa = curr_cost / (curr_all_reward + 1e-10)
    curr_coef = cpa_cons / (curr_cpa + 1e-10)
    curr_penalty = pow(curr_coef, beta)
    curr_penalty = 1.0 if curr_penalty > 1.0 else curr_coef
    curr_score = curr_penalty * curr_all_reward

    return curr_score

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])
        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])
        self.register_buffer("bias",
                             torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(1, 1, config['n_ctx'],
                                                                                           config['n_ctx']))
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']

    def forward(self, x, mask):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        mask = mask.view(B, -1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.where(self.bias[:, :, :T, :T].bool(), att, self.masked_bias.to(att.dtype))
        att = att + mask
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], config['n_inner']),
            nn.GELU(),
            nn.Dropout(config['resid_pdrop']),
            nn.Linear(config['n_inner'], config['n_embd']),
        )

    def forward(self, inputs_embeds, attention_mask):
        x = inputs_embeds + self.attn(self.ln1(inputs_embeds), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GAVE(nn.Module):

    def __init__(self, state_dim, act_dim, state_mean, state_std, hidden_size=64, action_tanh=False, K=20,
                 max_ep_len=96, scale=2000, warmup_steps=10000, weight_decay=0.0001,learning_rate=0.0001, time_dim=8,
                 target_return=4, device="cpu", expectile=0.99,
                 block_config={
                     "n_ctx": 1024,
                     "n_embd": 64,
                     "n_layer": 3,
                     "n_head": 1,
                     "n_inner": 512,
                     "activation_function": "relu",
                     "n_position": 1024,
                     "resid_pdrop": 0.1,
                     "attn_pdrop": 0.1,
                 }
                 ):
        super(GAVE, self).__init__()
        self.device = device

        self.length_times = 3
        self.hidden_size = hidden_size
        self.state_mean = state_mean
        self.state_std = state_std
        self.max_length = K
        self.max_ep_len = max_ep_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.target_return = target_return
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.time_dim = time_dim
        self.expectile = expectile
        block_config = block_config
        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.time_dim)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_reward = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)
        self.trans_return = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_reward = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_state = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_action = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_beta = nn.Sequential(
            nn.Linear(self.hidden_size, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        self.predict_return = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        self.predict_value = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lambda steps: min((steps + 1) / self.warmup_steps, 1))

        self.init_eval()

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = torch.cat((state_embeddings, time_embeddings), dim=-1)
        action_embeddings = torch.cat((action_embeddings, time_embeddings), dim=-1)
        returns_embeddings = torch.cat((returns_embeddings, time_embeddings), dim=-1)
        rewards_embeddings = torch.cat((rewards_embeddings, time_embeddings), dim=-1)
        state_embeddings = self.trans_state(state_embeddings)
        action_embeddings = self.trans_action(action_embeddings)
        returns_embeddings = self.trans_return(returns_embeddings)
        rewards_embeddings = self.trans_reward(rewards_embeddings)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(self.length_times)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.length_times * seq_length).to(stacked_inputs.dtype)
        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)
        x = x.reshape(-1, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3)
        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])
        if self.training:
            value_preds = self.predict_value(x[:, 1])
            beta_preds = self.predict_beta(x[:, 1]) + 0.5
            actions_1 = actions.clone().detach()*beta_preds
            action_embeddings_1 = self.embed_action(actions_1)
            action_embeddings_1 = torch.cat((action_embeddings_1, time_embeddings), dim=-1)
            action_embeddings_1 = self.trans_action(action_embeddings_1)
            stacked_inputs_1 = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings_1), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
            stacked_inputs_1 = self.embed_ln(stacked_inputs_1)
            stacked_attention_mask_1 = torch.stack(
                ([attention_mask for _ in range(self.length_times)]), dim=1
            ).permute(0, 2, 1).reshape(batch_size, self.length_times * seq_length).to(stacked_inputs_1.dtype)
            x_1 = stacked_inputs_1
            for block in self.transformer:
                x_1 = block(x_1, stacked_attention_mask_1)
            x_1 = x_1.reshape(-1, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3)
            return_preds_1 = self.predict_return(x_1[:, 2])
            return state_preds, action_preds, return_preds, None, return_preds_1, actions_1, value_preds
        return None, action_preds, None, None, None, None, None

    def get_action(self, states, actions, rewards, curr_score, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        curr_score = curr_score.reshape(1, -1, 1)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            curr_score = curr_score[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim),
                             device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            curr_score = torch.cat(
                [torch.zeros((curr_score.shape[0], self.max_length - curr_score.shape[1], 1),
                             device=curr_score.device), curr_score],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length - rewards.shape[1], 1), device=rewards.device),
                 rewards],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None
        _, action_preds, curr_score_preds, reward_preds, _, _, _ = self.forward(
            states, actions, rewards, curr_score, timesteps, attention_mask=attention_mask, **kwargs)
        return action_preds[0, -1]

    def step(self, states, actions, rewards, dones, all_reward, curr_score, timesteps, attention_mask, next_states):
        action_target, curr_score_target = torch.clone(actions).detach(), torch.clone(curr_score).detach()
        state_target = torch.clone(next_states).detach()
        curr_score_target = curr_score_target[:, 1:]
        state_preds, action_preds, curr_score_preds, reward_preds, curr_score_preds_1, action_1, value_preds = self.forward(
            states, actions, rewards, curr_score[:, :-1], timesteps, attention_mask=attention_mask,
        )
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_1 = action_1.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_1_frozen = action_1.clone().detach()
        state_dim = state_preds.shape[2]
        state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        curr_score_dim = curr_score_preds.shape[2]
        curr_score_preds = curr_score_preds.reshape(-1, curr_score_dim)[attention_mask.reshape(-1) > 0]
        curr_score_preds_1 = curr_score_preds_1.reshape(-1, curr_score_dim)[attention_mask.reshape(-1) > 0]
        curr_score_target = curr_score_target.reshape(-1, curr_score_dim)[attention_mask.reshape(-1) > 0]
        value_preds = value_preds.reshape(-1, curr_score_dim)[attention_mask.reshape(-1) > 0]
        value_preds_frozen = value_preds.clone().detach()

        # Loss function without learnable value function. It's more stable but may perform worse than a finetuned loss with learnable value function.
        # In this case, we simply boost exploration by maxmaizing curr_score_preds_1 in wo.
        # wo = torch.sigmoid(1 * (curr_score_preds_1-curr_score_preds.clone().detach()))
        # wo_frozen = wo.clone().detach()
        # loss1 = torch.mean((1-wo_frozen)*((action_preds - action_target) ** 2) + wo_frozen*((action_preds - action_1_frozen) ** 2))
        # loss2 = torch.mean((curr_score_preds - curr_score_target) ** 2)*200
        # loss3 = torch.mean(1-wo)*100
        # loss = loss1+loss2+loss3

        # The loss in the paper. It's param sensitive and need careful param selection.
        wo = torch.sigmoid(100 * (curr_score_preds_1 - curr_score_preds))
        wo_frozen = wo.clone().detach()
        diff = curr_score_target - value_preds
        weight = torch.where(diff > 0, self.expectile, (1 - self.expectile))
        loss1 = torch.mean((1 - wo_frozen) * ((action_preds - action_target) ** 2) +
                           wo_frozen * ((action_preds - action_1_frozen) ** 2))
        loss2 = torch.mean((curr_score_preds - curr_score_target) ** 2) * 200
        loss3 = torch.mean(weight * (diff ** 2)) * 100
        loss4 = torch.mean((curr_score_preds_1 - value_preds_frozen) ** 2) * 100
        loss = loss1 + loss2 + loss3 + loss4
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), .25)
        self.optimizer.step()
        return (loss.detach().cpu().item(), loss1.detach().cpu().item(), loss2.detach().cpu().item(), loss3.detach().cpu().item(),
                loss4.detach().cpu().item(), torch.mean(wo_frozen.squeeze()).cpu().item(), torch.mean(curr_score_target).cpu().item(),
                torch.mean(curr_score_preds).cpu().item(), torch.mean(curr_score_preds_1).cpu().item())

    def take_actions(self, state, target_return=None, pre_reward=None, budget=100, cpa=2):
        self.eval()
        target_return = target_return.to(self.device) if target_return is not None else self.target_return
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            all_reward = torch.zeros(1).to(self.device)
            self.eval_all_reward = torch.tensor(all_reward, dtype=torch.float32).reshape(1, 1).to(self.device)
            self.eval_curr_score = torch.tensor(target_return, dtype=torch.float32).reshape(1, 1).to(self.device)
        else:
            assert pre_reward is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0).to(self.device)
            self.eval_rewards[-1] = pre_reward
            pred_all_reward = self.eval_all_reward[0, -1] + pre_reward
            self.eval_all_reward = torch.cat([self.eval_all_reward, pred_all_reward.reshape(1, 1)], dim=1)
            curr_score = target_return - getScore(budget, cpa, self.eval_states[-1], pred_all_reward) / self.scale
            self.eval_curr_score = torch.cat([self.eval_curr_score, curr_score.reshape(1, 1)], dim=1)
            self.eval_timesteps = torch.cat(
                [self.eval_timesteps,
                 torch.ones((1, 1), dtype=torch.long).to(self.device) * self.eval_timesteps[:, -1] + 1], dim=1)
        self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim).to(self.device)], dim=0)
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1).to(self.device)])
        action = self.get_action(
            (self.eval_states.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device)) / torch.tensor(
                self.state_std).to(self.device),
            self.eval_actions.to(dtype=torch.float32),
            self.eval_rewards.to(dtype=torch.float32),
            self.eval_curr_score.to(dtype=torch.float32),
            self.eval_timesteps.to(dtype=torch.long)
        )
        self.eval_actions[-1] = action
        action = action.detach().cpu().numpy()
        return action
    def init_eval(self):
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.act_dim), dtype=torch.float32).to(self.device)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32).to(self.device)
        self.eval_target_return = None
        self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1).to(self.device)
        self.eval_episode_return, self.eval_episode_length = 0, 0

    def save_net(self, save_path, name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, name)
        torch.save(self.state_dict(), file_path)

    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/dt_model.pth')

    def load_net(self, load_path="saved_model/DTtest/dt.pt", device='cpu'):
        file_path = load_path
        self.load_state_dict(torch.load(file_path, map_location=device))
        print(f"Model loaded from {self.device}.")
