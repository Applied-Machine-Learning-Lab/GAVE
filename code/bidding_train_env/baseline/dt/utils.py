import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np
import pickle
import random


def getScore(budget, cpa_cons, states, all_reward):
    beta = 2
    curr_cost = budget * (1 - states[:, 1]).reshape(-1,1)
    curr_all_reward = all_reward.reshape(-1,1)
    curr_cpa = curr_cost / (curr_all_reward + 1e-10)
    curr_coef = cpa_cons / (curr_cpa + 1e-10)
    curr_penalty = pow(curr_coef, beta)
    curr_penalty = np.where(curr_penalty > 1.0, 1.0, curr_coef)
    curr_score = curr_penalty * curr_all_reward

    return curr_score

class EpisodeReplayBuffer(Dataset):
    def __init__(self, state_dim, act_dim, data_path, scale=2000, K=20):
        self.device = "cpu"
        super(EpisodeReplayBuffer, self).__init__()
        self.scale = scale

        self.state_dim = state_dim
        self.act_dim = act_dim
        training_data = pd.read_csv(data_path)

        def safe_literal_eval(val):
            if pd.isna(val):
                return val
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                print(ValueError)
                return val
        training_data["state"] = training_data["state"].apply(safe_literal_eval)
        training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
        self.trajectories = training_data

        (self.states, self.rewards, self.actions, self.returns, self.traj_lens, self.dones,
         self.next_states, self.budget, self.cpacons) = [], [], [], [], [], [], [], [], []
        state = []
        reward = []
        action = []
        dones = []
        next_state = []
        budget = []
        cpacons = []
        for index, row in self.trajectories.iterrows():
            state.append(row["state"])
            reward.append(row['reward'])
            action.append(row["action"])
            dones.append(row["done"])
            next_state.append(row["next_state"])
            budget.append(row["budget"])
            cpacons.append(row["CPAConstraint"])
            if row["done"]:
                if len(state) != 1:
                    self.states.append(np.array(state))
                    self.rewards.append(np.expand_dims(np.array(reward), axis=1))
                    self.actions.append(np.expand_dims(np.array(action), axis=1))
                    self.returns.append(sum(reward))
                    self.traj_lens.append(len(state))
                    self.dones.append(np.array(dones))
                    next_state[-1] = next_state[-2]
                    self.next_states.append(np.array(next_state))
                    self.budget.append(np.expand_dims(np.array(budget), axis=1))
                    self.cpacons.append(np.expand_dims(np.array(cpacons), axis=1))
                state = []
                reward = []
                action = []
                dones = []
                next_state = []
                budget = []
                cpacons = []
        self.traj_lens, self.returns = np.array(self.traj_lens), np.array(self.returns)

        tmp_states = np.concatenate(self.states, axis=0)
        self.state_mean, self.state_std = np.mean(tmp_states, axis=0), np.std(tmp_states, axis=0) + 1e-6

        self.trajectories = []
        for i in range(len(self.states)):
            all_reward = np.zeros(1 + len(self.rewards[i]))
            all_reward[0] = 0
            for ind in range(1, len(all_reward)):
                all_reward[ind] = all_reward[ind - 1] + self.rewards[i][ind - 1]
            s_rtg = np.concatenate((self.states[i], self.next_states[i][-1].reshape((1,-1))),axis=0)
            curr_score = getScore(self.budget[i][0], self.cpacons[i][0], s_rtg, all_reward)
            curr_score = curr_score[-1]-curr_score
            self.trajectories.append(
                {"observations": self.states[i], "actions": self.actions[i], "rewards": self.rewards[i],
                 "dones": self.dones[i], "next_states": self.next_states[i], "budget": self.budget[i],
                 "cpacons": self.cpacons[i], "all_reward": all_reward, "curr_score": curr_score})

        self.K = K
        self.pct_traj = 1.

        num_timesteps = sum(self.traj_lens)
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)  # lowest to highest
        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        self.sorted_inds = sorted_inds[-num_trajectories:]

        self.p_sample = self.traj_lens[self.sorted_inds] / sum(self.traj_lens[self.sorted_inds])

    def __getitem__(self, index):
        traj = self.trajectories[int(self.sorted_inds[index])]
        start_t = random.randint(0, max(traj['rewards'].shape[0] -self.K, 0))

        s = traj['observations'][start_t: start_t + self.K]
        a = traj['actions'][start_t: start_t + self.K]
        r = traj['rewards'][start_t: start_t + self.K].reshape(-1, 1)
        sn = traj['next_states'][start_t: start_t + self.K]
        all_reward = traj['all_reward'][start_t: start_t + self.K+1].reshape(-1, 1)
        curr_score = traj['curr_score'][start_t: start_t + self.K+1].reshape(-1, 1)
        if 'terminals' in traj:
            d = traj['terminals'][start_t: start_t + self.K]
        else:
            d = traj['dones'][start_t: start_t + self.K]
        timesteps = np.arange(start_t, start_t + s.shape[0])

        tlen = s.shape[0]

        s = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), s], axis=0)
        a = np.concatenate([np.ones((self.K - tlen, self.act_dim)) * -10., a], axis=0)
        r = np.concatenate([np.zeros((self.K - tlen, 1)), r], axis=0)
        sn = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), sn], axis=0)
        d = np.concatenate([np.ones((self.K - tlen)) * 2, d], axis=0)
        all_reward = np.concatenate([np.zeros((self.K - tlen, 1)), all_reward], axis=0)
        curr_score = np.concatenate([np.zeros((self.K - tlen, 1)), curr_score], axis=0)
        timesteps = np.concatenate([np.zeros((self.K - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.K - tlen)), np.ones((tlen))], axis=0)
        s = (s - self.state_mean) / self.state_std
        r = r / self.scale
        sn = (sn - self.state_mean) / self.state_std
        all_reward = all_reward / self.scale
        curr_score = curr_score / self.scale

        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)
        sn = torch.from_numpy(sn).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(d).to(dtype=torch.long, device=self.device)
        all_reward = torch.from_numpy(all_reward).to(dtype=torch.float32, device=self.device)
        curr_score = torch.from_numpy(curr_score).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(mask).to(device=self.device)
        return s, a, r, d, all_reward, curr_score, timesteps, mask, sn

    def discount_cumsum(self, x, gamma=1.):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

