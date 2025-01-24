import numpy as np
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.dt.dt import GAVE
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def run_dt(device="cpu", step_num=10000, dir="./data/trajectory/trajectory_data.csv", save_step=5000, model_param={},
           batch_size=32, save_dir="saved_model/DTtest", loss_report=2):
    train_model(device, step_num, dir=dir, save_step=save_step, model_param=model_param, batch_size=batch_size,
                save_dir=save_dir, loss_report=loss_report)


def train_model(device="cpu", step_num=10000, dir="./data/trajectory/trajectory_data.csv", save_step=5000, model_param={},
                batch_size=32, save_dir="saved_model/DTtest", loss_report=2):
    state_dim=16
    replay_buffer = EpisodeReplayBuffer(16, 1, data_path=dir)
    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                        save_dir)
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")

    model_param['state_mean'] = replay_buffer.state_mean
    model_param['state_std'] = replay_buffer.state_std
    model_param['device'] = device
    model = GAVE(state_dim=state_dim, act_dim=1,
                                hidden_size=model_param['hidden_size'], state_mean=model_param['state_mean'],
                                state_std=model_param['state_std'], device=model_param['device'],
                                learning_rate=model_param["learning_rate"], time_dim=model_param['time_dim'],
                                block_config=model_param['block_config'], expectile=model_param['expectile']
                                ).to(device)
    step_num = step_num
    batch_size = batch_size
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    model.train()
    i=0
    for states, actions, rewards, dones, all_reward, curr_score, timesteps, attention_mask, next_states in dataloader:
        states, actions, rewards, dones, all_reward, curr_score, timesteps, attention_mask, next_states = (states.to(device),
                                                                                        actions.to(device),
                                                                           rewards.to(device), dones.to(device), all_reward.to(device),
                                                                           curr_score.to(device), timesteps.to(device),
                                                                           attention_mask.to(device),
                                                                           next_states.to(device))

        train_loss = model.step(states, actions, rewards, dones, all_reward, curr_score, timesteps, attention_mask, next_states)
        i+=1
        if i%loss_report==0:
            logger.info("Step: {}, All loss: {}, loss1: {}, loss2: {}, loss3: {}, loss4: {}, w: {}, score_target: {}, score_preds: {}, score_preds1: {}"
                        .format(i, train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4],
                                train_loss[5], train_loss[6], train_loss[7], train_loss[8]))
        model.scheduler.step()
        if i % save_step == 0:
            model.save_net(save_dir, "{}.pt".format(str(i)))
    test_state = np.ones(state_dim, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


def load_model(device="cpu"):
    with open('./Model/DT/saved_model/normalize_dict.pkl', 'rb') as f:
        normalize_dict = pickle.load(f)
    model = GAVE(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                state_std=normalize_dict["state_std"]).to(device)
    model.load_net("Model/DTtest/saved_model", device=device)
    test_state = np.ones(16, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


if __name__ == "__main__":
    run_dt()
