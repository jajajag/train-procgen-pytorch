import numpy as np
import pandas as pd
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time

class Logger(object):
    
    def __init__(self, n_envs, logdir, eval_env=False):
        self.start_time = time.time()
        self.n_envs = n_envs
        self.logdir = logdir

        self.episode_rewards = []
        for _ in range(n_envs):
            self.episode_rewards.append([])
        self.episode_len_buffer = deque(maxlen = 40)
        self.episode_reward_buffer = deque(maxlen = 40)
        
        # JAG: Add eval
        self.eval_env = eval_env
        columns = [
                'timesteps', 'wall_time', 'num_episodes',
                'max_episode_rewards', 'mean_episode_rewards',
                'min_episode_rewards', 'max_episode_len',
                'mean_episode_len', 'min_episode_len']
        if eval_env:
            columns += [
                    'eval_mean_episode_rewards', 'eval_mean_episode_len']
            self.eval_episode_rewards = [[] for _ in range(n_envs)]
            self.eval_episode_len_buffer = deque(maxlen = 40)
            self.eval_episode_reward_buffer = deque(maxlen = 40)
        self.log = pd.DataFrame(columns = columns)
        self.writer = SummaryWriter(logdir)
        self.timesteps = 0
        self.num_episodes = 0

    def feed(self, rew_batch, done_batch):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards[i].append(rew_batch[i][j])
                if done_batch[i][j]:
                    self.episode_len_buffer.append(
                            len(self.episode_rewards[i]))
                    self.episode_reward_buffer.append(
                            np.sum(self.episode_rewards[i]))
                    self.episode_rewards[i] = []
                    self.num_episodes += 1
        self.timesteps += (self.n_envs * steps)

    def write_summary(self, summary):
        for key, value in summary.items():
            self.writer.add_scalar(key, value, self.timesteps)

    def dump(self):
        wall_time = time.time() - self.start_time
        if self.num_episodes > 0:
            episode_statistics = self._get_episode_statistics()
            episode_statistics_list = list(episode_statistics.values())
            for key, value in episode_statistics.items():
                self.writer.add_scalar(key, value, self.timesteps)
        else:
            episode_statistics_list = [None] * 6
        log = [self.timesteps] + [wall_time] \
                + [self.num_episodes] + episode_statistics_list
        self.log.loc[len(self.log)] = log

        # TODO: logger to append, not write!
        with open(self.logdir + '/log.csv', 'w') as f:
            self.log.to_csv(f, index = False)
        print(self.log.loc[len(self.log)-1])

    def _get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics['Rewards/max_episodes']  = np.max(
                self.episode_reward_buffer)
        episode_statistics['Rewards/mean_episodes'] = np.mean(
                self.episode_reward_buffer)
        episode_statistics['Rewards/min_episodes']  = np.min(
                self.episode_reward_buffer)
        episode_statistics['Len/max_episodes']  = np.max(
                self.episode_len_buffer)
        episode_statistics['Len/mean_episodes'] = np.mean(
                self.episode_len_buffer)
        episode_statistics['Len/min_episodes']  = np.min(
                self.episode_len_buffer)
        # JAG: If eval_env
        if self.eval_env:
            episode_statistics['Rewards/eval_mean_episodes'] = np.mean(
                    self.eval_episode_reward_buffer)
            episode_statistics['Len/eval_mean_episodes']  = np.mean(
                    self.eval_episode_len_buffer)

        return episode_statistics

    # JAG: Feed function for eval env
    def feed_eval(self, eval_rew_batch, eval_done_batch):
        eval_rew_batch = np.array(eval_rew_batch).T
        eval_done_batch = np.array(eval_done_batch).T
        steps = eval_rew_batch.shape[-1]

        for i in range(self.n_envs):
            for j in range(steps):
                self.eval_episode_rewards[i].append(eval_rew_batch[i][j])
                if eval_done_batch[i][j]:
                    self.eval_episode_len_buffer.append(
                            len(self.eval_episode_rewards[i]))
                    self.eval_episode_reward_buffer.append(
                            np.sum(self.eval_episode_rewards[i]))
                    self.eval_episode_rewards[i] = []
