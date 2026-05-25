from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import numpy as np

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_results = []
        self.episode_time_lengths = []
        self.current_episode_rewards = []
        self.current_episode_lengths = []
        
        # Counters for statistics
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.total_episodes = 0

        # NEW: reward components tracking
        self.episode_reward_components = {}
        self.current_episode_rc = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.current_episode_rewards = [0.0] * n_envs
        self.current_episode_lengths = [0] * n_envs
        # NEW: per-env reward component accumulator
        self.current_episode_rc = [dict() for _ in range(n_envs)]

    def _on_step(self) -> bool:
        dones = self.locals['dones']
        rewards = self.locals['rewards']
        infos = self.locals['infos']


        for i, done in enumerate(dones):
            self.current_episode_rewards[i] += rewards[i]
            self.current_episode_lengths[i] += 1

            # NEW: store reward components if available
            if "reward_components" in infos[i]:
                rc = infos[i]["reward_components"]

                for k, v in rc.items():
                    # print(f'{k}:{v}')
                    # print(self.current_episode_rc[i])
                    if k not in self.current_episode_rc[i]:
                        self.current_episode_rc[i][k] = 0.0
                    self.current_episode_rc[i][k] += v
                    # print(self.current_episode_rc[i])
            # print('--------------')
            # if not done:
            #     # sum the different rewards and then print them to check the contributions
            #     # print(infos[i]["reward_components"])
            #     # print('here:', self.current_episode_rc[i])

            if done:
                # print('--------------DONE--------------')
                # print('here:', self.current_episode_rc[i])

                self.episode_rewards.append(self.current_episode_rewards[i])
                self.episode_lengths.append(self.current_episode_lengths[i])
                # print('reward:', self.episode_rewards)

                # store final episode aggregated reward components
                for k, v in self.current_episode_rc[i].items():
                    if k not in self.episode_reward_components:
                        self.episode_reward_components[k] = []
                    self.episode_reward_components[k].append(v)
                # print(self.episode_reward_components)
                # exit(0)

                # reset per-env accumulator
                self.current_episode_rc[i] = {}
                
                # Extract episode result from info
                if 'episode_result' in infos[i]:
                    result = infos[i]['episode_result']
                    # print('-----------result----------------', result)
                    # print(self.episode_reward_components)
                    # print(self.current_episode_rewards)

                    self.episode_results.append(result)
                    
                    # Update counters based on episode result
                    if result == "success":
                        self.success_count += 1
                    elif result == "collision":
                        self.collision_count += 1
                    elif result == "timeout":
                        self.timeout_count += 1

                    if 'episode_time_length' in infos[i]:
                        self.episode_time_lengths.append(infos[i]['episode_time_length'])
                        
                    self.total_episodes += 1
                
                self.current_episode_rewards[i] = 0.0
                self.current_episode_lengths[i] = 0

        # Log statistics to TensorBoard every 1000 steps
        if self.n_calls % 1000 == 0 and self.total_episodes > 0:
            # Calculate mean reward and episode length
            if len(self.episode_rewards) > 0:
                mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                mean_episode_length = sum(self.episode_lengths) / len(self.episode_lengths)
                mean_episode_time_length = sum(self.episode_time_lengths) / len(self.episode_time_lengths)
                self.logger.record('metrics/mean_episode_reward', mean_reward)
                self.logger.record('metrics/mean_episode_length', mean_episode_length)
                self.logger.record('metrics/mean_episode_time_length', mean_episode_time_length)

            
            # Calculate and log success, collision, and timeout rates
            success_rate = self.success_count / self.total_episodes if self.total_episodes > 0 else 0
            collision_rate = self.collision_count / self.total_episodes if self.total_episodes > 0 else 0
            timeout_rate = self.timeout_count / self.total_episodes if self.total_episodes > 0 else 0
            
            self.logger.record('metrics/success_rate', success_rate)
            self.logger.record('metrics/collision_rate', collision_rate)
            self.logger.record('metrics/timeout_rate', timeout_rate)

            # NEW: reward component stats
            for k, vals in self.episode_reward_components.items():
                if len(vals) > 0:
                    self.logger.record(
                        f"reward_components/{k}_mean",
                        float(np.mean(vals))
                    )
                    self.logger.record(
                        f"reward_components/{k}_std",
                        float(np.std(vals))
                    )
            
            # Reset counters for the next logging interval
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_results = []
            self.episode_time_lengths = []
        return True

