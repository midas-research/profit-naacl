import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import pickle as pkl
from util import *

class Evaluator(object):
    def __init__(self, num_episodes, interval, save_path="", max_episode_length=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes, 0)
        self.rewards_list = []
        self.asset_memory = []

    def __call__(self, env, policy, debug=False, visualize=False, save=True):
        self.is_training = False
        observation = None
        result = []
        sharpe = []
        sortino = []
        calmar = []
        mdd = []
        cum_returns = []
        rewards_list = []

        for episode in range(self.num_episodes):

            # reset at the start of episode
            observation = env.reset()
            episode_steps = 0
            episode_reward = 0.0

            assert observation is not None

            # start episode
            done = False
            while not done:
                action = policy(observation)

                observation, reward, done, info = env.step(action)
                if (
                    self.max_episode_length
                    and episode_steps >= self.max_episode_length - 1
                ):
                    done = True

                if visualize:
                    env.render(mode="human")

                # update
                episode_reward += reward
                rewards_list.append(reward)
                episode_steps += 1

            if debug:
                prYellow(
                    "[Evaluate] #Episode{}: episode_reward:{}".format(
                        episode, episode_reward
                    )
                )

            result.append(episode_reward)
            sortino.append(info["sortino"])
            calmar.append(info["calmar"])
            sharpe.append(info["sharpe"])
            mdd.append(info["mdd"])
            cum_returns.append(info["cum_returns"])

        self.asset_memory = env.asset_memory

        result = np.array(result).reshape(-1, 1)
        self.results = np.hstack([self.results, result])
        self.rewards_list = rewards_list

        return (
            np.mean(result),
            np.mean(sharpe),
            np.mean(sortino),
            np.mean(calmar),
            np.mean(mdd),
            np.mean(cum_returns),
        )

    def save_results(self, fn):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel("Timestep")
        plt.ylabel("Asset_Value")
        plt.plot(self.asset_memory, "r")
        plt.savefig(os.path.join(fn, "asset_value.png"))
        with open(os.path.join(fn, "asset_value.pkl"), "wb") as f:
            pkl.dump(self.asset_memory, f)
        plt.close()