from configs_stock import *
from empyrical import sharpe_ratio, max_drawdown, calmar_ratio, sortino_ratio
import pyfolio
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import os

matplotlib.use("Agg")

class StockEnvTrade(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        all_data,
        day,
        args,
        turbulence_threshold=140,
        initial=True,
        previous_state=[],
        model_name="",
        iteration="",
    ):
        """
        all_data: list containing the dataset.
        day: the date on which the agent will start trading.
        last_day: last_day in the dataset.
        """

        self.args = args
        self.day = day
        self.all_data = all_data
        self.data = self.all_data[self.day]
        self.initial = initial
        self.previous_state = previous_state

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEAT_DIMS,)
        )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold

        # initalize state
        last_price = self.data["adj_close_last"].view(-1).tolist()
        target_price = self.data["adj_close_target"].view(-1).tolist()
        len_data = self.data["length_data"].view(-1).tolist()
        emb_data = self.data["embedding"].view(-1).tolist()
        text_diff = self.data["text_difficulty"].view(-1).tolist()
        vol_diff = self.data["volatility"].view(-1).tolist()
        price_text_diff = self.data["price_text_difficulty"].view(-1).tolist()
        price_diff = self.data["price_difficulty"].view(-1).tolist()
        all_diff = self.data["price_text_vol_difficulty"].view(-1).tolist()
        time_feats = self.data["time_features"].view(-1).tolist()
        self.state = (
            [INITIAL_ACCOUNT_BALANCE]  # balance
            + last_price  # stock prices initial
            + [0] * STOCK_DIM  # stocks on hold
            + emb_data  # tweet features
            + len_data  # tweet len
            + target_price  # target price
            + price_diff
            + vol_diff
            + text_diff
            + price_text_diff
            + all_diff
            + time_feats
        )
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        # value of assets cash + shares
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        # biased CL rewards, during test all diff values are 1 so no change will happen in rewards
        self.rewards_memory = []
        # self.reset()
        self._seed()
        self.model_name = model_name
        self.iteration = iteration

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            # check whether you have some stocks or not
            if self.state[index + STOCK_DIM + 1] > 0:
                # update balance by adding the money you get ater selling that particular stock
                self.state[0] += (
                    self.state[index + 1]
                    * min(abs(action), self.state[index + STOCK_DIM + 1])
                    * (1 - TRANSACTION_FEE_PERCENT)
                )
                # update the number of stocks
                self.state[index + STOCK_DIM + 1] -= min(
                    abs(action), self.state[index + STOCK_DIM + 1]
                )
                self.cost += (
                    self.state[index + 1]
                    * min(abs(action), self.state[index + STOCK_DIM + 1])
                    * TRANSACTION_FEE_PERCENT
                )
                self.trades += 1
            else:
                pass
        else:
            # if turbulence goes over threshold, just clear out all positions
            if self.state[index + STOCK_DIM + 1] > 0:
                # update balance
                self.state[0] += (
                    self.state[index + 1]
                    * self.state[index + STOCK_DIM + 1]
                    * (1 - TRANSACTION_FEE_PERCENT)
                )
                self.state[index + STOCK_DIM + 1] = 0
                self.cost += (
                    self.state[index + 1]
                    * self.state[index + STOCK_DIM + 1]
                    * TRANSACTION_FEE_PERCENT
                )
                self.trades += 1
            else:
                pass

    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index + 1]
            # print('available_amount:{}'.format(available_amount))
            # update balance
            self.state[0] -= (
                self.state[index + 1]
                * min(available_amount, action)
                * (1 + TRANSACTION_FEE_PERCENT)
            )

            self.state[index + STOCK_DIM + 1] += min(available_amount, action)

            self.cost += (
                self.state[index + 1]
                * min(available_amount, action)
                * TRANSACTION_FEE_PERCENT
            )
            self.trades += 1
        else:
            # if turbulence goes over threshold, just stop buying
            pass

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.all_data) - 1
        # print(actions)

        if self.terminal:
            print("Reached the end.")
            if not os.path.exists("results"):
                os.makedirs("results")
            plt.plot(self.asset_memory, "r")
            plt.savefig(
                "results/account_value_trade_{}_{}.png".format(
                    self.model_name, self.iteration
                )
            )
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(
                "results/account_value_trade_{}_{}.csv".format(
                    self.model_name, self.iteration
                )
            )
            end_total_asset = self.state[0] + sum(
                np.array(self.state[HOLDING_IDX:EMB_IDX])
                * np.array(self.state[TARGET_IDX:PRICEDIFF_IDX])
            )
            print("previous_total_asset:{}".format(self.asset_memory[0]))

            print("end_total_asset:{}".format(end_total_asset))
            print("total_reward:{}".format(end_total_asset - self.asset_memory[0]))
            print("total_cost: ", self.cost)
            print("total trades: ", self.trades)

            df_total_value.columns = ["account_value"]
            df_total_value["daily_return"] = df_total_value.pct_change(1)

            cum_returns = (
                (end_total_asset - self.asset_memory[0]) / (self.asset_memory[0])
            ) * 100
            sharpe = sharpe_ratio(df_total_value["daily_return"])
            sortino = sortino_ratio(df_total_value["daily_return"])
            calmar = calmar_ratio(df_total_value["daily_return"])
            mdd = max_drawdown(df_total_value["daily_return"]) * 100

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv(
                "results/account_rewards_trade_{}_{}.csv".format(
                    self.model_name, self.iteration
                )
            )

            return (
                self.state,
                self.reward,
                self.terminal,
                {
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "calmar": calmar,
                    "mdd": mdd,
                    "cum_returns": cum_returns,
                },
            )

        else:
            actions = actions * HMAX_NORMALIZE
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            begin_total_asset = np.array(self.state[HOLDING_IDX:EMB_IDX]) * np.array(
                self.state[LAST_PRICE_IDX:HOLDING_IDX]
            )
            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            end_total_asset = np.array(self.state[HOLDING_IDX:EMB_IDX]) * np.array(
                self.state[TARGET_IDX:PRICEDIFF_IDX]
            )

            self.asset_memory.append(self.state[0] + sum(end_total_asset))

            if self.args.diff == "price":
                self.reward = sum(
                    (end_total_asset - begin_total_asset)
                    * np.array(self.state[PRICEDIFF_IDX:VOLDIFF_IDX])
                )
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * REWARD_SCALING
            elif self.args.diff == "vol":
                self.reward = sum(
                    (end_total_asset - begin_total_asset)
                    * np.array(self.state[VOLDIFF_IDX:TEXTDIFF_IDX])
                )
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * REWARD_SCALING
            elif self.args.diff == "text":
                self.reward = sum(
                    (end_total_asset - begin_total_asset)
                    * np.array(self.state[TEXTDIFF_IDX:PRICE_TEXT_DIFF_IDX])
                )
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * REWARD_SCALING
            elif self.args.diff == "price_text":
                self.reward = sum(
                    (end_total_asset - begin_total_asset)
                    * np.array(self.state[PRICE_TEXT_DIFF_IDX:ALLDIFF_IDX])
                )
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * REWARD_SCALING
            elif self.args.diff == "pvt":
                self.reward = sum(
                    (end_total_asset - begin_total_asset)
                    * np.array(self.state[ALLDIFF_IDX:TIME_IDX])
                )
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * REWARD_SCALING
            else:
                self.reward = sum(end_total_asset - begin_total_asset)
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * REWARD_SCALING

            self.day += 1
            self.data = self.all_data[self.day]

            last_price = self.data["adj_close_last"].view(-1).tolist()
            target_price = self.data["adj_close_target"].view(-1).tolist()
            len_data = self.data["length_data"].view(-1).tolist()
            emb_data = self.data["embedding"].view(-1).tolist()
            text_diff = self.data["text_difficulty"].view(-1).tolist()
            vol_diff = self.data["volatility"].view(-1).tolist()
            price_text_diff = self.data["price_text_difficulty"].view(-1).tolist()
            price_diff = self.data["price_difficulty"].view(-1).tolist()
            all_diff = self.data["price_text_vol_difficulty"].view(-1).tolist()
            time_feats = self.data["time_features"].view(-1).tolist()
            self.state = (
                [self.state[0]]  # balance
                + last_price  # stock prices initial
                + list(self.state[(STOCK_DIM + 1) : (STOCK_DIM * 2 + 1)])
                + emb_data  # tweet features
                + len_data  # tweet len
                + target_price  # target price
                + price_diff
                + vol_diff
                + text_diff
                + price_text_diff
                + all_diff
                + time_feats
            )

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        if self.initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.all_data[self.day]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            # self.iteration=self.iteration
            self.rewards_memory = []
            # initiate state
            last_price = self.data["adj_close_last"].view(-1).tolist()
            target_price = self.data["adj_close_target"].view(-1).tolist()
            len_data = self.data["length_data"].view(-1).tolist()
            emb_data = self.data["embedding"].view(-1).tolist()
            text_diff = self.data["text_difficulty"].view(-1).tolist()
            vol_diff = self.data["volatility"].view(-1).tolist()
            price_text_diff = self.data["price_text_difficulty"].view(-1).tolist()
            price_diff = self.data["price_difficulty"].view(-1).tolist()
            all_diff = self.data["price_text_vol_difficulty"].view(-1).tolist()
            time_feats = self.data["time_features"].view(-1).tolist()
            self.state = (
                [INITIAL_ACCOUNT_BALANCE]  # balance
                + last_price  # stock prices initial
                + [0] * STOCK_DIM  # stocks on hold
                + emb_data  # tweet features
                + len_data  # tweet len
                + target_price  # target price
                + price_diff
                + vol_diff
                + text_diff
                + price_text_diff
                + all_diff
                + time_feats
            )
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.previous_state[1 : (STOCK_DIM + 1)])
                * np.array(self.previous_state[(STOCK_DIM + 1) : (STOCK_DIM * 2 + 1)])
            )
            self.asset_memory = [previous_total_asset]
            # self.asset_memory = [self.previous_state[0]]
            self.day = 0
            self.data = self.all_data[self.day]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            # self.iteration=iteration
            self.rewards_memory = []
            last_price = self.data["adj_close_last"].view(-1).tolist()
            target_price = self.data["adj_close_target"].view(-1).tolist()
            len_data = self.data["length_data"].view(-1).tolist()
            emb_data = self.data["embedding"].view(-1).tolist()
            text_diff = self.data["text_difficulty"].view(-1).tolist()
            vol_diff = self.data["volatility"].view(-1).tolist()
            price_text_diff = self.data["price_text_difficulty"].view(-1).tolist()
            price_diff = self.data["price_difficulty"].view(-1).tolist()
            all_diff = self.data["price_text_vol_difficulty"].view(-1).tolist()
            time_feats = self.data["time_features"].view(-1).tolist()
            self.state = (
                [self.previous_state[0]]  # balance
                + last_price  # stock prices initial
                # stocks on hold
                + self.previous_state[HOLDING_IDX:EMB_IDX]
                + emb_data  # tweet features
                + len_data  # tweet len
                + target_price  # target price
                + price_diff
                + vol_diff
                + text_diff
                + price_text_diff
                + all_diff
                + time_feats
            )

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]