import gym
import gym.utils

class StockEnvTrade(gym.Env):

    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    # 초기 자산은 10억원으로 설정
    def __init__(self, df, list_tickers, init_balance=10000000, portfolio_count=10, fee_buy=0, fee_sell=0, day=0, initial=True):
        import numpy

        df_1 = df[list_tickers]
        df_2 = df[df.columns[~df.columns.isin(list_tickers)]]

        self.init_balance = init_balance
        self.portfolio_count = portfolio_count
        self.fee_buy = fee_buy
        self.fee_sell = fee_sell
        self.df = df
        self.df_1 = df_1
        self.df_2 = df_2
        self.day = day
        self.initial = initial

        # Total number of stocks in our portfolio
        # 현금을 포함하는 것을 고려해야함
        # df_1은 주식 수, df_2는 시장 상황
        self.asset_dim = self.df_1.shape[1]
        self.market_sit = self.df_2.shape[1]

        # Action Space
        # action_space normalization and shape is self.asset_dim
        # 음수를 가지면 short 실행
        # self.action_space = gym.spaces.Box(low = -1, high = 1, shape = (self.asset_dim,))
        # 투자자산 포함 현금까지 포함해서 11종목에 빠지는 자산 없이 구성해야함
        self.action_space = gym.spaces.Box(low = -1, high = 1, shape = (self.asset_dim,))

        # State Space
        # Shape : [Current Balance]+[market situation]+[prices]+[owned shares]
        self.observation_space = gym.spaces.Box(low = 0, high = numpy.inf, shape = (1 + self.market_sit + 2*self.asset_dim,))

        # load data from a pandas dataframe
        self.data_1 = self.df_1.iloc[self.day,:]
        self.data_2 = self.df_2.iloc[self.day,:]
        self.terminal = False

        # initalize state
        self.state = [init_balance] + self.data_2.values.tolist() + self.data_1.values.tolist() + [0] * self.asset_dim

        # initialize reward
        self.reward = 0

        # memorize all the total balance change
        self.asset_memory = [init_balance]
        self.actions_memory=[ [ 1 / self.asset_dim ] * self.asset_dim]

        self._seed(0)

    def _sell_stock(self, index, action):
        import numpy
        # we need to round since we cannot buy half stock
        action = numpy.floor(action)
        if self.state[1+self.market_sit+self.asset_dim+index] > 0:
            # update balance = price stock * # of stock to sell * fee
            self.state[0] += self.state[1+self.market_sit+index] * min(abs(action), self.state[1+self.market_sit+self.asset_dim+index]) * (1 - self.fee_sell)
            self.state[1+self.market_sit+self.asset_dim+index] -= min(abs(action), self.state[1+self.market_sit+self.asset_dim+index])
        else:
            pass

    def _buy_stock(self, index, action):
        import numpy
        # we need to round since we cannot buy half stock
        action = numpy.floor(action)
        # update balance = price stock * # of stock to buy * fee
        self.state[0] -= self.state[1+self.market_sit+index] * action * (1 + self.fee_buy)
        self.state[1+self.market_sit+self.asset_dim+index] += action

    def step(self, actions):
        import numpy
        self.terminal = self.day >= len(self.df.index.unique())-1
        if self.terminal:
            return self.state, self.reward, self.terminal,{}
        else:
            actions = actions * self.portfolio_count

            begin_total_asset = self.state[0] + sum(numpy.array(self.state[(1+self.market_sit):(1+self.market_sit+self.asset_dim)])*numpy.array(self.state[(1+self.market_sit+self.asset_dim):(1+self.market_sit+2*self.asset_dim)]))
            argsort_actions = numpy.argsort(actions)

            sell_index = argsort_actions[:numpy.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:numpy.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data_1 = self.df_1.iloc[self.day,:]
            self.data_2 = self.df_2.iloc[self.day,:]

            #load next state i.e. the new value of the stocks
            self.state =  [self.state[0]] + self.data_2.values.tolist() + self.data_1.values.tolist() + list(self.state[(1+self.market_sit+self.asset_dim):(1+self.market_sit+2*self.asset_dim)])

            end_total_asset = self.state[0] + sum(numpy.array(self.state[(1+self.market_sit):(1+self.market_sit+self.asset_dim)])*numpy.array(self.state[(1+self.market_sit+self.asset_dim):(1+self.market_sit+2*self.asset_dim)]))

            self.reward = end_total_asset - begin_total_asset
            weights = self.normalization(numpy.array(self.state[(1+self.market_sit+self.asset_dim):(1+self.market_sit+2*self.asset_dim)]))

            self.actions_memory.append(weights.tolist())
            self.reward = self.reward

        return self.state, self.reward, self.terminal, {}

    def reset(self):

        self.day = 0
        self.data_1 = self.df_1.iloc[self.day,:]
        self.data_2 = self.df_2.iloc[self.day,:]
        self.terminal = False

        # memorize all the total balance change
        self.actions_memory=[[1/self.asset_dim]*self.asset_dim]

        # initiate state
        self.state = [self.init_balance] + self.data_2.values.tolist() + self.data_1.values.tolist() + [0] * self.asset_dim

        self._seed(0)

        return self.state

    def normalization(self, actions):
        import numpy
        output = actions / (numpy.sum(actions) + 1e-15)
        return output

    def save_action_memory(self):
        return self.actions_memory

    def render(self, mode='human',close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

def train(algo, policy, env_train, timesteps, seed=None, save=True):
    import stable_baselines3
    import time

    start = time.time()

    if algo == "A2C":
        model = stable_baselines3.A2C(policy, env_train, verbose=0, seed=seed)
    elif algo == "TD3":
        model = stable_baselines3.TD3(policy, env_train, verbose=0, seed=seed)
    elif algo == "PPO":
        model = stable_baselines3.PPO(policy, env_train, verbose=0, seed=seed)
    elif algo == "DDPG":
        model = stable_baselines3.DDPG(policy, env_train, verbose=0, seed=seed)
    elif algo == "SAC":
        model = stable_baselines3.SAC(policy, env_train, verbose=0, seed=seed)    

    model.learn(total_timesteps=timesteps)
    end = time.time()
    if save == True:
        model.save("results/" + algo + "_" + str(timesteps) + "_model")
    # print("Training time: ", (end - start) / 60, " minutes", flush=True)
    return model


def DRL_prediction(model, data, env, obs):
    # portfolio_weights_a2c[loop_count] = numpy.array(DRL_prediction(model, df_valid, test_env, test_obs))
    actions_memory = list()
    model.set_random_seed(524)
    for i in range(len(data.index.unique())):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if i == (len(data.index.unique()) - 2):
            actions_memory = env.env_method(method_name='save_action_memory')
    return actions_memory[0]


def asset_allocation(p_args):
    l_dict_price = p_args['prices']
    l_dict_asset_count = p_args['asset_count']
    l_dict_asset_ratio = p_args['asset_ratio']
    l_total_amount = p_args['asset_amount']
    l_asset_code = p_args['asset_list']
    l_balance_code = p_args['balance_code']
    l_std_date = p_args['std_date']

    sum_to_modify_amount = 0

    dict_asset_new = dict()
    dict_need_to_modify = dict()
    for equity_code in l_asset_code:
        asset_new = (l_total_amount + l_dict_asset_count[l_balance_code]) * l_dict_asset_ratio[equity_code]
        asset_previous = l_dict_asset_count[equity_code]
        try:
            dict_asset_new[equity_code] = int(asset_new / l_dict_price[equity_code])
        except:
            import sys
            print(l_std_date, equity_code, l_dict_price[equity_code])
            sys.exit(1)
        dict_need_to_modify[equity_code] = int(asset_new / l_dict_price[equity_code]) - asset_previous
        sum_to_modify_amount += dict_need_to_modify[equity_code] * l_dict_price[equity_code]

    # 조정 대상 금액이 잔액보다 크면, (잔액 + sum(보유수량*시가)) 규모에서 조정
    if sum_to_modify_amount > l_dict_asset_count[l_balance_code]:
        # print('Re-ALLOCATION')
        total_asset_new = l_dict_asset_count[l_balance_code]
        for equity_code in l_asset_code:
            total_asset_new += l_dict_asset_count[equity_code] * l_dict_price[equity_code]

        sum_to_modify_amount = 0
        for equity_code in l_asset_code:
            asset_new = total_asset_new * l_dict_asset_ratio[equity_code]
            asset_previous = l_dict_asset_count[equity_code]
            dict_asset_new[equity_code] = int(asset_new / l_dict_price[equity_code])
            dict_need_to_modify[equity_code] = int(asset_new / l_dict_price[equity_code]) - asset_previous
            sum_to_modify_amount += dict_need_to_modify[equity_code] * l_dict_price[equity_code]

    for equity_code in l_asset_code:
        dict_asset_new[equity_code] = l_dict_asset_count[equity_code] + dict_need_to_modify[equity_code]
    dict_asset_new[l_balance_code] = l_dict_asset_count[l_balance_code] - sum_to_modify_amount

    return dict_asset_new


def execute_daily(p_args):
    import numpy
    import os
    import pandas

    import stable_baselines3
    import stable_baselines3.common.vec_env

    l_dataframe = p_args['table']
    l_rebalance_count = p_args['rebalance_count']
    l_std_date = p_args['std_date']
    l_codes = p_args['list_code']
    l_model = p_args['algorithm']
    l_policy = p_args['policy']
    l_timesteps = p_args['timesteps']
    l_file_name = p_args['file_name']

    if os.path.exists(l_file_name):
        print('Skipped!\t', l_std_date)
    else:
        df_train = l_dataframe.head(450)
        df_valid = l_dataframe.tail(40)

        valid_length = df_valid.shape[0]
        # valid_items = df_valid.shape[1]
        valid_items = len(l_codes)

        # cumulative_returns_daily_drl_a2c = numpy.zeros([l_rebalance_count, valid_length])
        portfolio_weights_a2c = numpy.zeros([l_rebalance_count, valid_length, valid_items])

        loop_count = 0
        sequantial_seed = 524
        ### 강화학습 알고리즘 적용 시작 ###
        while (loop_count < l_rebalance_count):
            # print('강화학습 시작', l_std_date, flush=True)

            train_env = stable_baselines3.common.vec_env.DummyVecEnv([lambda: StockEnvTrade(df=df_train, list_tickers=l_codes)])
            model = train(l_model, l_policy, train_env, l_timesteps, seed=sequantial_seed, save=False)

            test_env = stable_baselines3.common.vec_env.DummyVecEnv([lambda: StockEnvTrade(df=df_valid, list_tickers=l_codes)])
            test_obs = test_env.reset()

            portfolio_weights_a2c[loop_count] = numpy.array(DRL_prediction(model, df_valid, test_env, test_obs))

            check = numpy.sum(portfolio_weights_a2c[loop_count])
            sequantial_seed += 1
            if check != valid_length:
                continue

            loop_count += 1
        # print('강화학습 끝', l_std_date, flush=True)
        ### 강화학습 알고리즘 적용 끝 ###

        ### 포트폴리오 ###
        df_port = pandas.DataFrame(numpy.mean(numpy.array(portfolio_weights_a2c), axis=0), index=df_valid.index, columns=l_codes)
        df_port = df_port.tail(1)
        # _logging(df_port)
        df_port_to_append = df_port.reset_index()
        df_port_to_append['기준일자'] = l_std_date
        
        df_port_to_append.to_csv(l_file_name, index=None)
        print('Complete!\t', l_std_date, flush=True)
