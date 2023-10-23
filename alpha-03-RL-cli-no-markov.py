def display(args):
    print(args)


# /opt/wai/miniconda3/envs/python3/bin/python /opt/wai/solutions/paper-job/master/alpha-03-RL-cli-no-markov.py A2C
# /opt/wai/miniconda3/envs/python3/bin/python /opt/wai/solutions/paper-job/master/alpha-03-RL-cli-no-markov.py TD3
# /opt/wai/miniconda3/envs/python3/bin/python /opt/wai/solutions/paper-job/master/alpha-03-RL-cli-no-markov.py PPO
# /opt/wai/miniconda3/envs/python3/bin/python /opt/wai/solutions/paper-job/master/alpha-03-RL-cli-no-markov.py DDPG
# /opt/wai/miniconda3/envs/python3/bin/python /opt/wai/solutions/paper-job/master/alpha-03-RL-cli-no-markov.py SAC
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    import multiprocessing
    import os
    import pandas
    import sys

    # 자체 제작 함수 모음
    import alpha_functions

    df_base = pandas.read_csv('data-0201-base.csv')
    df_base = df_base.set_index(['기준일자'])
    display(df_base)

    window_size = 490
    start_date = 20030101

    algorithm_name = sys.argv[1]

    # list_algorithm = [ 'A2C', 'TD3', 'PPO', 'DDPG', 'SAC']
    timesteps = 25000
    rebalance_count = 10
    policy = 'MlpPolicy'

    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./data/' + algorithm_name + '-No-Markov'):
        os.mkdir('./data/' + algorithm_name + '-No-Markov')

    dict_asset_hold = dict()
    list_tickers = [
        'A005930-삼성전자-종가',
        'A000660-SK하이닉스-종가',
        'A006400-삼성SDI-종가',
        'A005380-현대차-종가',
        'A000270-기아-종가',
        'A005490-POSCO홀딩스-종가',
        'A012330-현대모비스-종가',
        'A015760-한국전력-종가',
        'A033780-KT&G-종가',
        'A003550-LG-종가',
    ]
    for ticker in list_tickers:
        dict_asset_hold[ticker] = 0
    dict_asset_hold['A999999'] = 1000000000

    df_portfolio_ratio = pandas.DataFrame()
    list_jobs = list()
    for day, row in df_base.iterrows():
        if day < start_date:
            continue

        list_asset = list()
        dict_asset = dict()
        dict_asset['기준일자'] = day

        df_part = df_base[df_base.index < day]
        df_part = df_part.tail(window_size)

        dict_param = dict()
        dict_param['table'] = df_part
        dict_param['rebalance_count'] = rebalance_count
        dict_param['std_date'] = day
        dict_param['list_code'] = list_tickers
        dict_param['algorithm'] = algorithm_name
        dict_param['policy'] = policy
        dict_param['timesteps'] = timesteps
        dict_param['file_name'] = './data/' + algorithm_name + '-No-Markov/' + str(day) + '.csv'

        list_jobs.append(dict_param)

    p = multiprocessing.Pool(int(multiprocessing.cpu_count() / 3 * 2))
    p.map(alpha_functions.execute_daily, list_jobs)

