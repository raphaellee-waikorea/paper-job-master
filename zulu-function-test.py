def display(args):
    print(args)


def asset_allocation(p_args):
    l_dict_price = p_args['prices']
    l_dict_asset_count = p_args['asset_count']
    l_dict_asset_ratio = p_args['asset_ratio']
    l_total_amount = p_args['asset_amount']
    l_asset_code = dict_params['asset_list']
    l_balance_code = dict_params['balance_code']
    l_std_date = dict_params['std_date']

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


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    import pandas

    df_prices = pandas.read_csv('data-0201-base.csv')
    df_prices = df_prices.set_index(['기준일자'])

    # 종가로 판단해서 수량을 설정하고,
    # 시가로 수량을 조정하고,
    # 조정 후 수량을 종가로 평가한다.
    list_code = [
        'A005930', 'A000660', 'A006400', 'A005380', 'A000270',
        'A005490', 'A012330', 'A015760', 'A033780', 'A003550'
    ]
    balance_code = 'A999999'
    invest_amount = 1000000000

    # 총 자산금액
    asset_amount = 0
    # 조정 후 목표 비율: 종목 / 비율
    dict_asset_ratio = dict()
    # 현재 보유 자산: 종목 / 보유 수량
    dict_asset_count = dict()
    for equity_code in list_code:
        dict_asset_ratio[equity_code] = 0.1
        dict_asset_count[equity_code] = 0
    dict_asset_count[balance_code] = invest_amount

    day_to_start = 20030101
    day_to_finish = 20221229
    asset_amount = 0
    loop_count = 0
    prev_A005930_price = 0
    prev_A000660_price = 0
    prev_A006400_price = 0
    prev_A005380_price = 0
    prev_A000270_price = 0
    prev_A005490_price = 0
    prev_A012330_price = 0
    prev_A015760_price = 0
    prev_A033780_price = 0
    prev_A003550_price = 0
    for eod_date, row in df_prices.iterrows():
        if eod_date < day_to_start:
            continue
        if eod_date > day_to_finish:
            continue
        # 판단을 위한 가격
        df_asset = df_prices[df_prices.index <= eod_date]
        df_asset_today = df_asset.tail(1)
        df_asset_yesterday = df_asset.tail(2).head(1)

        dict_price_curr_C = dict()
        dict_price_curr_C['A005930'] = float(df_asset_today['A005930-삼성전자-종가'].tolist()[0])
        dict_price_curr_C['A000660'] = float(df_asset_today['A000660-SK하이닉스-종가'].tolist()[0])
        dict_price_curr_C['A006400'] = float(df_asset_today['A006400-삼성SDI-종가'].tolist()[0])
        dict_price_curr_C['A005380'] = float(df_asset_today['A005380-현대차-종가'].tolist()[0])
        dict_price_curr_C['A000270'] = float(df_asset_today['A000270-기아-종가'].tolist()[0])
        dict_price_curr_C['A005490'] = float(df_asset_today['A005490-POSCO홀딩스-종가'].tolist()[0])
        dict_price_curr_C['A012330'] = float(df_asset_today['A012330-현대모비스-종가'].tolist()[0])
        dict_price_curr_C['A015760'] = float(df_asset_today['A015760-한국전력-종가'].tolist()[0])
        dict_price_curr_C['A033780'] = float(df_asset_today['A033780-KT&G-종가'].tolist()[0])
        dict_price_curr_C['A003550'] = float(df_asset_today['A003550-LG-종가'].tolist()[0])

        dict_price_curr_O = dict()
        dict_price_curr_O['A005930'] = float(df_asset_today['A005930-삼성전자-시가'].tolist()[0])
        dict_price_curr_O['A000660'] = float(df_asset_today['A000660-SK하이닉스-시가'].tolist()[0])
        dict_price_curr_O['A006400'] = float(df_asset_today['A006400-삼성SDI-시가'].tolist()[0])
        dict_price_curr_O['A005380'] = float(df_asset_today['A005380-현대차-시가'].tolist()[0])
        dict_price_curr_O['A000270'] = float(df_asset_today['A000270-기아-시가'].tolist()[0])
        dict_price_curr_O['A005490'] = float(df_asset_today['A005490-POSCO홀딩스-시가'].tolist()[0])
        dict_price_curr_O['A012330'] = float(df_asset_today['A012330-현대모비스-시가'].tolist()[0])
        dict_price_curr_O['A015760'] = float(df_asset_today['A015760-한국전력-시가'].tolist()[0])
        dict_price_curr_O['A033780'] = float(df_asset_today['A033780-KT&G-시가'].tolist()[0])
        dict_price_curr_O['A003550'] = float(df_asset_today['A003550-LG-시가'].tolist()[0])

        if dict_price_curr_O['A005930'] == 0:
            dict_price_curr_O['A005930'] = prev_A005930_price
        else:
            prev_A005930_price = float(df_asset_today['A005930-삼성전자-시가'].tolist()[0])
        if dict_price_curr_O['A000660'] == 0:
            dict_price_curr_O['A000660'] = prev_A000660_price
        else:
            prev_A000660_price = float(df_asset_today['A000660-SK하이닉스-시가'].tolist()[0])
        if dict_price_curr_O['A006400'] == 0:
            dict_price_curr_O['A006400'] = prev_A006400_price
        else:
            prev_A006400_price = float(df_asset_today['A006400-삼성SDI-시가'].tolist()[0])
        if dict_price_curr_O['A005380'] == 0:
            dict_price_curr_O['A005380'] = prev_A005380_price
        else:
            prev_A005380_price = float(df_asset_today['A005380-현대차-시가'].tolist()[0])
        if dict_price_curr_O['A000270'] == 0:
            dict_price_curr_O['A000270'] = prev_A000270_price
        else:
            prev_A000270_price = float(df_asset_today['A000270-기아-시가'].tolist()[0])
        if dict_price_curr_O['A005490'] == 0:
            dict_price_curr_O['A005490'] = prev_A005490_price
        else:
            prev_A005490_price = float(df_asset_today['A005490-POSCO홀딩스-시가'].tolist()[0])
        if dict_price_curr_O['A012330'] == 0:
            dict_price_curr_O['A012330'] = prev_A012330_price
        else:
            prev_A012330_price = float(df_asset_today['A012330-현대모비스-시가'].tolist()[0])
        if dict_price_curr_O['A015760'] == 0:
            dict_price_curr_O['A015760'] = prev_A015760_price
        else:
            prev_A015760_price = float(df_asset_today['A015760-한국전력-시가'].tolist()[0])
        if dict_price_curr_O['A033780'] == 0:
            dict_price_curr_O['A033780'] = prev_A033780_price
        else:
            prev_A033780_price = float(df_asset_today['A033780-KT&G-시가'].tolist()[0])
        if dict_price_curr_O['A003550'] == 0:
            dict_price_curr_O['A003550'] = prev_A003550_price
        else:
            prev_A003550_price = float(df_asset_today['A003550-LG-시가'].tolist()[0])
        dict_params = dict()
        dict_params['prices'] = dict_price_curr_O

        dict_params['asset_count'] = dict_asset_count
        dict_params['asset_ratio'] = dict_asset_ratio
        dict_params['asset_amount'] = asset_amount
        dict_params['asset_list'] = list_code
        dict_params['balance_code'] = balance_code
        dict_params['std_date'] = eod_date

        asset_amount = 0
        dict_asset_count = asset_allocation(dict_params)
        for equity_code in list_code:
            asset_amount += dict_asset_count[equity_code] * dict_price_curr_C[equity_code]

        '''
        for equity_code in list_code:
            print(
                # 종목코드
                equity_code, '\t',
                # 보유 수량
                dict_asset_count[equity_code], '\t',
                # 당일 시가
                dict_price_curr_O[equity_code], '\t',
                # 당일 종가
                dict_price_curr_C[equity_code], '\t',
                # 당일 평가금액
                dict_asset_count[equity_code] * dict_price_curr_C[equity_code], '\t'
                )
        '''
        print('Date\t', eod_date, '\tBalance\t', dict_asset_count[balance_code], '\tAsset Amount\t', asset_amount)

        loop_count += 1
        if loop_count > 10:
            # break
            pass

