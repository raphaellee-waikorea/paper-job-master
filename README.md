# Paper sources of master degree in Hanyang University

## 1. Data definition

### 1.1 Types (data list)
- Korea market data (Adjust open/close price)
- Equities of KOSPI
- KOSPI index

|Brief Equity code|Equity name(Eng)|Equity name(Kor)|
|:---:|:---:|:---:|
|A005930|Samsung Electronics Co., Ltd|삼성전자|
|A000660|SK Hynix Inc.|SK하이닉스|
|A005490|POSCO HOLDINGS|POSCO홀딩스|
|A006400|Samsung SDI|삼성SDI|
|A051910|LG Chem Ltd.|LG화학|
|A005380|Hyundai Motor Company|현대차|
|A035420|NAVER|네이버|
|A003670|POSCO FUTURE M|포스코퓨처엠|
|A000270|KIA Corporation|기아|
|A012330|Hyundai Mobis|현대모비스|
|A035720|Kakao Corp.|카카오|
|A055550|SHINHAN FINANCIAL GROUP|신한지주|
|A066570|LG Electronics Inc.|LG전자|

### 1.2 Data duration (20 Years)
- 2003/01/01 ~ 2022/12/31

## 2. Algorithms

### 2.1 Decompose regime
- Regime shift

### 2.2 Portfolio Algorithms

#### 2.2.1 Equal weighting

#### 2.2.2 Modern Portfolio Theory
- Max Sharpe Ratio
- Min volatility
- Risk Parity

#### 2.2.3 Deep Reinforcement Learning
- Window size: 490 days (about 2 years)
- Action size: 1 day
- Execution: Sliding window and action
- Algorithm list
>> DDPG / A2C / SAC / TD3 / PPO

## 3. Process

#### 3.1 Creating benchmark index
- Item: KOSPI index
- Baseline: KOSPI index value at 2003/01/02
- Formulas   
>> $Kospi_t$: Current Kospi index   
>> $Kospi_b$: Kospi index at 2003/01/02
>>> $(Kospi_t - Kospi_b) / Kospi_b$

#### 3.2 Calculating Regime state
- Executing AutoRegression of Regime state probability
- Ref url: https://www.statsmodels.org/devel/examples/notebooks/generated/markov_autoregression.html

#### 3.3 Executing and saving Asset balance ratio of Algorithms
- Duration (17 Years): 2005/01/01 ~ 2022/12/31
