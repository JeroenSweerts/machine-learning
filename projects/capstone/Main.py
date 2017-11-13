'''Main program used for training and backtesting the model'''

import math
import warnings
import os
import random
import RLClasses
import pandas as pd
import numpy as np

#load raw DATA
DATA = RLClasses.Data('WTI.csv')
#preprocess DATA and split into training and test set
DATA.preprocess_data()
#initialize the model
TRAINMODEL = RLClasses.RLModelTrain(DATA)
#initialize the environment
ENV = RLClasses.Environment(DATA, TRAINMODEL)
#initialize the learning agent
AGENT = RLClasses.Agent()

def test2(DATA):
    '''This code is testing the model by calculating the sharpe ratio
    After a certain number of loops the training function call this function
    and adds the calculated sharpe ratio to a learning curve of sharpe ratios
    from the previous runs.'''

    test_model = RLClasses.RLModelTrain(DATA)
    test_env = RLClasses.Environment(DATA, test_model)
    test_agent = RLClasses.Agent()
    test_epsilon = 0

    price_arr = []
    position_arr = []
    return_arr = []
    date_arr = []

    for test_day in range(len(DATA.testdata)-1):
        test_env.newstate(DATA, test_model, test_agent, test_epsilon, test_day)
        price_arr.append(DATA.testdata['Last'].iloc[test_day])
        date_arr.append(DATA.testdata['Date'].iloc[test_day])
        position_arr.append(test_env.old_state_dict['position'])
        return_arr.append(((float(DATA.testdata['Last'].iloc[test_day+1])
                            / float(DATA.testdata['Last'].iloc[test_day]))-1)
                          * test_env.old_state_dict['position'])
        test_env.old_state_dict = test_env.new_state_dict

    if test_env.old_state_dict['position'] == -1:
        return_arr.append(((float(DATA.testdata['Last'].iloc[test_day + 1])
                            / float(DATA.testdata['Last'].iloc[test_day])) - 1)
                          * test_env.old_state_dict['position'])
        test_env.old_state_dict['position'] += 1

    if test_env.old_state_dict['position'] == 1:
        return_arr.append(((float(DATA.testdata['Last'].iloc[test_day + 1])
                            / float(DATA.testdata['Last'].iloc[test_day])) - 1)
                          * test_env.old_state_dict['position'])
        test_env.old_state_dict['position'] -= 1

    price_arr.append(DATA.testdata['Last'].iloc[test_day + 1])
    date_arr.append(DATA.testdata['Date'].iloc[test_day + 1])
    position_arr.append(test_env.old_state_dict['position'])
    summary_df = pd.DataFrame()
    summary_df['Date'] = date_arr
    summary_df['Last'] = price_arr
    summary_df['position'] = position_arr
    summary_df.to_csv('Summary.csv')
    sharpe_ratio = np.mean(return_arr)/float(np.std(return_arr))*math.sqrt(252)
    profit_loss = np.sum(return_arr)
    return profit_loss, sharpe_ratio

def train(DATA, TRAINMODEL, ENV, AGENT):
    '''method to train the model and calculate the learning curve.'''
    profits = []
    sharpe = []

    epsilon = TRAINMODEL.epsilon
    alpha = TRAINMODEL.alpha
    gamma = TRAINMODEL.gamma
    trialscounter = 0
    decaycounter = 1
    try:
        results = pd.read_csv(os.getcwd() + '\\' + 'profits_sharpe.csv')
    except:
        results = pd.DataFrame(columns=('profit', 'Sharpe'))

    if TRAINMODEL.timedecay == 1:
        TRAINMODEL.alpha, TRAINMODEL.gamma, TRAINMODEL.epsilon = 1, 1, 1

    for step in range(TRAINMODEL.trials):
        profit_loss_arr = []
        sharpe_arr = []
        print step
        if TRAINMODEL.timedecay == 1:
            epsilon = max((1 - (float(decaycounter
                                      / float(TRAINMODEL.trials))))
                          * TRAINMODEL.epsilon, 0)
            alpha = max((1 - (float(decaycounter
                                    / float(TRAINMODEL.trials))))
                        * TRAINMODEL.alpha, 0)
            gamma = max(((float(decaycounter / float(TRAINMODEL.trials)))) * TRAINMODEL.gamma, 0)
        decaycounter += 1

        startingpoint = random.randint(1, len(DATA.traindata) - TRAINMODEL.lengthofperiod)
        for day in range(startingpoint, startingpoint + TRAINMODEL.lengthofperiod - 1):
            ENV.newstate(DATA, TRAINMODEL, AGENT, epsilon, day)
            TRAINMODEL.update(ENV, TRAINMODEL, AGENT, DATA, gamma, alpha, trialscounter)
            ENV.old_state_dict = ENV.new_state_dict
            trialscounter += 1

        profit_loss, sharpe_ratio = test2(DATA)
        profit_loss_arr.append(profit_loss)
        sharpe_arr.append(sharpe_ratio)
        profits.append(np.sum(profit_loss_arr))
        sharpe.append(np.mean(sharpe_arr))
        results.loc[len(results) + 1] = [profits[-1], sharpe[-1]]
        print sharpe
        print len(TRAINMODEL.y_list)
        print(epsilon, gamma, alpha)
        pd.DataFrame(results).to_csv('profits_sharpe.csv', index=False)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for loop in range(1000):
        train(DATA, TRAINMODEL, ENV, AGENT)
