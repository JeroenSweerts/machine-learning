import os
import random
import math
from collections import Counter
import pandas as pd
import numpy as np
import TA
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor



class Data(object):
    '''This class deals with all the data needed for training and testing the model
       It reads the raw data with historical prices.
       It calculates some additional state variables.
       It splits the data into a training set and test set.
       It saves data for later use.'''

    def __init__(self, strfilename):
        '''strfilename is a string with the filename which contains the raw date
           arrtechnicalindicators is an array containing all the technical indicators
           that need to be added to the state'''
        self.filename = strfilename
        self.data = pd.DataFrame
        self.traindata = pd.DataFrame
        self.testdata = pd.DataFrame
        self.keys_list = ['CMO', 'CMO_slope', 'MACD', 'MACD_slope', \
                     'MOM', 'MOM_slope', 'ADXR', 'ADXR_slope', \
                     'RSI', 'RSI_slope', 'position']
        # we need this list because you can't predict
        # the order in which a dict loops through its elements
        self.statesactive_dict = {'CMO': 1, 'CMO_slope': 1, 'MACD': 1, 'MACD_slope': 1, \
                             'MOM': 1, 'MOM_slope': 1, 'ADXR': 1, 'ADXR_slope': 1, \
                             'RSI': 1, 'RSI_slope': 1, 'position': 1}
        self.datacontinuous = 0
        self.shift = 30
        self.normalize = 0

    def _open_file(self, strfilename):
        '''takes a string strfilename with the filename containing the raw data
        opens the csv file and reads it into a pandas dataframe which is returned
        by the method'''
        assert isinstance(strfilename, object)
        return pd.read_csv(strfilename)

    def preprocess_data(self):
        '''Opens the csv file with the raw data
           if the datacontinuous attribute is set to 0, then the data will be discretized
           We calculate the values of a range of technical indicators [ADXR,RSI,CMO,MACD,MOM]
           and their slopes.
           The shift attribute contains the number of days over which we calculate the slope.
           If the normalize attribute is set to 1, then we normalize the data.
           We only pick the state variables which have been activated in the statesactive_dict
           dictionary.
           We drop the first 75 rows of the data because most of the technical indicators
           use data over the past X days. This means that for those technical indicators
           the first X values will be NaN values.
           75% of the preprocessed data is used as training set and 25% as test set.
           The training set and test set are saved in cvs files for later use or analysis.'''

        _rawdata = self._open_file(self.filename)
        self.data = _rawdata

        if self.datacontinuous == 0:
            self.data['ADXR'] = TA.adxr(self.data['High'].as_matrix(), \
                self.data['Low'].as_matrix(), self.data['Last'].as_matrix(), 14, 'ASC')
            self.data['ADXR_slope'] = np.round(((self.data['ADXR'] - \
                self.data['ADXR'].shift(self.shift)) / self.shift).as_matrix(), 0)
            self.data['ADXR'] = np.round(self.data['ADXR'].as_matrix(), 0)

            self.data['RSI'] = TA.rsi(self.data['Last'].as_matrix(), 14, 'ASC')
            self.data['RSI_slope'] = np.round(((self.data['RSI'] - \
                self.data['RSI'].shift(self.shift)) / self.shift).as_matrix(), 0)
            self.data['RSI'] = np.round(self.data['RSI'].as_matrix(), 0)

            self.data['CMO'] = TA.cmo(self.data['Last'].as_matrix(), 45, 'ASC')
            self.data['CMO_slope'] = np.round(((self.data['CMO'] - \
                self.data['CMO'].shift(self.shift)) / self.shift).as_matrix(), 0)
            self.data['CMO'] = np.round(self.data['CMO'].as_matrix(), 0)

            self.data['MACD'] = TA.macd(self.data['Last'].as_matrix(), 12, 26, 2, 'ASC') * 100
            self.data['MACD_slope'] = np.round(((self.data['MACD'] - \
                self.data['MACD'].shift(self.shift)) / self.shift).as_matrix(), 0)
            self.data['MACD'] = np.round(self.data['MACD'].as_matrix(), 0)

            self.data['MOM'] = TA.mom(self.data['Last'].as_matrix(), 14, 'ASC')
            self.data['MOM_slope'] = np.round(((self.data['MOM'] - \
                self.data['MOM'].shift(self.shift)) / self.shift).as_matrix(), 0)
            self.data['MOM'] = np.round(self.data['MOM'].as_matrix(), 0)
        else:
            self.data['ADXR'] = TA.adxr(self.data['High'].as_matrix(), \
                self.data['Low'].as_matrix(), self.data['Last'].as_matrix(), 14, 'ASC')
            self.data['ADXR_slope'] = (self.data['ADXR'] - \
                self.data['ADXR'].shift(self.shift)) / self.shift

            self.data['RSI'] = TA.rsi(self.data['Last'].as_matrix(), 14, 'ASC')
            self.data['RSI_slope'] = (self.data['RSI'] - \
                self.data['RSI'].shift(self.shift)) / self.shift

            self.data['CMO'] = TA.cmo(self.data['Last'].as_matrix(), 45, 'ASC')
            self.data['CMO_slope'] = (self.data['CMO'] - \
                self.data['CMO'].shift(self.shift)) / self.shift

            self.data['MACD'] = TA.macd(self.data['Last'].as_matrix(), 12, 26, 2, 'ASC')
            self.data['MACD_slope'] = (self.data['MACD'] - \
                self.data['MACD'].shift(self.shift)) / self.shift

            self.data['MOM'] = TA.mom(self.data['Last'].as_matrix(), 14, 'ASC')
            self.data['MOM_slope'] = (self.data['MOM'] - \
                self.data['MOM'].shift(self.shift)) / self.shift
        self.data = self.data[75:]

        # normalize data
        if self.normalize == 1:
            for key in self.keys_list:
                if self.statesactive_dict[key] == 1 and key != 'position':
                    self.data[key] = (self.data[key] - self.data[key].mean()) \
                                     / (self.data[key].max() - self.data[key].min())

        # only pick the relevant state parameters, selected in the stateactive dictionary
        for key in self.keys_list:
            if self.statesactive_dict[key] == 0 and key != 'position':
                self.data = self.data.drop(key, 1)

        self.traindata = self.data.loc[:len(self.data) * 0.75, :]
        self.traindata = self.traindata.reset_index(drop=True)
        self.traindata.to_csv('trainData.csv')
        self.testdata = self.data.loc[len(self.data) * 0.75:, :]
        self.testdata = self.testdata.reset_index(drop=True)
        self.testdata.to_csv('testData.csv')

    def save_x_y_list(self, cols, x_list, y_list):
        '''saves the trained state action pairs for later use'''
        _df = pd.DataFrame(x_list, columns=cols)
        _df.to_csv(os.getcwd() + '\\' + 'X.csv')
        _df.to_csv('X.csv')

        _df = np.array(y_list)
        _df = pd.DataFrame(_df, columns=['Reward'])
        _df.to_csv(os.getcwd() + '\\' + 'Y.csv')

class RLModelTrain(object):
    def __init__(self, datainstance):
        """datainstance is an instance of the Data object.
        We need its key and statesactive_dict features for the initialization of X.
        X is a list containing all the state-action pairs that the learning agent
        is experiencing during the training phase and Y is a list with all the
        corresponding expected rewards.

        lengthofperiod is a variable indicating the size of the random subset taken out
        of the training dataset for each training cycle. For this model we used 365
        as the value for this variable.
        This means that for each training cycle (trial) we take a random subset of
        length 365 out of the entire training data set.

        trials is the number of times the model needs to loop through the
        training subdataset before the performance is tested on a random subset of
        the test dataset (with lenght=lengthofperiod).

        clf_all_list is a list with the 5 supervised learning models used as Q-functions
        in the Q-learning algorithm. A voting mechanism is applied to the predictions
        of each of the 5 Q-functions.

        alpha is the learning rate , gamma is the discount rate and epsilon the
        exploration rate of the Q-learning model.

        timedecay is a parameter where we can tell the model to apply time decay to
        alpha, gamma and epsilon parameters of the Q-learning algorithm.
        """
        self.lengthofperiod = 365
        self.trials = 100
        self.timedecay = 0
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.2
        self._init_x(datainstance)
        self._init_y()
        self._init_models()
        self.clf_win_list = [] # list of the models that won the vote
        self.voting = 'hard'
        self.y_list_size = 1000000 #the maximum size that X_list can take
        self.model_fit_days = 300 #defines after how many days the Q-functions need to re-fit

    def _init_x(self, datainstance):
        """X is a list of lists that contains the state-action pairs that the
        learning agent has gone through during the training phase. Each row
        (list) of X is a specific state-action pair.
         In the simplest case where only RSI is used as technical indicator,
         a state-action pair consists of the RSI value, the slope of the RSI
         curve over a period of which the length has been defined by the shift
         property of the Data class (in this case 30 days).
         Other elements of the state-action pair are the current trading
         position of the agent (-1 for short, 0 for neutral and +1 for long).
         Finally there are 3 possible action values (0 for buy, 1 for hold and
         2 for sell).
         This method will initialize X. It will first look for a file named
         X.csv containing the state-action pairs of the previous training
         session. If the X.csv file is not available, then X will be
         initialized. In this case the method enters the first row to
         X. It will initialize the state-action values in the following way:
         RSI=0; RSI slope=0, position=0 and action=hold (1).
         """
        try:
            _x = pd.read_csv(os.getcwd() + '\\' + 'X.csv')
            _x_headers = []  #contains the headers of the columns

            for key in datainstance.keys_list:
                if datainstance.statesactive_dict[key] == 1:
                    _x_headers.append(key)
            _x_headers.append('action')
            _x = _x.as_matrix(columns=_x_headers).tolist()
##            print(_x)
            print 'ok1'
        except:
            _x = []
            _row = []
            _x_headers = []  # contains the headers of the columns
            for key in datainstance.keys_list:
                if datainstance.statesactive_dict[key] == 1:
                    _row.append(0)
                    _x_headers.append(key)
            _x_headers.append('action')
            _row.append(1)  # start action is hold
            _x.append(_row)
        self.x_list = _x
        self.x_headers_list = _x_headers


    def _init_y(self):
        """Y contains the rewards linked to the state-action pairs in X.
        This method will initialize Y. It will first look for a file Y.csv
        which contains the Y values from a previous training session. This file
        allows the user to continue learning where the previous training session
        ended.
        If the file is not available, then a value 0 will be added to Y as the initial value."""
        try:
            _Y = pd.read_csv(os.getcwd() + '\\' + 'Y.csv')
            _Y = _Y['Reward'].as_matrix().tolist()
            print 'ok2'
        except:
            _Y = []
            _Y.append(0)
        self.y_list = _Y

    def _init_models(self):
        """This is the final step in initializing the model object before the
        training session can start.
        The training algorithm will use 5 supervised learning algorithms from
        the sklearn library.
        Each of these algorithms will predict the reward for a given
        state-action pair.
        The training of the models will be done by fitting the X_list and y_list
        data sets to the models.
        A voting mechanism will be applied on the 5 predicted rewards to decide
        the optimal action for the current state.
        """
        _clf1 = BaggingRegressor()
        _clf2 = AdaBoostRegressor()
        _clf3 = GradientBoostingRegressor()
        _clf4 = ExtraTreesRegressor()
        _clf5 = RandomForestRegressor()
        _clf_all = [_clf1, _clf2, _clf3, _clf4, _clf5]

        for _clf in _clf_all:
            _clf.fit(self.x_list, self.y_list)
        self.clf_all_list = _clf_all

    def voting_rule(self, action_subset, env_instance, agent_instance):
        """returns an integer between 0 and 2
        For each possible action, each of the 5 supervised learning models
        predict the expected reward. Then a voting rule is applied to select the
        best action for the current state.
        There are 2 possible voting rules the user can chose from:
        1. hard means a majority voting. The action with the most votes
        (selected by the majority of the supervised learning algorithms) wins
        2. soft means that the action with the highest average predicted reward
        wins
        """
        _best_action = None
        _test_state = env_instance.new_state_dict
        # the state dict on which we will try out the different actions
        if self.voting == 'hard':
            _voting_array = []
            #is a list of size 5 storing the preferred action for each Q-learning function
            _max_q_array = []
            #keeps for each element of the _voting_array the predicted maximum reward
            for _clf in self.clf_all_list:
                _reward_array = []
                #list with same size as action_subset containing the predicted
                #reward for each action
                for _action in action_subset:
                    _predictArr = []
                    # is a list containing the data of the _test_state dict in the right order
                    _test_state['action'] = agent_instance.actionlist.index(_action)
                    for _key in self.x_headers_list:
                        _predictArr.append(_test_state[_key])
                    _reward_array.append(_clf.predict([_predictArr]))
                _voting_array.append(action_subset[_reward_array.index(max(_reward_array))])
                #the action with the highest
                #reward is stored in the _voting_array
                _max_q_array.append(max(_reward_array))
            _best_action = Counter(_voting_array).most_common(1)[0][0]
            #best action is the one that occurs the most
            self.clf_win_list = []
            index_counter = 0
            _max_q = 0
            for action in _voting_array:
                if action == _best_action:
                    self.clf_win_list.append(self.clf_all_list[index_counter])
                    _max_q = _max_q + _max_q_array[index_counter]
                index_counter += 1
            _max_q = _max_q/float(len(self.clf_win_list))
            #the average of the predicted rewards of the winning votes
        else: #soft voting rule
            _voting_array = []
            # a list with same size as action_subset containing the average of the predicted rewards
            # by each of the 5 Q-functions
            for _action in action_subset:
                _reward_array = []
                #list with size 5 containing the predicted reward by each Q-learning function for
                # a specific action
                for _clf in self.clf_all_list:
                    _predictArr = []
                    # is a list containing the data of the _test_state dict in the right order
                    _test_state['action'] = agent_instance.actionlist.index(_action)
                    for _key in self.x_headers_list:
                        _predictArr.append(_test_state[_key])
                    _reward_array.append(_clf.predict([_predictArr]))
                _voting_array.append(sum(_reward_array)/float(len(self.clf_all_list)))
            _best_action = action_subset[_voting_array.index(max(_voting_array))]
            #_best_action is a string and needs to be converted to an index in the return statement
            _max_q = max(_voting_array)
        return agent_instance.actionlist.index(_best_action), _max_q

    def update(self, env_instance, rlmodel_instance, agent_instance,
               data_instance, gamma, alpha, trialscounter):
        '''udates the Q-function'''
        _predictArr = []
        for key in self.x_headers_list:
            _predictArr.append(env_instance.old_state_dict[key])

        _Q_s_a = 0 #V is the current Q(s,a) value
        _improved_estimate = 0

        if self.voting == 'hard':
            try:
                for _model in rlmodel_instance.clf_win_list:
                    _Q_s_a = _Q_s_a + _model.predict([_predictArr])
                _Q_s_a = _Q_s_a / float(len(rlmodel_instance.clf_win_list))
            except:
                _Q_s_a = rlmodel_instance.clf_all_list[0].predict([_predictArr])
        else:
            for _model in rlmodel_instance.clf_all_list:
                _Q_s_a = _Q_s_a + _model.predict([_predictArr])
            _Q_s_a = _Q_s_a / float(len(rlmodel_instance.clf_all_list))

        _improved_estimate = agent_instance.immediate_reward + (gamma*agent_instance.max_q)
        self.x_list.append(_predictArr)
        self.y_list.append((((1-alpha)*_Q_s_a) + (alpha*_improved_estimate))[0])

        if math.fmod(trialscounter, self.model_fit_days) == 0:
            for _model in self.clf_all_list:
                _model.fit(self.x_list, self.y_list)
            data_instance.save_x_y_list(self.x_headers_list, self.x_list, self.y_list)

        if len(self.y_list) > self.y_list_size:
            self.x_list = self.x_list[-self.y_list_size:]
            self.y_list = self.y_list[-self.y_list_size:]

class Agent(object):
    """The learning agent has 4 attributes. The actionlist contains all the allowed
     actions that the learning agent can take at a certain moment. This list can be
     reduced if there is a position constraint activated by the model.
     The bestaction property is the action with the highest expected reward at a
     given time.
     The max_q property is the expected reward of the bestaction.
     The immediate_reward is the immediate reward (over 1 time step) we get by
     taking the best_action """
    def __init__(self):
        self.actionlist = ['sell', 'hold', 'buy']
        self.bestaction = 1
        self.max_q = 0
        self.immediate_reward = 0

    def best_action(self, env_instance, data_instance, rlmodel_instance, epsilon, day):
        """returns an integer between 0 and 2 (0=buy, 1=hold, 2=sell).
        If the trading position is included as a state variable, then the list of
        allowed actions is reduced by the constraint that the learning agent can have
        a mximum short position of -1 and a maximum long position of +1. Any action
        would violate this constraint will not be allowed. All allowed actions are
        stored in the list action_subset.
        A random number between 0 and 1 is picked and if this number is below the value
        of the epsilon parameter, then the model will explore. This means a random action
        is picked. If the random value is above epsilon, then the model will exploit what
        it has learnt so far. This means that each of the 5 supervised learning algorithms
        will apply all the acquired knowledge stored in the X_list and y_list variables
        to predict the expected reward for each possible action in the action_subset.
        A voting mechanism decides which action is the _best_action."""
        _best_action = None
        action_subset = []
        if data_instance.statesactive_dict['position'] == 1:
            if env_instance.new_state_dict['position'] == -1:
                action_subset = self.actionlist[1:]
            if env_instance.new_state_dict['position'] == 0:
                action_subset = self.actionlist
            if env_instance.new_state_dict['position'] == 1:
                action_subset = self.actionlist[:-1]
        else:
            #when the position state has not been selected as a state variable
            action_subset = self.actionlist
        if random.random() < epsilon:
            _best_action = self.actionlist.index(random.choice(action_subset))
            _predict_arr = [] #translation of the new_state_dict dictionary into a list

##            for key in data_instance.keys_list:
            for _key in rlmodel_instance.x_headers_list:
##                if data_instance.statesactive_dict[key] == 1:
                _predict_arr.append(env_instance.new_state_dict[_key])
##            _predict_arr.append(_best_action)
            _predict_arr[-1] = _best_action

            _max_q = 0
            for _model in rlmodel_instance.clf_all_list:
                _max_q = _max_q + _model.predict([_predict_arr])
            _max_q = _max_q/float(len(rlmodel_instance.clf_all_list))
        else:
            _best_action, _max_q = rlmodel_instance.voting_rule(action_subset, env_instance, self)

        self.bestaction = _best_action
        self.max_q = _max_q
        _reward_multiplier = None #depending on the
        if env_instance.old_state_dict['action'] == 2:
            _reward_multiplier = 1

        if env_instance.old_state_dict['action'] == 0:
            _reward_multiplier = -1

        if env_instance.old_state_dict['action'] == 1:
            _reward_multiplier = env_instance.new_state_dict['position']
        self.immediate_reward = \
            ((float(data_instance.traindata['Last'].iloc[day])
              /float(data_instance.traindata['Last'].iloc[day-1]))-1) \
                    *_reward_multiplier*1000000
        return _best_action


class Environment(object):
    """Uses an instance of the Data class (datainstance) and an instance of the RLModel class
    (rlmodel_instance).
    Through the statesactive_dict property of the datainstance, we know which state variables are
    considered as state variables.
    The last row of the X-property of the rlmodel_instance gives us the old_state_dict values.
     """
    def __init__(self, datainstance, rlmodel_instance):
        """An environment object has 2 state attributes: the old_state_dict attribute contains the
         state-action pair of the previous time step and the new_state_dict attribute contains the
         state-action pairs for the next time step.
         The old_state_dict attribute is initialized by the _init_statedicts() method.
         The new_state_dict attribute is calculated by the newstate() method.
         Both new_state_dict old_state_dict are dictionaries"""
        self._init_statedicts(datainstance, rlmodel_instance)

    def _init_statedicts(self, datainstance, rlmodel_instance):
        """This method initializes the old_state_dict attribute by taking the last row of the
        initialized X attribute of RLModel instance.
        The new_state_dict attribute is also initialized with zeros and will be updated
        in the newstate() method."""
        _old_state_dict = {}
        _new_state_dict = {}

        for key in datainstance.keys_list:
            if datainstance.statesactive_dict[key] == 1:
                _old_state_dict[key] = \
                    rlmodel_instance.x_list[-1][rlmodel_instance.x_headers_list.index(key)]
                _new_state_dict[key] = 0
        _old_state_dict['action'] = \
            rlmodel_instance.x_list[-1][rlmodel_instance.x_headers_list.index('action')]
        _new_state_dict['action'] = 0
        self.old_state_dict = _old_state_dict
        self.new_state_dict = _new_state_dict


    def newstate(self, data_instance, rlmodel_instance, agent_instance, epsilon, day):
        """The new_state_dict dictionary is updated.
        The technical index values and slope come from the training data set.
        The new trading position (long, short, neutral) is derived from the position
        in the old_state_dict dictionary and the old_state_dict action.
        The action in the new_state_dict dictionary is calculated by the best_action()
        method in the Agent class.
        The best_action() method will try out all the allowed actions and pick the
        action with the highest expected future reward. 5 supervised learning algorithms
        will will calculate the expected future reward for each allowed action and a
        voting mechanism will decide on the optimal action for the new_state_dict.
        """

        for key in data_instance.keys_list:
            if data_instance.statesactive_dict[key] == 1 and key != 'position':
                self.new_state_dict[key] = data_instance.traindata[key].iloc[day]
        _old_action = agent_instance.actionlist[int(self.old_state_dict['action'])]


        #calculate new position
        if data_instance.statesactive_dict['position'] == 1:
            _old_position = self.old_state_dict['position']
            if _old_action == 'buy': self.new_state_dict['position'] = _old_position + 1
            if _old_action == 'sell': self.new_state_dict['position'] = _old_position - 1
            if _old_action == 'hold': self.new_state_dict['position'] = _old_position

        #calculate new best action
        self.new_state_dict['action'] = \
            agent_instance.best_action(self, data_instance, rlmodel_instance, epsilon, day)
