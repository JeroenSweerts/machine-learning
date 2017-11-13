import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import timeit
import csv
import pandas as pd
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.next_waypoint = random.choice(env.valid_actions)
        self.OldState={'nextWaypoint': random.choice(self.env.valid_actions[1:]),'forwardlegal':random.choice(['yes','no']),'leftlegal':random.choice(['yes','no']),'rightlegal':random.choice(['yes','no']),'deadline':0}
        self.OldAction=random.choice(self.env.valid_actions)
        self.penalty=0
        self.numsteps=1

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        

    def update(self, t):
        # Gather inputs
        global success
        global Qtable
        global action
        global rewardArr
##        global mode

        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self) #{'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        deadline = self.env.get_deadline(self)

        try:
            self.OldAction=action
            self.OldState=self.state
        except:
            pass

        # TODO: Update state
        forwardlegal,leftlegal,rightlegal=self.calclLegal(inputs)
        #deadline=0
        if(deadline<10):
            deadlineState=1
        else:
            deadlineState=0

        self.state={'nextWaypoint': self.next_waypoint,'forwardlegal':forwardlegal,'leftlegal':leftlegal,'rightlegal':rightlegal,'deadline':deadlineState}

        # TODO: Select action according to your policy
        rewardArray=[]
        for actionItem in [None,'forward','left','right']:
            rewardArray.append(float(Qtable[str(((self.state['nextWaypoint'],self.state['forwardlegal'],self.state['leftlegal'],self.state['rightlegal'],self.state['deadline']), actionItem))]))
        BestAction=[None,'forward','left','right'][rewardArray.index(max(rewardArray))]

        if (mode=='train'):
            try:
##                if (random.random() < self.epsilon):
                maxeps=max((1-(float(self.env.trialnum/float(self.trials))))*self.epsilon,0)
                if (random.random() < maxeps):
                    action=random.choice([None,'forward','left','right'])
                else:
                    action=BestAction
            except:
                action=random.choice([None,'forward','left','right'])
        else:
            action=BestAction

        # Execute action and get reward
        reward,arrived = self.env.act(self, action)
        success=success+arrived
        rewardArr.append(reward)
        if reward < 0: #Assign penalty if reward is negative
            self.penalty+= 1
        self.numsteps+=1
        

        # TODO: Learn policy based on state, action, reward
        if (mode=='train'):
            try:
                maxalpha=max((1-(float(self.env.trialnum/float(self.trials))))*self.alpha,0)
                maxgamma=max((1-(float(self.env.trialnum/float(self.trials))))*self.gamma,0)

                V=float(Qtable[str(((self.OldState['nextWaypoint'],self.OldState['forwardlegal'],self.OldState['leftlegal'],self.OldState['rightlegal'],self.OldState['deadline']), self.OldAction))])
                X=reward+(maxgamma*float(Qtable[str(((self.state['nextWaypoint'],self.state['forwardlegal'],self.state['leftlegal'],self.state['rightlegal'],self.state['deadline']), action))]))
                Qtable[str(((self.OldState['nextWaypoint'],self.OldState['forwardlegal'],self.OldState['leftlegal'],self.OldState['rightlegal'],self.OldState['deadline']), self.OldAction))]=((1-maxalpha)*V)+(maxalpha*X)
            except:
                pass

##        print(('T= ',self.env.trialnum))

    def calclLegal(self,inputs): #calculates for each of the possible actions if they are legal or not in the next time step
        forwardlegal='no'
        leftlegal='no'
        rightlegal='no'

        if inputs['light']=='green':
            forwardlegal='yes'
            if inputs['oncoming']=='right' or inputs['oncoming']=='forward':
                leftlegal='no'
            else:
                leftlegal=='yes'

            rightlegal=='yes'
        else:
            forwardlegal='no'
            leftlegal=='no'
            if inputs['left']=='forward':
                rightlegal='no'
            else:
                rightlegal='yes'

        return forwardlegal,leftlegal,rightlegal

    #Create a list of all the possible states
    def initStateSpace(self):
        states = []
        for nextWaypoint in Environment.valid_actions[0:]:
            for forwardlegal in ['yes','no']:
                for leftlegal in ['yes','no']:
                    for rightlegal in ['yes','no']:
                        for deadline in range(0,2):
                            states.append((nextWaypoint,forwardlegal,leftlegal,rightlegal,deadline))
        return states

    #Create a dictionary (key-value pairs) of all possible state-actions and their values
    #This creates our Q-value look up table
    def initStateActions(self,states):
        av = {}
        for state in states:
            av[str((state, None))] = 0.0
            av[str((state, 'forward'))] = 0.0
            av[str((state, 'left'))] = 0.0
            av[str((state, 'right'))] = 0.0
        return av

    def setAlpha(self,alpha):
        self.alpha=alpha

    def setGamma(self,gamma):
        self.gamma=gamma

    def setEpsilon(self,epsilon):
        self.epsilon=epsilon

    def setMode(self,mode):
        self.mode=mode

    def setTrials(self,trials):
        self.trials=trials




def run(alpha,gamma,epsilon,mode,trials):
    """Run the agent for a finite number of trials."""
    global Qtable #table representing the Q-function. This table contains the value for each possible state/action pair

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    e.trialnum=0
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    a.setAlpha(alpha)
    a.setGamma(gamma)
    a.setEpsilon(epsilon)
    a.setMode(mode)
    a.setTrials(trials)
    # check if there is a trained Qtable available
    try:
        with open('Qtable.csv') as csv_file:
            reader = csv.reader(csv_file)
            Qtable = dict(reader)
    except:
        Qtable=a.initStateActions(a.initStateSpace()) #we create a table that contains the values for all possible state/action pairs


    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    with open('Qtable.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in Qtable.items():
           writer.writerow([key, value])

    print('penalty rate: ',a.penalty/float(a.numsteps))
    

if __name__ == '__main__':
    global success
    global rewardArr


    successArr=[]
    epsilonArr=[]
    gammaArr=[]
    alphaArr=[]

    epsilon= 1
    gamma=1
    alpha=1


    #TRAIN
##    for alpha in [0.27,0.28,0.29,0.3,0.31,0.32,0.33]:
    mode='train'
    rewardArr=[]
    success=0
    trials=0
    run(alpha,gamma,epsilon,mode,trials)

#   DRIVE
##    for alpha in [0.27,0.28,0.29,0.3,0.31,0.32,0.33]:
    mode='drive'
    rewardArr=[]
    success=0
    trials=1000
    run(alpha,gamma,epsilon,mode,trials)
    print('avg reward: ',sum(rewardArr)/trials)
    print('success rate: ',success/float(trials),success)
    successArr.append(success/float(trials))
    epsilonArr.append(epsilon)
    gammaArr.append(gamma)
    alphaArr.append(alpha)

    resultspd=pd.DataFrame(successArr,columns=['succes rate'])
    resultspd['epsilon']=epsilonArr
    resultspd['gamma']=gammaArr
    resultspd['alpha']=alphaArr
    resultspd.to_csv('resultsAlpha.csv')
    print(resultspd)
