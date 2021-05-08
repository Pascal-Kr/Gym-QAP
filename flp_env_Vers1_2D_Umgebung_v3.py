import numpy as np
import gym
from gym import spaces
from numpy.random import default_rng
import pickle
import os
import math
import matplotlib.pyplot as plt
from PIL import Image
from gym_flp import rewards
from IPython.display import display, clear_output
import anytree
from anytree import Node, RenderTree, PreOrderIter, LevelOrderIter, LevelOrderGroupIter
    


class qapEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}  

    def __init__(self, mode=None, instance=None):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.DistanceMatrices, self.FlowMatrices = pickle.load(open(os.path.join(__location__,'discrete', 'qap_matrices.pkl'), 'rb'))
        self.transport_intensity = None
        self.instance = instance
        self.mode = mode
        
        
        while not (self.instance in self.DistanceMatrices.keys() or self.instance in self.FlowMatrices.keys() or self.instance in ['Neos-n6', 'Neos-n7', 'Brewery']):
            print('Available Problem Sets:', self.DistanceMatrices.keys())
            self.instance = input('Pick a problem:').strip()
     
        self.D = self.DistanceMatrices[self.instance]
        self.F = self.FlowMatrices[self.instance]
        
        # Determine problem size relevant for much stuff in here:
        self.n = len(self.D[0])
        self.x = math.ceil((math.sqrt(self.n)))
        self.y = math.ceil((math.sqrt(self.n)))
        self.size = int(self.x*self.y)
        self.observation_space_values=(self.x,self.y,3)
        self.max_steps = self.n - 1

        self.action_space = spaces.Discrete(int((self.n**2-self.n)*0.5)+1)
                
        # If you are using images as input, the input values must be in [0, 255] as the observation is normalized (dividing by 255 to have values in [0, 1]) when using CNN policies.       
        if self.mode == "rgb_array":
            self.observation_space = spaces.Box(low = 0, high = 255, shape=(1, self.n, 3), dtype = np.uint8) # Image representation
        elif self.mode == 'human':
            self.observation_space = spaces.Box(low=1, high = self.n, shape=(self.n,), dtype=np.float32)
        
        self.states = {}    # Create an empty dictonary where states and their respective reward will be stored for future reference
        self.actions = self.pairwiseExchange(self.n)
        
        # Initialize Environment with empty state and action
        self.action = None
        self.state = None
        self.internal_state = None
        
        #Initialize moving target to incredibly high value. To be updated if reward obtained is smaller. 
        
        self.movingTargetReward = np.inf 
        self.MHC = rewards.mhc.MHC()    # Create an instance of class MHC in module mhc.py from package rewards
    
    def reset(self):
        self.step_counter = 0  #Zählt die Anzahl an durchgeführten Aktionen
        state_1D = default_rng().choice(range(1,self.n+1), size=self.n, replace=False) 

        #MHC, self.TM = self.MHC.compute(self.D, self.F, state)
        self.internal_state = state_1D.copy()
        self.fromState = self.internal_state.copy()
        newState = self.fromState.copy()
        MHC, self.TM = self.MHC.compute(self.D, self.F, newState)
        state_2D = np.array(self.get_image())
        
        self.last_MHC = MHC
        
        self.movingTargetReward = MHC
        
        return state_1D, state_2D, MHC
    
    def step(self, action):
        # Create new State based on action 
        self.step_counter += 1 
        
        self.fromState = self.internal_state.copy()
        
        swap = self.actions[action]
        self.fromState[swap[0]-1], self.fromState[swap[1]-1] = self.fromState[swap[1]-1], self.fromState[swap[0]-1]
        
        
        newState = self.fromState.copy()
        best_state = self.fromState.copy()
        
        
        MHC, self.TM = self.MHC.compute(self.D, self.F, newState)
                
        
        reward = self.last_MHC - MHC 
        self.last_MHC = MHC
        
        self.movingTargetReward = MHC if MHC < self.movingTargetReward else self.movingTargetReward
        
        Actual_Minimum = self.movingTargetReward
            
        newState = np.array(self.get_image())
        self.state = newState.copy()
            
        self.internal_state = self.fromState.copy()
        
        if self.step_counter==self.max_steps:
            done = True
        else:
            done = False
        
        return newState, reward, done, {}, MHC, Actual_Minimum, best_state
        #return newState, reward, done
    
    def render(self, mode=None):
        if self.mode == 'rgb_array':
            #img = Image.fromarray(self.state, 'RGB')     
            img = self.get_image()

        
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    def close(self):
        pass
        
    def pairwiseExchange(self, x):
        actions = [(i,j) for i in range(1,x) for j in range(i+1,x+1) if not i==j]
        actions.append((1,1))
        return actions      
    
        # FOR CNN #
    def get_image(self):
        rgb = np.zeros((self.x,self.y,3), dtype=np.uint8)
            
        sources = np.sum(self.TM, axis = 1)
        sinks = np.sum(self.TM, axis = 0)
            
        R = np.array((self.fromState-np.min(self.fromState))/(np.max(self.fromState)-np.min(self.fromState))*255).astype(int)
        G = np.array((sources-np.min(sources))/(np.max(sources)-np.min(sources))*255).astype(int)
        B = np.array((sinks-np.min(sinks))/(np.max(sinks)-np.min(sinks))*255).astype(int)
                        
        k=0
        a=0
        Zeilen_ZAEHLER =0
        for s in range(len(self.fromState)):
            rgb[k][a] = [R[s], G[s], B[s]]
            a+=1
            if a>(self.x-1):
                Zeilen_ZAEHLER+=1
                k= Zeilen_ZAEHLER
                a=0
        
        newState = np.array(rgb)
        self.state = newState.copy()
        img = Image.fromarray(self.state, 'RGB')                     

        return img