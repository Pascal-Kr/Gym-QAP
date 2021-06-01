from gym_flp.envs.flp_env_2D_Umgebung_final import qapEnv
import tensorflow as tf
import tensorflow
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import time
import random
import os
from tqdm import tqdm


env = qapEnv(instance = 'Neos-n7', mode = 'rgb_array')
LOAD_MODEL = "models/TEST-Differenz-vorher-nachher-128F-64Mini-0.001lr-lin-20k__1220.00max__661.80avg__218.00min__1621650949.model"
model = load_model(LOAD_MODEL)
EPISODES = 10
end_costs=[]

for episode in range(1, EPISODES + 1):
    print('')
    done=False
    episode_reward=0
    s0 = env.reset()
    current_state = s0
    print(env.initial_MHC)
    while not done:
        #env.render()
        #Predictions = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
        Predictions_state = np.expand_dims(current_state, axis=0)/255
        Predictions = model.predict(Predictions_state)
        best_predict = np.argmax(Predictions)
        action=best_predict
        new_state, reward, done, MHC = env.step(action)
        print(action)
        print(MHC)
        current_state = new_state
    
    end_costs.append(MHC)


        


