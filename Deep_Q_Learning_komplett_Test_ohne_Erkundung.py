from gym_flp.envs.flp_env_2D_Umgebung_final import qapEnv
import tensorflow as tf
import tensorflow
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import time
import random
import os


#Für einen möglichen weiteren kompletten Durchgang mit vorhandenem Modell
LOAD_MODEL = "models/TEST-Differenz-vorher-nachher-128F-64Mini-0.001lr-lin-20k__1220.00max__661.80avg__218.00min__1621650949.model"
#LOAD_MODEL = None

DISCOUNT = 0.8
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)
env=qapEnv(mode ='rgb_array', instance='Neos-n7')

EPISODES = 100
MODEL_NAME = 'Testlauf-ohne-Erkunden'   #64 Filter und 2 Convolutional Layer

# Exploration settings
epsilon = 0

#  Stats settings
AGGREGATE_STATS_EVERY = 1
SHOW_PREVIEW = False

# For stats
ep_rewards = []
ep_Minimums = []
ep_Costs = []

# Gleiche Startbedingungen zum besseren Vergleich
random.seed(1)
np.random.seed(1)
tensorflow.random.set_seed(1)


# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')
    
    

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter 

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    # Added because of version
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

class DQNAgent:
    def __init__(self):

        # main model  # gets trained every step
        self.model = self.create_model()
        
        # Target model this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self):
        if LOAD_MODEL is not None:
            model = load_model(LOAD_MODEL)
        else:
            model = Sequential()

            model.add(Conv2D(128, (2, 2), input_shape=env.observation_space_values))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
            model.add(Activation('relu'))

            model.add(Conv2D(128, (2, 2)))
            model.add(Activation('relu'))
            
            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(128))

            model.add(Dense(len(env.actions), activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
            model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
 # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []  #basically the images 
        y = []  #basically the actions

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    




agent = DQNAgent()

# Iterate over episodes
#for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
for episode in range(1, EPISODES + 1):
    # Update tensorboard step every episode
    print('')
    agent.tensorboard.step = episode


    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()
    print(env.initial_MHC)

    # Reset flag and start iterating until episode ends
    done = False
    while not done:   #Solange Aktionen ausführen, bis Endzustand oder Anzahl an Schritten erreicht wurde

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, len(env.actions))

        new_state, reward, done, MHC = env.step(action)
        print(action)
        print(MHC)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward


        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    ep_Minimums.append(env.Actual_Minimum)
    ep_Costs.append(MHC)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        Minimum_costs = min(ep_Minimums[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        average_end_costs = sum(ep_Costs[-AGGREGATE_STATS_EVERY:])/len(ep_Costs[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(End_Costs=average_end_costs, Costs_Minimum=Minimum_costs, reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)


agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
print(env.best_state)


env.close()

