import gym
import universe  # register the universe environments
import LSTM
import ActivationFunctions
import numpy as np
import ConvolutionalLayer

env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()

# 1024 x 768 x 3
# myLSTM = LSTM.LSTM(2359296, 2, ActivationFunctions.ActivationFunctions.apply_logistic_sigmoid)

myCNN = ConvolutionalLayer.ConvolutionalLayer(25, 8, 1, 0, 1024, 768)

previous_view = None

while True:
    choice = 0
    if previous_view != None:
        # choice = np.argmax(myLSTM.apply(previous_view))
        myCNN.apply(previous_view)

    key = ''
    if choice == 0:
        key = 'ArrowUp'
    elif choice == 1:
        key = 'ArrowRight'
    elif choice == 2:
        key = 'ArrowLeft'
    elif choice == 3:
        key = 'ArrowDown'

    action_n = [[('KeyEvent', key, True)] for ob in observation_n]  # your agent here
    observation_n, reward_n, done_n, info = env.step(action_n)

    if observation_n != [None]:
        previous_view = observation_n[0]['vision']
    else:
        previous_view = None

    env.render()
