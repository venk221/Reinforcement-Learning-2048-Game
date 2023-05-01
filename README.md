# Deep Q-learning of the game 2048
Team members:
- Celso T. do Cabo
- Rushabh Kheni
- Venkatesh R. Mullur

## Environment information

The package `gym-2048` does not work with newer versions of Python. We have installed `python==3.7` in a new environment (I know version 3.10 does not work since it requires numpy 1.14)

Some other packages needed include `Pytorch`, `Matplotlib`, `gym` and `Numpy`

## Files

- `play_game.py` Runs the trained model for 1,000 times for statistical run, saves a file with the maximum value obtained in each run
- `play_game.ipynb` Runs the environment for player and for one model so that it can save the game for rendering
- `show_played_game.py` Renders an episode of the game using Tk (last reference)
- `dqn_model.py` has the neural network to obtain the action. Since the state is a 4x4 matrix there is a flatten in the beginning and then the network will find the best action
- `replay_memory.py` has a replay memory buffer implementation with functions to add experience to the buffer and sample experiences with some batch size from the buffer
- `agent_dqn.py` has the training loop for the DQN model
- `agent_ddqn.py` has the training loop for the double DQN model 
- `new_action.py` has the function to take an action from the neural network based on the observation (it is used in the play_game.py file)
- `plots.ipynb` Jupyter notebook for plotting the reward


## How to play the game

Initially, the model was trained using either the code agent_dqn.py or agent_ddqn.py, the trained model can be used in the play_game.py to generate a statistical run or be used as input for the play_game.ipynb where the episode will be saved in a new format so that the file show_played_game.py can render it. 

There is probably a need to change some of the paths to load or save the model. 

## Important links

https://pypi.org/project/gym-2048/

https://github.com/voice32/2048_RL

https://github.com/SergioIommi/DQN-2048

https://github.com/navjindervirdee/2048-deep-reinforcement-learning

https://tjwei.github.io/2048-NN/ - This link also has a repository, however, it's a complex model

https://github.com/FelipeMarcelino/2048-Gym - Rendering the episodes
