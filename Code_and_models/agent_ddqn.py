import random
import numpy as np
from collections import deque
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from dqn_model import DQN
import gym_2048
import gym

# Test
#torch.manual_seed()
#np.random.seed()
#random.seed()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU = torch.device('cpu')
print(device)

batch_size = 32
learning_rate = 5e-5
no_op_max = 30
gamma = 0.99
min_replay = 500001 
epsilon_initial = 1
epsilon_final = 0.025
buffer_size = 50000
target_update_freq = 1000
decay_step = (epsilon_initial-epsilon_initial)/min_replay
replayer_buffer = deque(maxlen=buffer_size)
reward_buffer = deque([0.0], maxlen=30)
episode_reward = 0.0

online_net = DQN(size_state=4, num_actions=4)
target_net = DQN(size_state=4, num_actions=4)
target_net.load_state_dict(online_net.state_dict())
env = gym.make('2048-v0')
optim = torch.optim.Adam(online_net.parameters(), lr=learning_rate)    

def make_action(online_net, observation, device):
    CPU = torch.device('cpu')
    obs_t = torch.as_tensor(observation.copy(), dtype=torch.float32)
    q_values = (obs_t.unsqueeze(0))
    online_net.to(device)
    action_t = online_net(q_values.to(device))
    action = np.argmax(action_t.to(CPU).detach().numpy())
    return action 
    
online_net.to(device)
target_net.to(device)

obs = env.reset()
episode_reward = 0
moves = 0
step = 0
done = False
while not done or step < 20:
    action = env.np_random.choice(range(4), 1).item()
    new_state, reward, done, info = env.step(action)
    moves += 1
    transition = (obs, action, reward, done, new_state)
    replayer_buffer.append(transition)
    obs = new_state


    print('Next Action: "{}"\n\nReward: {}'.format(
      gym_2048.Base2048Env.ACTION_STRING[action], reward))
    env.render()

    if done:
        obs = env.reset()
        step += 1
        print('Episode ', step)

avg = []
max_values_epoch = []
len_epis = 0
frame = 0
best_reward = 0
for step in range(min_replay):
    epsilon = np.interp(step, [0, 20000], [epsilon_initial, epsilon_final])
    num_episodes = 0
    #self.reward_buffer.clear()
    #self.replayer_buffer = []
    len_append = []
    
    while num_episodes < 1:
        
        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            action = env.np_random.choice(range(4), 1).item()
        else:
            action = make_action(online_net, obs, device)

        new_state, reward, done, info = env.step(action)
        transition = (obs, action, reward, done, new_state)
        replayer_buffer.append(transition)
        obs = new_state

        episode_reward += reward
        len_epis += 1
        frame += 1
        max_value = obs.max()
        if done:
            obs = env.reset()

            reward_buffer.append(episode_reward)
            episode_reward = 0
            num_episodes += 1
            
            len_append.append(len_epis)
            max_values_epoch.append(max_value)
            len_epis = 0

        len_avg = np.mean(len_append)            

    # Starting gradient step

        transitions = random.sample(replayer_buffer, batch_size)

        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_states = np.asarray([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)
        new_states_t = torch.as_tensor(new_states, dtype=torch.float32)
        #print(new_states_t.shape)
        # Compute targets

        with torch.no_grad():
            target_online_q_values = online_net(new_states_t.to(device))
            best_q_indices = target_online_q_values.argmax(dim=1, keepdim=True)
        
            target_q_values = target_net(new_states_t.to(device))

            max_target_q_values = torch.gather(input=target_q_values, dim=1, index=best_q_indices)

            targets = rewards_t + gamma * (1 - dones_t) * max_target_q_values

        # Compute the loss
        
        q_values = online_net(obses_t.to(device))

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = F.mse_loss(action_q_values, targets)

        # Gradient Descent
        optim.zero_grad()
        loss.backward()
        optim.step()

    # Update Target network
    if step % target_update_freq == 0:
        target_net.load_state_dict(online_net.state_dict())

    avg.append(np.mean(reward_buffer))
    
    if np.mean(reward_buffer) > best_reward:
        best_reward = np.mean(reward_buffer)
        torch.save(online_net.state_dict(),'./Double_dqn_best_model.pth')
        print('Best reward: ', best_reward)

    # Logging
    if step % 10 ==0:
        print()
        print('Step ', step)
        print('Avg reward ', np.mean(reward_buffer))
        print('Epsilon ', epsilon)
        print('Random sample ', rnd_sample)
        print('Length episodes ', len_avg)
        print('Reward buffer ', len(reward_buffer))
        print('Episodes buffer ',len(replayer_buffer))
        print("Frame ", f"{frame:,}")
        print('Max value ', max_value)

    if step % 1000 == 0:
        torch.save(target_net.state_dict(),'./Double_dqn_trained_model.pth')
        torch.save(online_net.state_dict(),'./Double_dqn_trained_model_1.pth')
        with open('./Double_dqn_max_value.npy', 'wb') as f:
            np.save(f, max_values_epoch)
        with open('./Double_dqn_reward.npy', 'wb') as f:
            np.save(f, avg)

torch.save(target_net.state_dict(),'./Double_dqn_trained_model.pth')
torch.save(online_net.state_dict(),'./Double_dqn_trained_model_1.pth')