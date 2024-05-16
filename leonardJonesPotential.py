import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym

epsilon = 1.51e-22
sigma = 0.263

def lennard_jones_potential(x, epsilon, sigma):
    return 4 * epsilon * ((sigma / x) ** 12 - (sigma / x) ** 6)


class PotentialFunctionEnv():
    def __init__(self):
        self.action_space = 3
        self.low = np.float32(np.array([lennard_jones_potential(0.27, epsilon, sigma)]))
        self.high = np.float32(np.array([lennard_jones_potential(0.7, epsilon, sigma)]))
        self.state = random.uniform(0.25, 0.7)
        self.time = 100

    def step(self, action, prev_state):
        if action == 0:  # sola hareket
            actionValue = -0.01
        elif action == 1:  # sağa hareket
            actionValue = 0.01
        else:
            actionValue = 0.0

        self.state += actionValue
        self.time -= 1

        delta_potential = lennard_jones_potential(self.state, epsilon, sigma) - lennard_jones_potential(prev_state, epsilon, sigma)
        if delta_potential < 0:
            reward = 1
        else:
            reward = -1

        if self.time <= 0:
            done = True
        else:
            done = False

        return np.array([self.state]), reward, done, {}

    def reset(self):
        self.state = random.uniform(0.25, 0.7)
        self.time = 100
        return np.array([self.state])

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]



env = PotentialFunctionEnv()
input_size = env.low.shape[0]
output_size = env.action_space
model = DQN(input_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

episode_start_states = []
episode_end_states = []
memory = ReplayMemory(capacity=7000)
num_episodes = 20
for episode in range(num_episodes):
    state = env.reset()
    episode_start_states.append(state)
    done = False
    score = 0
    num_initial_random_episodes = 1



    while not done:
        if episode < num_initial_random_episodes:
            # Başlangıçta rastgele bir aksiyon seç
            action = random.randint(0, env.action_space - 1)
        else:
            # DQN modelini kullanarak aksiyon seç
            state = torch.tensor(state, dtype=torch.float32)
            q_values = model.forward(state)
            action = torch.argmax(q_values).item()

        next_state, reward, done, _ = env.step(action, prev_state=state)
        score += reward
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)



        transition = (state, action, reward, next_state)
        memory.push(transition)

        state = next_state


        batch_size = 32
        gamma = 0.10
        b = 1
        a = 1/b
        gamma -= 0.005*a
        b += 0.1
        eps = 0.1
        if len(memory) > batch_size:
            transitions = memory.sample(batch_size)
            actions = [torch.tensor(transition[1], dtype=torch.int64) for transition in transitions]
            states = [torch.tensor(transition[0], dtype=torch.float32) for transition in transitions]
            next_states = [next_state for _ in range(batch_size)]
            rewards = [torch.tensor(transition[2], dtype=torch.float32) for transition in transitions]

            state_batch = torch.stack(states)


            action_batch = torch.tensor(actions, dtype=torch.int64)
            reward_batch = torch.tensor(rewards, dtype=torch.float32)
            next_state_batch = torch.tensor(next_state, dtype=torch.float32)
            next_state_batch = next_state_batch.unsqueeze(1)


            current_q_values = model(state_batch).gather(1, action_batch.unsqueeze(-1))

            reward_batch = reward_batch.unsqueeze(1)
            next_q_values = model(next_state_batch).max(1)[0].unsqueeze(1)
            target_q_values = reward_batch.unsqueeze(1) + gamma * next_q_values


            loss = loss_fn(current_q_values, target_q_values.detach())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Episode {}: Score: {}".format(episode, score))

