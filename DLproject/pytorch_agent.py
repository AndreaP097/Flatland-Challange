import torch
import numpy as np

from torch.nn.functional import relu
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from collections import deque


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class DuelingDQN(torch.nn.Module):

    def __init__(self):

        super(DuelingDQN, self).__init__()

        self.dense1 = torch.nn.Linear(231, 128)
        self.dense2 = torch.nn.Linear(128, 128)

        self.pre_v1 = torch.nn.Linear(128, 128)
        self.pre_v2 = torch.nn.Linear(128, 128)
        self.v = torch.nn.Linear(128, 1)

        self.pre_a1 = torch.nn.Linear(128, 128)
        self.pre_a2 = torch.nn.Linear(128, 128)
        self.a = torch.nn.Linear(128, 5)

    def forward(self, x):

        x = relu(self.dense1(x))
        x = relu(self.dense2(x))

        v = relu(self.pre_v1(x))
        v = relu(self.pre_v2(v))
        v = self.v(v)

        # advantage calculation
        a = relu(self.pre_a1(x))
        a = relu(self.pre_a2(a))
        a = self.a(a)

        Q = v + (a - a.mean())

        return Q

class experience_replay:

    def __init__(self, buffer_size):
        self.buffer = deque(maxlen = buffer_size)   
        

    def add_experience(self, state, action, reward, next_state, done):
        exp = (np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.buffer.append(exp)

    def remove_batch(self, batch_size):
        batch_idx = np.random.choice(len(self.buffer), batch_size, replace = False)
        batch = np.array(self.buffer, dtype='object')[batch_idx.astype(int)]
        return batch

class agent():

    def __init__(self, learning_rate, gamma, batch_size):

        self.name = 'ma5'
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = 1.
        self.min_eps = 0.001
        self.batch_size = batch_size
        self.buffer_len = 10000
        self.replay = experience_replay(self.buffer_len)
        self.model = DuelingDQN().to(device)
        self.model_target = DuelingDQN().to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def act(self, state):

        if np.random.rand() <= self.eps:
            action = np.random.choice(range(5))
        
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                actions = self.model(state)
            action = np.argmax(actions.cpu().numpy())
             
        return action 


    def train(self):

        if len(self.replay.buffer) < self.batch_size:
            return

        batch = self.replay.remove_batch(self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for exp in range(self.batch_size):
            
            state, action, reward, next_state, done = batch[exp]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = torch.from_numpy(np.array(states).reshape(self.batch_size, 231)).float().to(device)    
        actions = torch.from_numpy(np.array(actions).reshape(self.batch_size, 1)).long().to(device)
        rewards = torch.from_numpy(np.array(rewards).reshape(self.batch_size, 1)).float().to(device)
        next_states = torch.from_numpy(np.array(next_states).reshape(self.batch_size, 231)).float().to(device)   
        dones = torch.from_numpy(np.array(dones).reshape(self.batch_size, 1)).float().to(device)
        
        Q_predictions = self.model(states).gather(1, actions)   # selects along axis=1 the Q-vals of the actions we've taken

        # max(1) -> named tuple (values, indices) of max values along axis one, we're interested in the values
        # unsqueeze(-1) adds a dimension of size one in position -1, similar to reshape(.., 1) 
        max_q_val = self.model_target(next_states).detach().max(1)[0].unsqueeze(-1)

        Q_targets = rewards + (self.gamma * max_q_val * (1 - dones))

        loss_fn = mse_loss(Q_predictions, Q_targets)
        self.optimizer.zero_grad()
        loss_fn.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=10)
        self.optimizer.step()

        self.copy_weights()


    def copy_weights(self):
        tau = 0.0001
        for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_eps(self):
        if self.eps > self.min_eps:
            self.eps = self.eps * 0.998

    def save_model(self):
        torch.save(self.model.state_dict(), '.vscode/DLproject/trained_models/' + self.name + '.local')
        torch.save(self.model_target.state_dict(), '.vscode/DLproject/trained_models/' + self.name + '.target')

    def load_model(self):
        self.model.load_state_dict(torch.load('.vscode/DLproject/trained_models/' + self.name + '.local'))
        self.model_target.load_state_dict(torch.load('.vscode/DLproject/trained_models/' + self.name + '.target'))
    