import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from models import MinecraftNet

class DQN:
    def __init__(self, state_dim, action_dim, goal_dim, lr,
                 H = 5, EPISILO = 0.9, hidden_size = 128, device = torch.device('cpu'),
                 norm_scale = 10):
        self.norm_scale = norm_scale

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.EPISILO = EPISILO
        self.norm_scale = norm_scale
        self.device = device
        
        self.eval_net = MinecraftNet(goal_dim, action_dim, H, hidden_size=hidden_size).to(device)
        self.target_net = MinecraftNet(goal_dim, action_dim, H, hidden_size=hidden_size).to(device)

        self.learn_step_counter = 0
        self.Q_NETWORK_ITERATION = 100
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    
    def select_action(self, state, goal, mask=None, goal_test = False):
        state = state.reshape(1, -1).float().to(self.device)
        goal = goal.reshape(1, -1).float().to(self.device)

        if mask is not None:
            mask = mask.view(1, self.action_dim).bool()

        
        action_value = torch.zeros((self.action_dim))

        if np.random.randn() <= self.EPISILO or goal_test:# greedy policy
            action_value = self.eval_net.forward(state, goal).view(1, self.action_dim)
            action_value = action_value + torch.randn_like(action_value) / self.norm_scale
            if mask is not None:
                action_value = action_value.masked_fill(mask, -10e7) 
            action = torch.max(action_value, 1)[1].item()
        else: # random policy
            if mask is not None:
                action_probs = torch.ones((self.action_dim)).to(self.device).masked_fill(mask.view(-1), 0)
            else:
                action_probs = torch.ones((self.action_dim)).to(self.device)
            action = torch.distributions.categorical.Categorical(probs=action_probs).sample().item()
        return action, action_value.view(-1)
    
    def update(self, buffer, n_iter, batch_size, masks = None, print_en = False):
        if self.learn_step_counter % self.Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1
        sum_loss = 0 
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action, reward, next_state, goal, gamma, done = buffer.sample(batch_size)
            
            # convert np arrays into tensors
            state = torch.FloatTensor(state).to(self.device).view(batch_size, self.state_dim)
            action = torch.FloatTensor(action).to(self.device).view(batch_size, 1).long()
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device).view(batch_size, self.state_dim)
            goal = torch.FloatTensor(goal).to(self.device).view(batch_size, 1)
            gamma = torch.FloatTensor(gamma).reshape((batch_size,1)).to(self.device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(self.device)



            q_eval = self.eval_net(state, goal).view(batch_size, self.action_dim).gather(1, action)
            q_next = self.eval_net(next_state, goal).view(batch_size, self.action_dim).detach()
            q_target_next = self.target_net(next_state, goal).view(batch_size, self.action_dim).detach()
            if masks is not None:
                action_masks = masks.view(self.goal_dim, self.action_dim)[goal.view(batch_size).long()]
                q_next = q_next.masked_fill(action_masks.view(batch_size, self.action_dim), -10e7)
                q_target_next = q_target_next.masked_fill(action_masks.view(batch_size, self.action_dim), -10e7)
            next_action = q_next.max(1)[1].view(batch_size, 1)
            q_target = reward + (1-done)*gamma * q_target_next.gather(1, next_action).view(batch_size, 1)
            loss = self.loss_func(q_eval, q_target)

            sum_loss += loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return (sum_loss / n_iter).item()
            
                
    def save(self, directory, name):
        torch.save(self.eval_net.state_dict(), '%s/%s_eval.pth' % (directory, name))
        torch.save(self.target_net.state_dict(), '%s/%s_target.pth' % (directory, name))
        torch.save({'optimizer': self.optimizer.state_dict()}, '%s/%s_state.pth' % (directory, name))
        
    def load(self, directory, name):
        self.eval_net.load_state_dict(torch.load('%s/%s_eval.pth' % (directory, name), map_location='cpu'))
        self.target_net.load_state_dict(torch.load('%s/%s_target.pth' % (directory, name), map_location='cpu'))  
        self.optimizer.load_state_dict(torch.load('%s/%s_state.pth' % (directory, name), map_location='cpu')['optimizer'])

    def pack_weights(self):
        return ({key:value.cpu() for key,value in self.eval_net.state_dict().items()}, \
                {key:value.cpu() for key,value in self.target_net.state_dict().items()})
    def unpack_weights(self, weights):
        self.eval_net.load_state_dict(weights[0])
        self.target_net.load_state_dict(weights[1])
        self.eval_net.to(self.device) 
        self.target_net.to(self.device) 
        
        
        
        
      
        
        
