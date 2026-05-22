import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from abc import abstractmethod

from race_sim.agent import Agent
from race_sim.types import Action

# ==========================================
# 1. ARQUITECTURA DE LA RED NEURONAL
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================
# 2. BUFFER DE MEMORIA
# ==========================================
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# ==========================================
# 3. EL AGENTE DQN (Clase Base Abstracta)
# ==========================================
class BaseDQNAgent(Agent):
    def __init__(self, state_size, action_size, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q-Network y Target Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)
        
        self.memory = ReplayBuffer(int(1e5), 64)
        self.t_step = 0
        self.eps = 1.0 # Lo guardamos como atributo para acceder a él fácilmente
        
    # =====================================================================
    # RETO: DISCRETIZACIÓN
    # =====================================================================
    @abstractmethod
    def index_to_action(self, action_idx: int) -> Action:
        """
        Convierte el índice discreto elegido por la red (0 a action_size-1)
        en un objeto Action continuo del simulador.
        """
        pass

    # =====================================================================
    # MÉTODOS DEL CICLO DE VIDA DEL AGENTE
    # =====================================================================
    def act(self, obs: np.ndarray) -> Action:
        """Devuelve un objeto Action evaluando el estado."""
        state_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        # Epsilon-greedy para elegir el índice de la acción
        if random.random() > self.eps:
            action_idx = np.argmax(action_values.cpu().data.numpy())
        else:
            action_idx = random.choice(np.arange(self.action_size))
            
        # Convertimos el índice numérico en la Acción física
        self.last_action_idx = int(action_idx) 
        return self.index_to_action(self.last_action_idx)

    def train(self, state, action_idx, reward, next_state, done):
        self.memory.add(state, action_idx, reward, next_state, done)
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, 0.99)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update de la red target
        tau = 1e-3
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = random.seed(seed)
            torch.manual_seed(seed)
            
    def save(self, filepath: str):
        torch.save(self.qnetwork_local.state_dict(), filepath)
        
    def load(self, filepath: str):
        self.qnetwork_local.load_state_dict(torch.load(filepath))
        self.eps = 0.0 # Al cargar un modelo entrenado, desactivamos la exploración aleatoria

