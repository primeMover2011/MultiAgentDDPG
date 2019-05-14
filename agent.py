import random
from model import Actor, Critic
from ounoise import OUNoise
import torch
import torch.optim as optim

GAMMA = 0.99  # discount factor
TAU = 0.01  # for soft update of target parameters
LR_ACTOR = 0.001  # learning rate of the actor
LR_CRITIC = 0.001  # learning rate of the critic

class Agent():

    def __init__(self, state_size, action_size, num_agents, device, gamma=GAMMA,
                 tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, random_seed=0):

        """
            Initialize an Agent object.
        :param state_size: size of state
        :param action_size: size of action
        :param num_agents: number of agents
        :param gamma: discount factor
        :param tau: factor for soft update of target parameters
        :param lr_actor: Learning rate of actor
        :param lr_critic: Learning rate of critic
        :param random_seed: Random seed
        :param device: cuda or cpu
        """

        self.device=device
        self.gamma = gamma
        self.tau=tau


        self.num_agents=num_agents

        self.state_size = state_size
        self.action_size = action_size
        self.full_state_size = state_size * num_agents
        self.full_action_size = action_size * num_agents
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, device, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, device, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.full_state_size, self.full_action_size, device=device, random_seed=random_seed).to(device)
        self.critic_target = Critic(self.full_state_size, self.full_action_size, device=device, random_seed=random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=0)

        self.noise = OUNoise(action_size, random_seed)

    def save_model(self, agent_number):
        torch.save(self.actor_local.state_dict(), f'models/checkpoint_actor_{agent_number}.pth')
        torch.save(self.critic_local.state_dict(), f'models/checkpoint_critic_{agent_number}.pth')

    def load_model(self, agent_number):
        checkpoint = torch.load(f'models/checkpoint_actor_{agent_number}.pth', map_location=torch.device('cpu'))
        self.actor_local.load_state_dict(checkpoint)

        checkpoint = torch.load(f'models/checkpoint_critic_{agent_number}.pth', map_location=torch.device('cpu'))
        self.critic_local.load_state_dict(checkpoint)


    def act(self, state, noise = 0., train = False):
        """Returns actions for given state as per current policy.
        :param state: state as seen from single agent
        """

        if train is True:
            self.actor_local.train()
        else:
            self.actor_local.eval()

        action = self.actor_local(state)
        if noise > 0:
            noise = torch.tensor(noise*self.noise.sample(), dtype=state.dtype, device=state.device)
        return action + noise

    def target_act(self, state, noise = 0.):
        #self.actor_target.eval()
        # convert to cpu() since noise is in cpu()
        self.actor_target.eval()
        action = self.actor_target(state).cpu()
        if noise > 0.:
            noise = torch.tensor(noise*self.noise.sample(), dtype=state.dtype, device=state.device)
        return action + noise

    def update_critic(self, rewards, dones, all_states, all_actions, all_next_states, all_next_actions):
        with torch.no_grad():
            Q_targets_next = self.critic_target(all_next_states, all_next_actions)
            # Compute Q targets for current states (y_i)
        q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local(all_states, all_actions)
        # critic_loss = F.mse_loss(q_expected, q_targets)
        critic_loss = ((q_expected - q_targets.detach()) ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, all_states, all_predicted_actions):
        """Update actor network

        :param all_states: all states
        :param all_predicted_actions: all predicted actions
        """
        actor_loss = -self.critic_local(all_states, all_predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

    def update_targets(self):
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def reset(self):
        self.noise.reset()