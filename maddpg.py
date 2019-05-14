import numpy as np
import torch

from agent import Agent
from ReplayBuffer import ReplayBuffer

BUFFER_SIZE = int(1000000)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.95  # discount factor
UPDATE_EVERY = 2


class MADDPGAgent():

    def __init__(self, state_size, action_size, num_agents, random_seed, device,
                 buffer_size = BUFFER_SIZE, batch_size = BATCH_SIZE,
                 gamma = GAMMA, update_every = UPDATE_EVERY, tau=0.01, lr_actor=0.001, lr_critic=0.001):
        """Initialize Agent objects

        :param state_size: state size per agent
        :param action_size: action size per agent
        :param num_agents: number of agents
        :param random_seed: random seed
        :param buffer_size: size of replay buffer
        :param batch_size: size of batches drawn from replay buffer
        :param gamma: discount factor
        :param update_every: after how many steps to update the models
        """

        self.device = device
        self.batch_size=batch_size
        self.losses = []
        self.state_size = state_size
        self.action_size = action_size
        # Initialize the agents
        self.num_agents = num_agents
        self.agents = [Agent(state_size=state_size, action_size=action_size,
                             num_agents=num_agents, random_seed=random_seed,
                             gamma=gamma, tau=tau, device=device,
                             lr_actor=lr_actor, lr_critic=lr_critic) for _ in range(num_agents)]



        # Replay memory
        self.memory = ReplayBuffer(buffer_size=buffer_size,batch_size=batch_size, random_seed=random_seed, device=device)
        self.gamma = gamma
        self.update_every = update_every
        # Time steps for UPDATE EVERY
        self.time_step = 0

    def act(self, states, noise = 0., train=False):
        """Agents act with actor_local"""
        states = torch.from_numpy(states).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            actions = [agent.act(states[:,i], noise = noise, train = False) for i, agent in enumerate(self.agents)]
            actions = torch.stack(actions).transpose(1,0)
            actions = np.vstack([action.cpu().numpy() for action in actions])

        return actions

    def step(self, states, actions, rewards, next_states, dones, learn=True):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        self.time_step += 1

        # Learn, if enough samples are available in memory
        if self.time_step % self.update_every == 0:
            if learn is True and len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def save_models(self):
        for i, agent in enumerate(self.agents):
            agent.save_model(i)
    def load_models(self):
        for i, agent in enumerate(self.agents):
            agent.load_model(i)



    def reset(self):
        for agent in self.agents:
            agent.reset()

    def learn(self, experiences, GAMMA):
        states, actions, rewards, next_states, dones = experiences

        # next actions as input for critic
        target_next_actions = [agent.target_act(next_states[:, agent_number]) for agent_number, agent in enumerate(self.agents)]
        target_next_actions = torch.stack(target_next_actions).transpose(1, 0).contiguous()
        target_next_actions = self.flatten(target_next_actions).to(self.device)

        predicted_actions_t = [agent.act(states[:, agent_number], train=True) for agent_number, agent in enumerate(self.agents)]
        predicted_actions_t = torch.stack(predicted_actions_t).transpose(1, 0).contiguous()
        predicted_actions = predicted_actions_t.to(self.device)

        flat_states = self.flatten(states)
        flat_actions = self.flatten(actions)
        flat_next_states = self.flatten(next_states)



        for agent_number, agent in enumerate(self.agents):

            agent.update_critic(rewards=rewards[:, agent_number].unsqueeze(-1),
                                dones=dones[:, agent_number].unsqueeze(-1),
                                all_states=flat_states,
                                all_actions=flat_actions,
                                all_next_states=flat_next_states,
                                all_next_actions=target_next_actions)

            predicted_actions_for_agent = predicted_actions.detach()
            predicted_actions_for_agent[:, agent_number] = predicted_actions[:, agent_number]
            predicted_actions_for_agent = self.flatten(predicted_actions_for_agent)

            agent.update_actor(all_states=flat_states, all_predicted_actions=predicted_actions_for_agent)

            agent.update_targets()

#            actor_loss, critic_loss = agent.learn(rewards=rewards[:,agent_number].unsqueeze(-1),
#                        dones = dones[:, agent_number].unsqueeze(-1),
#                        all_states=flat_states, all_actions=flat_actions,
#                        all_next_states=flat_next_states, all_next_actions=target_next_actions,
#                        all_predicted_actions=predicted_actions_for_agent)
            #self.losses.append([actor_loss, critic_loss])

    def flatten(self, tensor):
        return tensor.view(tensor.shape[0], tensor.shape[1] * tensor.shape[2])