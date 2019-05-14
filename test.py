import numpy as np
import torch
from maddpg import MADDPGAgent
from unityagents import UnityEnvironment



def test_agent(env, brain_name, agent, device, real_time=False):
    env_info = env.reset(train_mode=not real_time)[brain_name]
    states = env_info.vector_observations
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)
    game = 0
    while True:
        action = agent.act(states, noise=0., train=True)
        env_info = env.step(action)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += rewards
        states = next_states
        if np.any(dones):
            game += 1
        if game > 10:
            break
    return np.mean(scores)

def main():

    device   = torch.device("cpu")
    env = UnityEnvironment(file_name='tennis/tennis', base_port=64739)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = MADDPGAgent(state_size=state_size, action_size=action_size, num_agents=num_agents,
                              random_seed=0, buffer_size=100, device=device,
                              batch_size=100, update_every=1, tau=0.001, lr_actor=0.003,
                              lr_critic=0.005)
    agent.load_models()
    test_agent(env, brain_name, agent, device, real_time=True)


if __name__ == "__main__":
    main()
