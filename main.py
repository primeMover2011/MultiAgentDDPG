from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import os
from maddpg import MADDPGAgent
import matplotlib.pyplot as plt
import datetime
import torch

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def plot(scores=[], ylabels=["Scores"], xlabel="Episode #", title="", text=""):
    fig, ax = plt.subplots()

    for score, label in zip(scores, ylabels):
        ax.plot(np.arange(len(score)), score, label=label)
    ax.grid()
    ax.legend(loc='upper left', shadow=False, fontsize='x-large')
    fig.tight_layout()
    fig.savefig(f"plot_{datetime.datetime.now().isoformat().replace(':', '')}.png")
    plt.show()


def experiment(n_episodes=20000, ou_noise = 2.0, ou_noise_decay_rate = 0.998, train_mode=True,
               threshold=0.5, buffer_size=1000000, batch_size=512, update_every=2, tau=0.01,
               lr_actor=0.001, lr_critic=0.001):
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
    :param n_episodes: maximum number of training episodes
    :param train_mode: when 'True' set environment to training mode
    :param threshold: score after which the environment is solved
    :return scores_all, moving_average: List of all scores and moving average.
    """

    env = UnityEnvironment(file_name="Tennis/Tennis", base_port=64738)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    state_size = states.shape[1]

    maddpgagent = MADDPGAgent(state_size=state_size, action_size=action_size, num_agents=num_agents,
                              random_seed=0, buffer_size=buffer_size, device=device,
                              batch_size=batch_size, update_every=update_every, tau=tau, lr_actor=lr_actor,
                              lr_critic=lr_critic)

    scores_window = deque(maxlen=100)
    scores_all = []
    moving_average = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        states = env_info.vector_observations
        maddpgagent.reset()
        scores = np.zeros(num_agents)
        while True:
            actions = maddpgagent.act(states, noise = ou_noise)
            env_info = env.step(actions)[brain_name]  
            next_states = env_info.vector_observations
            rewards = np.asarray(env_info.rewards)  
            dones = np.asarray(env_info.local_done).astype(np.uint8)
            maddpgagent.step(states, actions, rewards, next_states, dones)
            scores += rewards 
            states = next_states 
            if np.any(dones):
                break

        ep_best_score = np.max(scores)
        scores_window.append(ep_best_score)
        scores_all.append(ep_best_score)
        moving_average.append(np.mean(scores_window))
        ou_noise *= ou_noise_decay_rate

        print('\rEpisode {}\tAverage Training Score: {:.3f}\tMin:{:.3f}\tMax:{:.3f}'
              .format(i_episode, np.mean(scores_window), np.min(scores_window), np.max(scores_window)), end='')

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Training Score: {:.3f}\tMin:{:.3f}\tMax:{:.3f}\tMoving Average: {:.3f}'
                  .format(i_episode, np.mean(scores_window), np.min(scores_window), np.max(scores_window),moving_average[-1]))

        if moving_average[-1] > threshold:
            print('<-- Environment solved after {:d} episodes! \
            \n<-- Moving Average: {:.3f}'.format(
                i_episode, moving_average[-1]))
            maddpgagent.save_models()
            break
    return scores_all, moving_average





def main():
    os.environ['NO_PROXY'] = 'localhost,127.0.0.*'
    try:
        os.chdir(os.path.join(os.getcwd(), 'p3_collab-compet/solution'))
        print(os.getcwd())
    except:
        pass

    scores_all, moving_average = experiment(n_episodes=20000, ou_noise=2.0, ou_noise_decay_rate=0.998, train_mode=True,
                   threshold=0.5, buffer_size=1000000, batch_size=512, update_every=2, tau=0.01,
                   lr_actor=0.001, lr_critic=0.001)

    plot(scores=[scores_all, moving_average], ylabels=["Scores", "Average Score"], xlabel="Episode #", title="", text="")


if __name__ == "__main__":
    main()
