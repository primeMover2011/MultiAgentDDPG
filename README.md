
[![Playing Table Tennis](http://img.youtube.com/vi/rYCCLhIvtHQ/0.jpg)](http://www.youtube.com/watch?v=rYCCLhIvtHQ)

[Watch on Youtube](https://www.youtube.com/watch?v=rYCCLhIvtHQ&feature=youtu.be)


# Project 3: Collaboration and Competition

### Introduction

In this project, we will solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Clone this repository, and navigate to the `MultiAgentDDPG/` folder.  Then, install several dependencies.

```bash
git clone https://github.com/primeMover2011/MultiAgentDDPG.git
cd MultiAgentDDPG
```

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

3. Extract the contents of the file to a folder of you choice, preferably as a subfolder of this repository.

### Instructions

- Install [conda](https://conda.io/en/latest/miniconda.html) 

cd in to the directory where you cloned this repository, create a virtual environment and install the required python packages using these commands

```bash
cd MultiAgentDDPG
conda env create -f environment.yml
```

- activate the environment using

```bash
conda activate MultiAgentDDPG
```

- update the location of the environment in _main.py_ and in _test.py_

```python
    env = UnityEnvironment(file_name='YOUR_PATH_HERE', base_port=64739)
```

- Watch the pretrained agent.

```python
  python test.py
```


- Train your own agent using default parameters. 

```python
  python main.py
```

- Read [the report](report.md) and play around with the code, change some hyperparameters!

#HINT
_in main.py_ change
```python
     scores_all, moving_average = experiment(n_episodes=20000, ou_noise=2.0, ou_noise_decay_rate=0.998, train_mode=True,
                   threshold=0.5, buffer_size=1000000, batch_size=512, update_every=2, tau=0.01,
                   lr_actor=0.001, lr_critic=0.001)
```


# Enjoy!


