# Project 3: Collaboration and Competition
Yu Tao

## Overview

In this project, two reinforcement learning (RL) agents were trained to play tennis against each other. For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. This project environment is similar to, but not identical to the [Tennis environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md) on the Unity ML-Agents GitHub page.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of **+0.1**.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of **-0.01**.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of **8** variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. **2** continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of **+0.5** (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least **+0.5**.

## Learning Algorithm

### Deep Deterministic Policy Gradient (DDPG)
This project implements an off-policy method called **Deep Deterministic Policy Gradient**, the details can be found in [this paper](https://arxiv.org/pdf/1509.02971.pdf), written by researchers at Google Deepmind. The DDPG algorithm belongs to the actor-critic methods, that use deep function approximators to learn policies in high-dimensional, continuous action spaces.

The implementation details can be found in the **DDPG_agent.py** file in this repository.

### Actor-Critic Method
Actor-critic methods leverage the strengths of both policy-based and value-based methods. Using a policy-based approach, the agent (actor) learns how to act by directly estimating the optimal policy and maximizing reward through gradient ascent. Meanwhile, employing a value-based approach, the agent (critic) learns how to estimate the value (i.e., the future cumulative reward) of different state-action pairs. Actor-critic methods combine these two approaches in order to accelerate the learning process. Actor-critic agents are also more stable than value-based agents, while requiring fewer training samples than policy-based agents.

The implementation details can be found in the **model.py** file in this repository.

### Hyperparameters

The DDPG used the following hyperparameters (details in ddpg_agent.py)

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

### Model Architecture

The model architecture is as follows (details in model.py):

For the actor part, it consists of 3 fully connnected layers:
```
self.fc1 = nn.Linear(state_size, fc1_units)
self.fc2 = nn.Linear(fc1_units, fc2_units)
self.fc3 = nn.Linear(fc2_units, action_size)
```

For the critic part, it consists of 3 fully connnected layers:
```
self.fcs1 = nn.Linear(state_size, fcs1_units)
self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
self.fc3 = nn.Linear(fc2_units, 1)
```

The Actor received 33 variables (observation space) as input and generated 4 numbers (predicted action)as output. The first two fully connected layers were followed by ReLU activation function while the third fully connected layers were followed by tanh activation function to output vector between -1 and 1. The Actor is used to approximate the optimal policy π deterministically.

The Critic received 33 variables (observation space) as input, and its first hidden layer was stacked with the Actor's output layer as the Critic's second hidden layer. Eventually giving predictions on the target value, the optimal action-value function Q(s,a), by using the Actor's best-believed action.

The visualization of the model architecture is as follows:

![Model_architecture](./images/Model_architecture.png)

The networks used the Adam optimizer, and the learning rate was set to 0.001, with a batch size of 128.

### Plot of Rewards

![Rewards](./images/Rewards.png)

The second environment was used, which is to train on **20 identical agents**. This model solved the environment in **20** episodes, which meets the requirement that the agents are able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30. The final model is saved in **checkpoint_actor.pth** and **checkpoint_critic.pth**.

## Ideas for Future Work

To improve the performance of the agent(s), there are several ideas to tune the DDPG algorithm we have used:
- Through **trial and error** and test out other combinations of the hyperparameters, there might be other sets of values that could solve the environment faster.
- DDPG used minibatches taken uniformly from the replay buffer. We can test out the **prioritized replay buffer** and compare the results.
- The requirement of the task was set to a score of +30 over 100 consecutive episodes. With a **higher target score**, the agents might do better.

There are also other actor-critic methods available for us to explore, such as [A3C - Asynchronous Advantage Actor-Critic](https://arxiv.org/abs/1602.01783), [PPO - Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347.pdf), [D4PG - Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/pdf/1804.08617.pdf), etc. We can run this project on these algorithms and compare the results.
