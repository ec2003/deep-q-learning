import gymnasium as gym
import numpy as np
import random
from time import sleep
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .DQN import DQN
from .replay_memory import ReplayMemory
from .config import HYPERPARAMETERS

class TaxiDQL():
    # Hyperparameters (adjustable)
    learning_rate_a = HYPERPARAMETERS['learning_rate_a']         # learning rate (alpha)
    discount_factor_g = HYPERPARAMETERS['discount_factor_g']         # discount rate (gamma)    
    network_sync_rate = HYPERPARAMETERS['network_sync_rate']          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = HYPERPARAMETERS['replay_memory_size']       # size of replay memory
    mini_batch_size = HYPERPARAMETERS['mini_batch_size']            # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    ACTIONS = ['S','N','E','W','P','D'] # 0: South, 1: North, 2: East, 3: West, 4: Pickup, 5: Dropoff

    # Early stopping
    reward_threshold = 8.0
    window_size = 1000
    consecutive_success = 0
    max_consecutive_success = 3

    # Train the FrozeLake environment
    def train(self, episodes, render=False):
        # Create a dummy environment to get initial state/action space size for network initialization
        # This is needed because the actual environment is created per episode.
        env = gym.make('Taxi-v3', render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network ONCE before the training loop
        policy_dqn = DQN(in_states=num_states, h1_nodes=HYPERPARAMETERS['num_hidden_nodes'], out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=HYPERPARAMETERS['num_hidden_nodes'], out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training on first random map):')
        self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0

        for i in tqdm(range(episodes)):
            # Create a NEW FrozenLake instance for each episode (this will generate a random map)
            env = gym.make('Taxi-v3', render_mode='human' if render else None)
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            # For Taxi-v3, a successful episode typically yields a reward of 20.
            if reward == 20: # Check for successful dropoff
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode)>0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0
            
            # In tiến trình và kiểm tra early stopping
            if (i + 1) % 1000 == 0:
                avg_reward = np.mean(rewards_per_episode[max(0, i-self.window_size+1):(i+1)])
                print(f"Episode {i + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")
                if avg_reward >= self.reward_threshold:
                    consecutive_success += 1
                    print(f"Threshold passed as Avg Reward: {avg_reward:.2f} (>= {self.reward_threshold})")
                    if consecutive_success >= self.max_consecutive_success:
                        print(f"Early stopping at episode {i + 1}")
                        break
                else:
                    consecutive_success = 0

            # Close environment after each episode
            env.close()

        # Save policy (after all episodes)
        torch.save(policy_dqn.state_dict(), "taxi_dql.pt")

        # Create new graph
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)

        # Save plots
        plt.savefig('taxi_dql.png')

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy on a random map
    def test(self, episodes):
        env = gym.make('Taxi-v3', render_mode='human')

        num_states = env.observation_space.n
        num_actions = env.action_space.n
        env.close()

        # Load learned policy ONCE before the test loop
        policy_dqn = DQN(in_states=num_states, h1_nodes=HYPERPARAMETERS['num_hidden_nodes'], out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("taxi_dql.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in tqdm(range(episodes)):
            # Create a NEW FrozenLake instance for each test episode (this will generate a random map)
            env = gym.make('Taxi-v3', render_mode='human')
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):
                # Select best action
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)

        env.close()

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(min(num_states, 10)):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states
