HYPERPARAMETERS = {
    'learning_rate_a' : 0.001,         # learning rate (alpha)
    'discount_factor_g' : 0.95,         # discount rate (gamma)    
    'network_sync_rate' : 100,          # number of steps the agent takes before syncing the policy and target network
    'replay_memory_size' : 50000,       # size of replay memory
    'mini_batch_size' : 64,            # size of the training data set sampled from the replay memory
    'num_hidden_nodes': 256
}

