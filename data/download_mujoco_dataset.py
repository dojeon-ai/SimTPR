import argparse
import gym
import d4rl
    

if __name__ == '__main__':
    envs = ['halfcheetah', 'walker2d', 'hopper', 'ant']

    # Create the environment
    env = gym.make('halfcheetah-medium-expert-v0')

    # d4rl abides by the OpenAI gym interface
    env.reset()
    obs, act, rew, info = env.step(env.action_space.sample())

    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos
    import pdb
    pdb.set_trace()
    
    dataset = env.get_dataset()
    print(dataset['observations']) # An N x dim_observation Numpy array of observations

    # Alternatively, use d4rl.qlearning_dataset which
    # also adds next_observations.
    dataset = d4rl.qlearning_dataset(env)

#halfcheetah-medium-expert-v0/v2
#walker2d-medium-expert-v0/v2
#hopper-medium-expert-v0/v2
#ant-medium-expert-v0/v2

