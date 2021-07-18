import numpy as np
import matplotlib.pyplot as plt
import time

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
# from flatland.envs.rail_env import RailEnv
from mod_rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

from collections import deque
from pytorch_agent import agent
# from tensorflow_model import agent2
from observation_utils import normalize_observation


np.random.seed(1)

observation_tree_depth = 2
TreeObservation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=ShortestPathPredictorForRailEnv())

speed_ration_map = {1.: 1,       # Fast passenger train
                    1. / 2.: 0,  # Fast freight train
                    1. / 3.: 0,  # Slow commuter train
                    1. / 4.: 0}  # Slow freight train

env = RailEnv(width=50, height=50,
              rail_generator=sparse_rail_generator(
                max_num_cities = 5,
                max_rails_between_cities = 2,
                max_rails_in_city = 3, 
                seed=0
            ),
              schedule_generator=sparse_schedule_generator(speed_ration_map), 
              number_of_agents=5, 
              obs_builder_object=TreeObservation,
              remove_agents_at_target=True)
env.reset()

env_renderer = RenderTool(env, screen_height=1000, screen_width=1200)

n_features_per_node = env.obs_builder.observation_dim
n_nodes = 0
for i in range(observation_tree_depth + 1):
    n_nodes += np.power(4, i)   
state_size = n_features_per_node * n_nodes      # in our case is 231

# The action space of flatland is 5 discrete actions
action_size = 5

agent = agent(0.00001, 0.99, 512) # learning rate, epsilon, batch size
# agent.load_model()

n_trials = 2000
n_steps = 400

# Empty dictionary for all agent action
action_dict = dict()
done_buffer = deque(maxlen=100)
done_list = []
score_buffer = deque(maxlen=100)
score_list = []

print("Starting Training")
for trials in range(1, n_trials + 1):

    # Reset environment and get initial observations for all agents
    obs, info = env.reset()
    env_renderer.reset()

    score = 0
    n_dones = 0
    update_weights = 0

    update = [False] * env.get_num_agents()
    first_done = [True] * env.get_num_agents()

    agent_obs = [None] * env.get_num_agents()
    agent_obs_buffer = [None] * env.get_num_agents()
    agent_action_buffer = [2] * env.get_num_agents()

    for a in range(env.get_num_agents()):

        if obs[a]:
            agent_obs[a] = normalize_observation(obs[a], observation_tree_depth, observation_radius=10)

    # Run episode
    for step in range(n_steps):

        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            
            # Choose an action only if it's required
            if info['action_required'][a]:
                action = agent.act(agent_obs[a].reshape(1,231))
                update[a] = True
            else:
                action = 0
                update[a] = False
            action_dict.update({a: action})

        # Environment step which returns the observations for all agents, their corresponding reward and whether they're done
        next_obs, all_rewards, done, info = env.step(action_dict)

        # env_renderer.render_env(show=True, show_observations=True, show_predictions=True)

        # Update replay buffer and train agent
        for a in range(env.get_num_agents()):
            
            # The flag "first done" is used, specifically in the multi-agent setting, to add an 'experience' to the replay buffer *just* the first time the agent is done 
            if done[a] and first_done[a]:
                agent.replay.add_experience(agent_obs[a], action_dict[a], all_rewards[a], agent_obs[a], done[a])
                first_done[a] = False

            if next_obs[a]:
                if update[a]:
                    next_observation = normalize_observation(next_obs[a], observation_tree_depth, observation_radius=10)
                    agent.replay.add_experience(agent_obs[a], action_dict[a], all_rewards[a], next_observation, done[a]) 
                agent_obs[a] = normalize_observation(next_obs[a], observation_tree_depth, observation_radius=10)

            score += all_rewards[a] / env.get_num_agents()

        if done['__all__']:
            env_done = 1 
            break

    for i in range(200):
        agent.train()

    agent.update_eps()

    agents_done = 0
    for i in range(env.get_num_agents()):
        if done[i] == 1:
            agents_done += 1

    score_buffer.append(score / n_steps)
    score_list.append(np.mean(score_buffer)) 
    done_buffer.append(agents_done / env.get_num_agents())
    done_list.append(np.mean(done_buffer))

    
    if trials % 100 == 0: 
        with open('.vscode/DLproject/csv/ma5.csv', "a") as f:
            f.write("\n")
            np.savetxt(f, X=np.transpose(np.asarray([score_buffer, done_buffer])), delimiter=';')

    if trials % 20 == 0:
        agent.save_model()
    
    print(
        '\rEpisode Nr. {}\t Score = {:.3f}\tDones: {:.2f}%\tepsilon {:.2f}\tEpisode done {}\t'.format(trials, np.mean(score_buffer), 100 * np.mean(done_buffer), agent.eps, done), end=" ")


fig, axs = plt.subplots(2)
fig.suptitle('Dones and Score')
axs[0].plot(range(n_trials), done_list)
axs[1].plot(range(n_trials), score_list)
plt.show()
