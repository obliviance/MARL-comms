import random
import numpy as np
import pettingzoo.mpe as mpe

from mpe_noise_env import MPEEnv
import mpe_policies

# Evaluate a given policy function on the provided MPE environment
def evaluate_policy(
    env,
    policy_fn,
    q_table,
    num_episodes: int = 100,
):
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        done = {agent: False for agent in env.agents}
        
        while not all(done.values()):
            actions = {}
            for agent in env.agents:
                if not done[agent]:
                    obs = observations[agent]
                    action_space = env.action_space(agent)
                    actions[agent] = policy_fn(q_table, obs, action_space, agent)
                else:
                    actions[agent] = None
            
            observations, rewards, terminations, truncations, _ = env.step(actions)
            done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

    env.close()
    
    print("Evaluation completed.")

if __name__ == "__main__":
    noise_modes = ["dropout", "gaussian", "flip"]
    noise_probs = [0.0, 0.1, 0.3, 0.5]
    policy_name = "SARSA"

    # Training environment 
    training_env = MPEEnv(render_mode="human", max_cycles=100, continuous_actions=False, noise_mode="none", noise_prob=0.0)
    
    # Train the SARSA policy
    trained_q_table = mpe_policies.train_sarsa(training_env, num_episodes=10, epsilon=0.1, alpha=0.5, gamma=0.9)
    
    for noise_mode in noise_modes:
        for noise_prob in noise_probs:
            print(f"Evaluating for Noise Mode: {noise_mode}, Noise Probability: {noise_prob}")
            # Evaluate the trained policy
            eval_env = MPEEnv(render_mode=None, max_cycles=100, continuous_actions=False, noise_mode=noise_mode, noise_prob=noise_prob)
            evaluate_policy(eval_env, mpe_policies.run_sarsa, q_table=trained_q_table, num_episodes=100)
