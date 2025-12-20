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
    num_episodes,
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
    noise_probs = [0.0, 0.1, 0.3, 0.5, 1.0]
    policy_name = "SARSA"

    # Training SARSA policy with different noise settings
    trained_q_table = None
    for noise_mode in noise_modes:
        for noise_prob in noise_probs:
            print(f"Training SARSA with Noise Mode: {noise_mode}, Noise Probability: {noise_prob}")
            training_env = MPEEnv(render_mode=None, max_cycles=50, continuous_actions=False, noise_mode=noise_mode, noise_prob=noise_prob, policy_name=policy_name, run_mode="training")
            trained_q_table = mpe_policies.train_sarsa(training_env, num_episodes=1000, epsilon=0.1, alpha=0.5, gamma=0.9)

    # Evaluating the trained SARSA policy with different noise settings    
    for noise_mode in noise_modes:
        for noise_prob in noise_probs:
            print(f"Evaluating for Noise Mode: {noise_mode}, Noise Probability: {noise_prob}")
            eval_env = MPEEnv(render_mode=None, max_cycles=50, continuous_actions=False, noise_mode=noise_mode, noise_prob=noise_prob, policy_name=policy_name, run_mode="evaluation")
            evaluate_policy(eval_env, mpe_policies.run_sarsa, q_table=trained_q_table, num_episodes=500)
