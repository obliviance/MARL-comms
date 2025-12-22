import random
import numpy as np
import pettingzoo.mpe as mpe
import json
import glob
import torch


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
                    if agent == "speaker_0":
                        actions[agent] = mpe_policies.speaker_policy(obs, action_space)
                    else:
                        actions[agent] = policy_fn(q_table, obs, action_space, agent)
                else:
                    actions[agent] = None
            
            observations, rewards, terminations, truncations, _ = env.step(actions)
            done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

    env.close()
    
    print("Evaluation completed.")

if __name__ == "__main__":
    noise_modes = ["dropout", "flip", "gaussian"]
    noise_probs = [0.0, 0.1, 0.3, 0.5, 1.0]
    policy_name = "PPO"

    """# Training SARSA policy with different noise settings
    trained_q_table = None
    for noise_mode in noise_modes:
        for noise_prob in noise_probs:
            print(f"Training SARSA with Noise Mode: {noise_mode}, Noise Probability: {noise_prob}")
            training_env = MPEEnv(render_mode=None, max_cycles=50, continuous_actions=False, noise_mode=noise_mode, noise_prob=noise_prob, policy_name=policy_name, run_mode="training", success_threshold=-12.0)
            trained_q_table = mpe_policies.train_base_q_learning(training_env, num_episodes=1000, epsilon=0.1, alpha=0.5, gamma=0.9)

    # Print training stats
    print("Training completed.")
    eval_files = glob.glob(f"eval_data/{policy_name}_training_*.json")
    for eval_file in eval_files:
        with open(eval_file, 'r') as f:
            data = json.load(f)
            avg_reward = np.mean(data['results']['average_rewards'])
            success_rate = np.mean(data['results']['success_rate'])
            print(f"File: {eval_file} | Avg Reward: {avg_reward:.3f} | Success Rate: {success_rate:.3f}")

    # Evaluating the trained SARSA policy with different noise settings    
    for noise_mode in noise_modes:
        for noise_prob in noise_probs:
            print(f"Evaluating for Noise Mode: {noise_mode}, Noise Probability: {noise_prob}")
            eval_env = MPEEnv(render_mode=None, max_cycles=50, continuous_actions=False, noise_mode=noise_mode, noise_prob=noise_prob, policy_name=policy_name, run_mode="evaluation", success_threshold=-12.0)
            evaluate_policy(eval_env, mpe_policies.run_base_q_learning, q_table=trained_q_table, num_episodes=500)"""

    # Example: Evaluating PPO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for noise_mode in noise_modes:
        for noise_prob in noise_probs:
            model_path = f"ppo_agents_flip_0.0.pth"
            print(f"Evaluating PPO for Noise Mode: {noise_mode}, Noise Probability: {noise_prob}")
            eval_env = MPEEnv(render_mode=None, max_cycles=50, continuous_actions=False, noise_mode=noise_mode, noise_prob=noise_prob, policy_name=policy_name, run_mode="evaluation", success_threshold=-12.0)
            try:
                ppo_agents = mpe_policies.load_ppo_model(model_path, eval_env, device)
                evaluate_policy(eval_env, mpe_policies.run_ppo, q_table=ppo_agents, num_episodes=500)
            except FileNotFoundError:
                print(f"Model file {model_path} not found, skipping evaluation.")
    
