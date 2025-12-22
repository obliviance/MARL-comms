"""PPO implementation adapted for Simple Speaker Listener environment.

Adapted from CleanRL-style PPO to work with vector observations in the
Simple Speaker Listener MPE environment with communication noise.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.distributions.categorical import Categorical

from mpe_noise_env import MPEEnv
from ppo_agent import Agent
from mpe_policies import speaker_policy


def batchify_obs(obs, device):
    """Converts PZ style observations to dict of torch arrays."""
    obs = {a: torch.tensor(obs[a], dtype=torch.float32).to(device) for a in obs}
    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    x = np.stack([x[a] for a in x], axis=0)
    x = torch.tensor(x).to(device)
    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.agents)}
    return x


def speaker_policy(obs: np.ndarray, action_space):
    """Heuristic speaker policy from train_idqn.py."""
    try:
        n = action_space.n
        goal_idx = int(np.argmax(obs[:n]))
        if 0 <= goal_idx < n:
            return goal_idx
    except Exception:
        pass
    return action_space.sample()


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    max_cycles = 50
    total_episodes = 5000
    
    # Environment parameters
    noise_mode = "dropout"
    noise_prob = 0.3
    policy_name = "PPO"
    run_mode = "training"

    """ ENV SETUP """
    env = MPEEnv(
        render_mode=None,
        max_cycles=max_cycles,
        continuous_actions=False,
        noise_mode=noise_mode,
        noise_prob=noise_prob,
        policy_name=policy_name,
        run_mode=run_mode,
        success_threshold=-12.0
    )
    num_agents = len(env.agents)
    
    # Get observation and action spaces for each agent
    obs_sizes = {agent: env.observation_space(agent).shape[0] for agent in env.agents}
    num_actions_dict = {agent: env.action_space(agent).n for agent in env.agents}
    
    """ LEARNER SETUP """
    # Only PPO for listener
    agents = {
        "listener_0": Agent(obs_size=obs_sizes["listener_0"], num_actions=num_actions_dict["listener_0"]).to(device)
    }
    optimizers = {
        "listener_0": optim.Adam(agents["listener_0"].parameters(), lr=0.0001, eps=1e-5)
    }

    """ ALGO LOGIC: EPISODE STORAGE"""
    # Only store listener data for PPO
    rb_actions = torch.zeros(max_cycles).to(device)
    rb_logprobs = torch.zeros(max_cycles).to(device)
    rb_rewards = torch.zeros(max_cycles).to(device)
    rb_terms = torch.zeros(max_cycles).to(device)
    rb_values = torch.zeros(max_cycles).to(device)

    """ TRAINING LOGIC """
    for episode in range(total_episodes):
        # Collect episode data
        with torch.no_grad():
            next_obs, info = env.reset()
            total_episodic_return = 0
            end_step = 0
            
            # Store observations for training (only listener needed for PPO)
            episode_obs_list = []

            for step in range(0, max_cycles):
                obs_dict = batchify_obs(next_obs, device)
                
                # Store listener observation
                episode_obs_list.append(obs_dict["listener_0"].clone())

                # Speaker uses heuristic
                sp_obs = next_obs["speaker_0"]
                sp_act = speaker_policy(sp_obs, env.action_space("speaker_0"))
                
                # Listener uses PPO
                ls_obs = obs_dict["listener_0"].unsqueeze(0)
                ls_action, ls_logprob, _, ls_value = agents["listener_0"].get_action_and_value(ls_obs)
                
                actions_dict = {
                    "speaker_0": sp_act,
                    "listener_0": ls_action.item()
                }

                # Execute environment
                next_obs, rewards, terms, truncs, infos = env.step(actions_dict)

                # Store episode data (only listener)
                rb_rewards[step] = rewards["listener_0"]
                # 0 if done, 1 if not done
                rb_terms[step] = 1.0 - float(terms["listener_0"] or truncs["listener_0"])
                rb_actions[step] = ls_action
                rb_logprobs[step] = ls_logprob
                rb_values[step] = ls_value.flatten()

                total_episodic_return += rewards["listener_0"]

                if any(terms.values()) or any(truncs.values()):
                    end_step = step + 1
                    break

        # Bootstrap value if not done
        with torch.no_grad():
            # Initialize within the episode loop
            rb_advantages = torch.zeros(max_cycles).to(device)
            next_advantage = 0
            gae_lambda = 0.95

            for t in reversed(range(end_step)):
                # Bootstrap logic
                next_value = rb_values[t + 1] if t + 1 < end_step else 0
                # rb_terms[t] is 0 if t was the last step
                mask = rb_terms[t] 
                
                delta = rb_rewards[t] + gamma * next_value * mask - rb_values[t]
                rb_advantages[t] = delta + gamma * gae_lambda * mask * next_advantage
                next_advantage = rb_advantages[t]

            # Calculate returns
            rb_returns = rb_advantages + rb_values

            # Normalize the whole batch here!
            b_advantages = (rb_advantages[:end_step] - rb_advantages[:end_step].mean()) / (rb_advantages[:end_step].std() + 1e-8)

        # Convert episodes to batch of individual transitions
        b_logprobs = rb_logprobs[:end_step]
        b_actions = rb_actions[:end_step]
        b_returns = rb_returns[:end_step]
        b_values = rb_values[:end_step]
        b_advantages = rb_advantages[:end_step]
        b_obs = torch.stack(episode_obs_list)

        # Optimize policy and value network
        b_index = np.arange(len(b_logprobs))
        clip_fracs = []
        
        for repeat in range(3):
            np.random.shuffle(b_index)
            for start in range(0, len(b_logprobs), batch_size):
                end = start + batch_size
                batch_index = torch.tensor(b_index[start:end]).to(device)
                
                agent_name = "listener_0"
                agent_obs = b_obs[batch_index]
                agent_actions = b_actions[batch_index]
                agent_logprobs = b_logprobs[batch_index]
                agent_returns = b_returns[batch_index]
                agent_values = b_values[batch_index]
                agent_advantages = b_advantages[batch_index]

                _, newlogprob, entropy, value = agents[agent_name].get_action_and_value(
                    agent_obs, agent_actions.long()
                )
                logratio = newlogprob - agent_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # Normalize advantages
                advantages = agent_advantages
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                else:
                    advantages = advantages - advantages.mean()

                # Policy loss
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - agent_returns) ** 2
                v_clipped = agent_values + torch.clamp(
                    value - agent_values, -clip_coef, clip_coef
                )
                v_loss_clipped = (v_clipped - agent_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizers[agent_name].zero_grad()
                loss.backward()
                optimizers[agent_name].step()

        # Log progress
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if episode % 10 == 0:
            print(f"Training episode {episode}")
            print(f"Episodic Return: {np.mean(total_episodic_return)}")
            print(f"Episode Length: {end_step}")
            print(f"Value Loss: {v_loss.item()}")
            print(f"Policy Loss: {pg_loss.item()}")
            print(f"Old Approx KL: {old_approx_kl.item()}")
            print(f"Approx KL: {approx_kl.item()}")
            print(f"Clip Fraction: {np.mean(clip_fracs)}")
            print(f"Explained Variance: {explained_var.item()}")
            print("-------------------------------------------\n")
    
    # Close environment to save metrics
    env.close()
    print(f"Training complete. Results saved to eval_data/{policy_name}_{run_mode}_{noise_mode}_{noise_prob}_results.json")

    # Save the model
    model_path = f"ppo_agents_{noise_mode}_{noise_prob}.pth"
    torch.save({agent_name: agents[agent_name].state_dict() for agent_name in agents}, model_path)
    print(f"Model saved to {model_path}")

"""    # Example rendering after training
    render_env = MPEEnv(
        render_mode="human",
        max_cycles=max_cycles,
        continuous_actions=False,
        noise_mode=noise_mode,
        noise_prob=noise_prob,
        policy_name=policy_name,
        run_mode="evaluation"
    )

    for agent_name in agents:
        agents[agent_name].eval()

    with torch.no_grad():
        for episode in range(5):
            obs, infos = render_env.reset()
            obs_dict = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            
            while not any(terms) and not any(truncs):
                actions_list = []
                for agent_name in render_env.agents:
                    agent_obs = obs_dict[agent_name].unsqueeze(0)
                    action, _, _, _ = agents[agent_name].get_action_and_value(agent_obs)
                    actions_list.append(action)
                actions = torch.cat(actions_list)
                
                obs, rewards, terms, truncs, infos = render_env.step(unbatchify(actions, render_env))
                obs_dict = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]
    
    render_env.close()
"""