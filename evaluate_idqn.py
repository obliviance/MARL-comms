import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from idqn_model import DQN
from train_idqn import apply_comm_dropout, COMM_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(
    model_path: str,
    noise_prob: float,
    num_episodes: int = 50,
    max_cycles: int = 25
):
    env = simple_speaker_listener_v4.parallel_env(max_cycles=max_cycles, continuous_actions=False)
    
    listener_id = next(a for a in env.possible_agents if "listener" in a)
    speaker_id = next(a for a in env.possible_agents if "speaker" in a)
    
    obs, infos = env.reset()
    obs_dim = env.observation_space(listener_id).shape[0]
    n_actions = env.action_space(listener_id).n
    
    # load network
    q_net = DQN(obs_dim, n_actions).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    q_net.load_state_dict(state_dict)
    q_net.eval()
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, infos = env.reset()
        done = {agent: False for agent in env.possible_agents}
        
        s_l = apply_comm_dropout(listener_id, obs[listener_id], drop_prob=noise_prob)
        ep_reward = 0.0
        
        while not all(done.values()):
            actions = {}
            
            # speaker simple random action
            actions[speaker_id] = env.action_space(speaker_id).sample()
            
            # listener greedy action from trained Q-network
            if done[listener_id]:
                actions[listener_id] = 0
            else:
                with torch.no_grad():
                    s_tensor = torch.tensor(s_l, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    q_vals = q_net(s_tensor)
                    a_l = int(q_vals.argmax(dim=1).item())
                actions[listener_id] = a_l
                
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done = {ag: bool(terminations[ag] or truncations[ag]) for ag in env.possible_agents}
            
            r_l = rewards[listener_id]
            ep_reward += r_l
            
            # next noisy observation for listener
            s_l = apply_comm_dropout(listener_id, next_obs[listener_id], drop_prob=noise_prob)
            obs = next_obs
            
        episode_rewards.append(ep_reward)
        
    env.close()
    avg_reward = float(np.mean(episode_rewards))
    return avg_reward

if __name__ == "__main__":
    model_path = "idqn_listener_clean.pth"  # from train_idqn.py
    noise_levels = [0.0, 0.1, 0.3, 0.5]

    print("Evaluating trained IDQN at different noise levels:")
    results = []
    for p in noise_levels:
        avg_r = evaluate_model(model_path, noise_prob=p, num_episodes=50)
        results.append((p, avg_r))
        print(f"Noise = {p:.1f}, average episode reward = {avg_r:.3f}")

    # Save results to a .npy or .txt for plotting later if you like
    np.save("idqn_eval_results.npy", np.array(results, dtype=float))
