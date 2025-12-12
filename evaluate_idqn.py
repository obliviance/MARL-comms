import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from idqn_model import DQN
from train_idqn import COMM_DIM, speaker_policy
from mpe_env import mask_landmarks, apply_comm_noise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load trained dqn for the listener and evaluate it in the simple speaker listener env at a given noise level 
def evaluate_model(
    model_path: str,
    noise_prob: float,
    num_episodes: int = 50,
    max_cycles: int = 25,
    success_threshold: float = -12.0,
    noise_mode: str = "dropout",
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
    successes = 0
    
    for ep in range(num_episodes):
        obs, infos = env.reset()
        done = {agent: False for agent in env.possible_agents}
        
        s_raw = apply_comm_noise(listener_id, obs[listener_id], mode=noise_mode, noise_prob=noise_prob)
        s_l = mask_landmarks(s_raw)
        ep_reward = 0.0
        
        while not all(done.values()):
            actions = {}
            
            # # speaker simple random action
            # actions[speaker_id] = env.action_space(speaker_id).sample()
            sp_obs = obs[speaker_id]
            actions[speaker_id] = speaker_policy(sp_obs, env.action_space(speaker_id))
            
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
            s_l = apply_comm_noise(listener_id, obs[listener_id], mode=noise_mode, noise_prob=noise_prob)
            obs = next_obs
            
        episode_rewards.append(ep_reward)
        if ep_reward > success_threshold:
            successes += 1
        
    env.close()
    avg_reward = float(np.mean(episode_rewards))
    success_rate = successes/num_episodes
    reward_stability = float(np.std(episode_rewards))
    return avg_reward, success_rate, reward_stability

if __name__ == "__main__":
    model_path = "idqn_listener_clean.pth"  # from train_idqn.py
    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]
    NOISE_MODE = "gaussian" # none, dropout, gaussian, flip
    
    print(f"Evaluating trained IDQN with noise mode: {NOISE_MODE}:")
    results = []
    for p in noise_levels:
        avg_r, succ, std = evaluate_model(model_path, noise_prob=p, noise_mode=NOISE_MODE)
        results.append((p, avg_r, succ, std))
        print(f"Noise = {p:.1f}, average episode reward = {avg_r:.3f}, success rate = {succ*100:.1f}%, reward std = {std:.2f}")

    # success retention
    results_arr = np.array(results, dtype=float)
    base_success = results_arr[0,2]
    
    print("\nSuccess retention relative to 0.0 noise:")
    for noise, avg_r, succ, std in results_arr:
        if base_success > 0:
            retention = succ / base_success
            print(
                f"Noise = {noise:.1f}, success = {succ*100:.1f}%, "
                f"success retention = {retention*100:.1f}%"
            )
        else:
            print(
                f"Noise = {noise:.1f}, success = {succ*100:.1f}%, "
                "success retention = N/A (baseline success is 0)"
            )

    np.save("idqn_eval_results.npy", np.array(results, dtype=float))
