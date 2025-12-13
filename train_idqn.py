from idqn_model import DQN, ReplayBuffer
from mpe_env import apply_comm_noise, mask_landmarks
from pettingzoo.mpe import simple_speaker_listener_v4
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# communication part: last 3 dims of listener obs: [self_vel, landmark, comm]
COMM_DIM = 3    # from MPE docs: listener obs shape (11,) last part is communciation
NOISE_PROB = 0.0    # default training without noise; test with differetn values later
NOISE_MODE = "dropout" # modes: none, dropout, gaussian, flip

# simple heuristic for speaker:
# argmax if obs looks like one hot goal od and marches action space size
def speaker_policy(obs: np.ndarray, action_space):
    try:
        # if hasattr(action_space, "n") and action_space.n == obs.shape[0]:
        #     return int(np.argmax(obs))
        n = action_space.n
        goal_idx = int(np.argmax(obs[:n]))
        if 0 <= goal_idx < n:
            return goal_idx
    except Exception:
        pass
    return action_space.sample()

# TRAINING LOOP

def train_idqn(
    num_ep: int = 500,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.999,
    noise_prob: float = NOISE_PROB,
    target_update_freq: int = 1000
):
    # train idqn on listener_0 in the simple_speaker_listener_v4 env
    # uses message drop noise on the listener's obeservation
    
    env = simple_speaker_listener_v4.parallel_env(
        max_cycles = 50, continuous_actions = False
    )
    
    listener_id = next(a for a in env.possible_agents if "listener" in a)
    speaker_id = next(a for a in env.possible_agents if "speaker" in a)
    
    # reset once to get shapes
    obs, infos = env.reset()
    obs_dim = env.observation_space(listener_id).shape[0]
    nA = env.action_space(listener_id).n
    
    q_net = DQN(obs_dim, nA).to(DEVICE) # main NN that approximates Q func
    target_net = DQN(obs_dim, nA).to(DEVICE) #target NN that computes target vals in Bellman update
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()
    
    eps = eps_start
    global_step = 0
    ep_rewards = []
    
    for ep in range(num_ep):
        obs, infos = env.reset()
        done = {agent: False for agent in env.possible_agents}
        
        # initial noisy observation for listener
        s_raw = apply_comm_noise(listener_id, obs[listener_id], mode=NOISE_MODE, noise_prob=noise_prob)
        s_l = mask_landmarks(s_raw)
        ep_reward = 0.0
        
        while not all(done.values()):
            actions = {}
            
            # --- SPEAKER
            sp_obs = obs[speaker_id]
            sp_act = speaker_policy(sp_obs, env.action_space(speaker_id))
            actions[speaker_id] = sp_act
            
            # --- LISTENER 
            if done[listener_id]: # check if listener is done w/ current ep
                actions[listener_id] = 0
            else:
                # eps greedy to pick an action
                if random.random() < eps:
                    # pick random action for action space
                    a_l = env.action_space(listener_id).sample()
                else:
                    with torch.no_grad():
                        # get listener's obs and convert to 2D tensore of shape [1, obs_dim]
                        s_tensor = torch.tensor(
                            s_l, dtype = torch.float32, device = DEVICE
                        ).unsqueeze(0)
                        q_values = q_net(s_tensor) # run DQN and return Q values
                        a_l = int(q_values.argmax(dim=1).item())
                        
                actions[listener_id] = a_l
            
            # step env
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done = {
                agent: bool(terminations[agent] or truncations[agent]) for agent in env.possible_agents
            }

            r_l = rewards[listener_id]
            ep_reward += r_l
            
            # next noisy obs for listener
            s_l_next = apply_comm_noise(listener_id, next_obs[listener_id], mode=NOISE_MODE, noise_prob=noise_prob)
            done_l = done[listener_id]
            
            # store transition
            buffer.push(s_l, a_l, r_l, s_l_next, done_l)
            s_l = s_l_next
            obs = next_obs
    
            # --- DQN UPDATE: compute target values using the Bellman eq and target network
            global_step += 1
            if len(buffer) >= batch_size:   # start learning
                # grab batch_size transitions from buffer
                states, actions_b, rewards_b, next_states, dones_b = buffer.sample(batch_size)
                
                states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
                actions_b = torch.tensor(actions_b, dtype=torch.long, device=DEVICE)
                rewards_b = torch.tensor(rewards_b, dtype=torch.float32, device=DEVICE)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
                dones_b = torch.tensor(dones_b, dtype=torch.float32, device=DEVICE)
                
                # Q(s,a)
                q_vals = q_net(states).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    # q values at next states according to target network
                    next_q_vals = target_net(next_states).max(dim=1)[0]
                    # bellman eq
                    targets = rewards_b + gamma * next_q_vals * (1.0 - dones_b)
                
                # get mean squared error between q values and targets
                loss = nn.functional.mse_loss(q_vals, targets)
                
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()
                
                # eps decay
                eps = max(eps_end, eps * eps_decay)
            
            # update target network periodically
            if global_step % target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())
            
        ep_rewards.append(ep_reward)
        # print(f"episodes {ep+1}/{num_ep}, reward={ep_reward:.2f}, epsilon={eps:.3f}")
    
    env.close()
    return q_net, ep_rewards

# Main
if __name__ == "__main__":
    # train without noise for baseline
    trained_q, rewards = train_idqn(num_ep=5000, noise_prob=0.0)
    
    # save model weights for later eval
    torch.save(trained_q.state_dict(), "idqn_listener_clean.pth")
    np.save("idqn_rewards_clean.npy", np.array(rewards))
    print("Training complete. model saved to idqn_listener_clean.pth")