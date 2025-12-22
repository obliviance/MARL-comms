import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def load_result(file_path):
    """Loads a single JSON result file."""
    with open(file_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding {file_path}")
            return None
        
def plot_training_curve(data, output_path):
    """Plots training curves for average reward and success rate."""
    
    episodes = np.arange(len(data["successes"]))
    successes = data["successes"]
    # Compute success rate as a cumulative percentage of total episodes
    success_rates = np.cumsum(successes) / (np.arange(len(successes)) + 1)
    
    # Use cumulative_rewards (one value per episode) instead of flattening step rewards
    episode_rewards = np.array(data["agents"]["listener_0"]['cumulative_rewards']).flatten()
    avg_rewards = np.array(data["agents"]["listener_0"]['average_rewards']).flatten()
    
    # Compute moving average for smoothing (window of 50 episodes)
    window = 50
    if len(episode_rewards) >= window:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        smoothed_episodes = episodes[window-1:]
    else:
        smoothed_rewards = episode_rewards
        smoothed_episodes = episodes

    plt.figure(figsize=(12, 5))    
    
    # Success Rate Plot
    plt.subplot(1, 2, 2)
    plt.plot(episodes, success_rates, label='Success Rate', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title('Training Curve: Success Rate')
    plt.grid(True)
    plt.legend()

    # Episode Reward Plot
    plt.subplot(1, 2, 1)
    # Plot raw rewards with low alpha (transparency) and no label to keep legend clean
    plt.plot(episodes, episode_rewards, color='orange', alpha=0.3)
    # Plot smoothed rewards with the label
    plt.plot(smoothed_episodes, smoothed_rewards, label='Episode Reward (Smoothed)', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Curve: Episode Reward')
    plt.grid(True)
    plt.legend(loc='lower right') # Specify location to avoid overlapping data

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def print_stats(data):
    """Prints average reward and success rate from the data."""
    baseline_success_rate = 0.956  # Baseline success rate for comparison
    avg_episode_reward = np.mean(data['agents']['listener_0']['episode_rewards'])
    success_rate = np.mean(data['successes'])
    reward_std = np.std(data['agents']['listener_0']['episode_rewards'])
    success_retention = success_rate / baseline_success_rate
    print(f"Avg Episode Reward: \n{avg_episode_reward:.3f} \n Reward Std Dev: \n{reward_std:.3f} \n Success Rate: \n{success_rate:.3f} \n Success Retention: \n{success_retention:.3f}")

def load_results(data_dir):
    """Loads all JSON result files from the specified directory."""
    results = []
    path = Path(data_dir)
    if not path.exists():
        return results
    
    for file in path.glob("*.json"):
        with open(file, 'r') as f:
            try:
                results.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Error decoding {file}")
    return results

def aggregate_data(results):
    """Groups data by noise mode, policy, and probability for aggregation."""
    # Structure: {noise_mode: {policy: {prob: [metrics]}}}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for res in results:
        mode = res['noise']['mode']
        policy = res['policy']
        prob = res['noise']['prob']
        
        success_rate = np.mean(res.get('successes', []))
        
        # Average cumulative reward across all agents for the run
        agent_means = []
        for agent_data in res.get('agents', {}).values():
            agent_means.append(np.mean(agent_data.get('cumulative_rewards', [])))
        
        avg_reward = np.mean(agent_means) if agent_means else 0
        
        data[mode][policy][prob].append({
            'success_rate': success_rate,
            'avg_reward': avg_reward
        })
    return data

def plot_metrics(aggregated_data, output_dir="plots"):
    """Generates and saves plots for each noise mode and metric."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = [('success_rate', 'Success Rate'), ('avg_reward', 'Avg Cumulative Reward')]
    
    for mode, policies in aggregated_data.items():
        for metric_key, ylabel in metrics:
            plt.figure(figsize=(10, 6))
            
            for policy, probs_dict in policies.items():
                sorted_probs = sorted(probs_dict.keys())
                means = []
                stds = []
                
                for p in sorted_probs:
                    vals = [run[metric_key] for run in probs_dict[p]]
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                
                plt.errorbar(sorted_probs, means, yerr=stds, label=policy, marker='o', capsize=5)
            
            plt.title(f"{ylabel} vs Noise Probability ({mode} noise)")
            plt.xlabel("Noise Probability")
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.savefig(os.path.join(output_dir, f"{mode}_{metric_key}.png"))
            plt.close()

def plot_results():
    """Main function to load results, aggregate data, and plot metrics."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # 1. Data Setup based on the provided table
    data = {
        "Training Noise": [0.0]*15 + [0.3]*15,
        "Noise Type": (["Dropout"]*5 + ["Gaussian"]*5 + ["Symbol Flip"]*5) * 2,
        "Test Noise": [0.0, 0.1, 0.3, 0.5, 1.0] * 6,
        "Avg Reward": [
            -2.139, -2.080, -2.096, -2.081, -2.060, # 0.0 Dropout
            -2.126, -2.149, -2.172, -2.109, -2.298, # 0.0 Gaussian
            -2.081, -2.040, -1.962, -1.972, -1.994, # 0.0 Symbol Flip
            -2.712, -2.782, -3.105, -2.784, -3.059, # 0.3 Dropout
            -2.057, -2.045, -2.079, -1.978, -2.096, # 0.3 Gaussian
            -3.686, -4.058, -3.868, -3.604, -4.073  # 0.3 Symbol Flip
        ],
        "Success Rate": [
            95.6, 97.6, 97.6, 98.2, 98.8, # 0.0 Dropout
            97.2, 96.8, 97.0, 97.0, 96.4, # 0.0 Gaussian
            97.4, 97.2, 97.2, 98.4, 98.2, # 0.0 Symbol Flip
            91.4, 90.4, 87.4, 90.2, 85.8, # 0.3 Dropout
            98.4, 98.8, 98.4, 99.2, 98.6, # 0.3 Gaussian
            77.4, 73.2, 74.8, 78.2, 71.2  # 0.3 Symbol Flip
        ]
    }

    df = pd.DataFrame(data)

    # 2. Plotting Configuration
    noise_types = df["Noise Type"].unique()
    train_noises = df["Training Noise"].unique()

    # Create a figure with 6 subplots (3 Noise Types x 2 Training Conditions)
    fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharex=True)
    fig.suptitle("Performance Metrics across Noise Types and Training Conditions", fontsize=20)

    for i, n_type in enumerate(noise_types):
        for j, t_noise in enumerate(train_noises):
            ax = axes[i, j]
            subset = df[(df["Noise Type"] == n_type) & (df["Training Noise"] == t_noise)]
            
            # Plot Reward on primary y-axis
            color_reward = 'tab:blue'
            lns1 = ax.plot(subset["Test Noise"], subset["Avg Reward"], marker='o', 
                        color=color_reward, label="Avg Episode Reward", linewidth=2)
            ax.set_ylabel("Avg Episode Reward", color=color_reward, fontsize=12)
            ax.tick_params(axis='y', labelcolor=color_reward)
            
            # Create secondary y-axis for Success Rate
            ax2 = ax.twinx()
            color_success = 'tab:green'
            lns2 = ax2.plot(subset["Test Noise"], subset["Success Rate"], marker='s', 
                            color=color_success, label="Success Rate (%)", linewidth=2, linestyle='--')
            ax2.set_ylabel("Success Rate (%)", color=color_success, fontsize=12)
            ax2.tick_params(axis='y', labelcolor=color_success)
            ax2.set_ylim(0, 105) # Fixed scale for percentages
            
            # Formatting
            ax.set_title(f"Noise: {n_type} | Train Noise: {t_noise}", fontsize=14, fontweight='bold')
            if i == 2: ax.set_xlabel("Test Noise Level", fontsize=12)
            
            # Combine legends
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc='lower left', frameon=True, fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    base_path = Path(__file__).parent
    results = load_results(base_path / "eval_data")
    train_file = base_path / "eval_data" / "PPO_training_flip_0.0_results.json"
    train_data = load_result(train_file)
    if results:
        aggregated = aggregate_data(results)
        plot_metrics(aggregated, base_path / "plots")
        plot_training_curve(train_data, base_path / "plots" / "PPO_training_curve.png")
        plot_results()
    else:
        print("No data found to plot.")
