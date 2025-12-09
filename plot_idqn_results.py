import numpy as np
import matplotlib.pyplot as plt

# training curve

rewards = np.load("idqn_rewards_clean.npy")

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.title("IDQN training curve")
plt.show()

print("Last 100 episodes avg:", np.mean(rewards[-100:]))

# evaluation results


eval_results = np.load("idqn_eval_results.npy")

noise_levels   = eval_results[:, 0]
avg_rewards    = eval_results[:, 1]
success_rates  = eval_results[:, 2] * 100.0
reward_std     = eval_results[:, 3]

plt.figure()
plt.errorbar(
    noise_levels,
    avg_rewards,
    yerr=reward_std,
    marker="o",
    linestyle="-",
)
plt.xlabel("Noise (drop probability)")
plt.ylabel("Average episode reward")
plt.title("IDQN performance vs communication noise")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(noise_levels, success_rates, marker="o", linestyle="-")
plt.xlabel("Noise (drop probability)")
plt.ylabel("Success rate (%)")
plt.title("IDQN success rate vs communication noise")
plt.ylim(0, 100)
plt.grid(True)
plt.show()
