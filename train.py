from collections import deque
import numpy as np
import torch

NUM_EPISODES = 2000
T_MAX = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

TARGET_SCORE = 13

def train(env, agent, brain_name, num_episodes=NUM_EPISODES, t_max=T_MAX, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY):
    scores_window = deque(maxlen=100)
    scores = []

    epsilon = epsilon_start

    for i in range(1, num_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        for t in range(t_max):
            action = agent.act(state, epsilon)
            env_info = env.step(action)[brain_name]

            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                break

        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        scores_window.append(score)
        scores.append(score)

        avg_score = np.mean(scores_window)

        print(f"\repisode {i}\tepsilon: {epsilon:.2f}\t avg score: {avg_score:.2f}\t", end="")

        if i % 100 == 0:
            print(f"\repisode {i}\tepsilon: {epsilon:.2f}\t avg score: {avg_score:.2f}\t")

        if avg_score >= TARGET_SCORE:
            print(f"\nenv solved in {i - 100} episodes!\t avg score: {avg_score:.2f}")

            torch.save(agent.local_net.cpu().state_dict(), "checkpoint.pth")

            break

    return scores
