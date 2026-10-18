"""Day 3: tabular Q-learning for a 5-state robot. Python 3, stdlib only."""
import random

N_STATES = 5
GOAL = 4
ACTIONS = [-1, 1]
ALPHA = 0.1
GAMMA = 0.9
EPISODES = 1500
MAX_STEPS = 60
random.seed(7)


def step(state, action):
    # Right/left move succeeds with probability 0.8; otherwise robot stays.
    success = random.random() < 0.8
    next_state = max(0, min(GOAL, state + action)) if success else state
    done = next_state == GOAL
    reward = 10 if done else -1
    return next_state, reward, done


def choose_action(state, q, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    # Break ties randomly to avoid a fixed initial preference.
    best = max(q[state])
    candidates = [i for i, v in enumerate(q[state]) if v == best]
    return ACTIONS[random.choice(candidates)]


q = [[0.0, 0.0] for _ in range(N_STATES)]
recent_returns = []
for episode in range(EPISODES):
    state = 0
    epsilon = max(0.05, 1.0 - episode / 1100)
    episode_return = 0
    for _ in range(MAX_STEPS):
        action = choose_action(state, q, epsilon)
        next_state, reward, done = step(state, action)
        ai = ACTIONS.index(action)
        td_target = reward + (0 if done else GAMMA * max(q[next_state]))
        td_error = td_target - q[state][ai]
        q[state][ai] += ALPHA * td_error
        episode_return += reward
        state = next_state
        if done:
            break
    recent_returns.append(episode_return)
    if (episode + 1) % 300 == 0:
        avg = sum(recent_returns[-100:]) / 100
        print(f'Episode {episode+1:4d} | epsilon={epsilon:.2f} | avg return={avg:.2f}')

print('\nLearned Q-table and greedy policy:')
for s in range(N_STATES):
    if s == GOAL:
        print(f's{s}: terminal')
    else:
        best_action = ACTIONS[q[s].index(max(q[s]))]
        direction = 'right' if best_action == 1 else 'left'
        print(f's{s}: left={q[s][0]:.2f}, right={q[s][1]:.2f}, best={direction}')
