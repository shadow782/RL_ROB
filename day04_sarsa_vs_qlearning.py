"""Day 4: SARSA vs Q-Learning on a cliff grid. Python stdlib only."""
import random

WIDTH, HEIGHT = 8, 4
START, GOAL = (0, 3), (7, 3)
CLIFF = {(x, 3) for x in range(1, 7)}
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ALPHA, GAMMA, EPSILON = 0.5, 1.0, 0.1


def step(s, a):
    dx, dy = ACTIONS[a]
    nx = max(0, min(WIDTH - 1, s[0] + dx))
    ny = max(0, min(HEIGHT - 1, s[1] + dy))
    ns = (nx, ny)
    if ns in CLIFF:
        return START, -100, False  # cliff: reset to start, episode continues
    if ns == GOAL:
        return GOAL, -1, True
    return ns, -1, False


def greedy(q, state):
    values = [q.get((state, a), 0.0) for a in range(4)]
    best = max(values)
    return random.choice([a for a, val in enumerate(values) if val == best])


def choose(q, s, eps):
    return random.randrange(4) if random.random() < eps else greedy(q, s)


def train(algorithm, episodes=2500, seed=7):
    random.seed(seed)
    q = {}
    returns = []
    for episode in range(episodes):
        s = START
        a = choose(q, s, EPSILON)
        total = 0
        for _ in range(1000):
            ns, reward, done = step(s, a)
            total += reward
            if done:
                target = reward
            elif algorithm == 'sarsa':
                na = choose(q, ns, EPSILON)
                target = reward + GAMMA * q.get((ns, na), 0.0)
            else:
                target = reward + GAMMA * max(q.get((ns, aa), 0.0) for aa in range(4))
            old = q.get((s, a), 0.0)
            q[(s, a)] = old + ALPHA * (target - old)
            if done:
                break
            s = ns
            a = na if algorithm == 'sarsa' else choose(q, s, EPSILON)
        returns.append(total)
    return q, returns


def print_policy(q):
    signs = '^>v<'
    for y in range(HEIGHT):
        row = []
        for x in range(WIDTH):
            s = (x, y)
            row.append('S' if s == START else 'G' if s == GOAL else 'X' if s in CLIFF else signs[greedy(q, s)])
        print(' '.join(row))


if __name__ == '__main__':
    for algo in ('sarsa', 'qlearning'):
        q, ret = train(algo)
        print('\n', algo.upper(), '| last 100 episodes mean reward:', round(sum(ret[-100:]) / 100, 1))
        print_policy(q)
