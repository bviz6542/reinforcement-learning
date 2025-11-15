import numpy as np
import gymnasium as gym

# ---------- utils ----------
def logsumexp(x, axis=None, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    if not keepdims:
        y = np.squeeze(y, axis=axis)
    return y

def soft_value_iteration(P, R_sa, gamma=0.99, alpha=0.5, n_iters=500):
    """
    P: [S,A,S] transition
    R_sa: [S,A] reward
    alpha: temperature (smaller => more greedy)
    returns: V[S], Q[S,A], pi[S,A]
    """
    S, A, _ = P.shape
    V = np.zeros(S)
    for _ in range(n_iters):
        Q = R_sa + gamma * np.einsum("sas,s->sa", P, V)
        V_new = alpha * logsumexp(Q / alpha, axis=1)
        if np.max(np.abs(V_new - V)) < 1e-10:
            V = V_new
            break
        V = V_new
    logits = (Q - V[:, None]) / alpha
    pi = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return V, Q, pi

def expected_state_visitation(P, pi, start_dist, horizon):
    """
    mu[t,s] = P(s_t = s) under pi, finite horizon
    """
    S, A, _ = P.shape
    mu = np.zeros((horizon, S))
    mu[0] = start_dist
    for t in range(horizon - 1):
        # mu[t+1, s'] = sum_s mu[t,s] sum_a pi[s,a] P[s,a,s']
        mu[t+1] = np.einsum("s,sa,sas->s", mu[t], pi, P)
    return mu

def demo_state_visitation(demos, S, horizon, pad_last=True):
    """
    demos: list of trajectories, each is list of (s,a,s')
    returns empirical mu_demo[t,s]
    """
    mu = np.zeros((horizon, S))
    for traj in demos:
        if len(traj) == 0:
            continue
        s = traj[0][0]
        last_s = s
        for t in range(horizon):
            mu[t, s] += 1
            if t < len(traj):
                last_s = traj[t][2]
                s = last_s
            else:
                if pad_last:
                    s = last_s
    mu /= len(demos)
    return mu

def sample_demos(env, pi, n_traj=50, horizon=30, seed=0):
    rng = np.random.default_rng(seed)
    demos = []
    for _ in range(n_traj):
        s, _ = env.reset(seed=int(rng.integers(1_000_000)))
        traj = []
        for _t in range(horizon):
            a = rng.choice(env.action_space.n, p=pi[s])
            s_next, r, terminated, truncated, _ = env.step(a)
            traj.append((s, a, s_next))
            s = s_next
            if terminated or truncated:
                break
        demos.append(traj)
    return demos

# ---------- build P from FrozenLake env.P ----------
def extract_P_from_frozenlake(env):
    """
    FrozenLake-v1 exposes env.unwrapped.P: dict[state][action] -> list of (prob, next_state, reward, done)
    We'll convert it to a dense tensor P[s,a,s'].
    """
    P_dict = env.unwrapped.P
    S = env.observation_space.n
    A = env.action_space.n
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))  # expected immediate reward
    for s in range(S):
        for a in range(A):
            for (prob, s_next, r, done) in P_dict[s][a]:
                P[s, a, s_next] += prob
                R[s, a] += prob * r
    return P, R

# ---------- MaxEnt IRL ----------
def maxent_irl(P, demos, start_state, horizon=30, gamma=0.99, alpha=0.5, lr=0.2, n_iters=200):
    """
    Learn state-reward r(s) (one-hot features) so that expected state visitation matches demos.
    - Reward used as "entering next state": R(s,a)=r(s_next)
    """
    S, A, _ = P.shape
    theta = np.zeros(S)  # r(s)

    start_dist = np.zeros(S)
    start_dist[start_state] = 1.0

    mu_demo = demo_state_visitation(demos, S, horizon)
    feat_demo = mu_demo.sum(axis=0)  # expected state counts

    for it in range(n_iters):
        # Build R_sa from theta as reward-on-next-state
        R_sa = np.zeros((S, A))
        for s in range(S):
            for a in range(A):
                # expected next-state reward
                R_sa[s, a] = np.dot(P[s, a], theta)

        # Plan soft-optimal policy under current reward
        _, _, pi = soft_value_iteration(P, R_sa, gamma=gamma, alpha=alpha)

        # Compute expected visitation under pi
        mu_pi = expected_state_visitation(P, pi, start_dist, horizon)
        feat_pi = mu_pi.sum(axis=0)

        grad = feat_demo - feat_pi
        theta += lr * grad

        if (it + 1) % 50 == 0:
            print(f"[IRL] iter={it+1:4d} |grad|={np.linalg.norm(grad):.4f}")

    # Final policy under learned reward
    R_sa = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            R_sa[s, a] = np.dot(P[s, a], theta)
    _, _, pi = soft_value_iteration(P, R_sa, gamma=gamma, alpha=alpha, n_iters=1000)

    return theta, pi

# ---------- run ----------
def main():
    # Make a small discrete env with accessible dynamics
    env = gym.make("FrozenLake-v1", is_slippery=True)  # stochastic transitions
    P, R_env = extract_P_from_frozenlake(env)

    S = env.observation_space.n
    A = env.action_space.n

    start_state = 0  # FrozenLake starts at 0 usually
    horizon = 100

    # 1) Create an "expert" (soft planned) using the env's *true* reward (from env.P)
    V_star, Q_star, pi_expert = soft_value_iteration(P, R_env, gamma=0.99, alpha=0.5)
    demos = sample_demos(env, pi_expert, n_traj=80, horizon=horizon, seed=0)

    # 2) Run MaxEnt IRL: infer state rewards from demos
    theta, pi_irl = maxent_irl(P, demos, start_state=start_state, horizon=horizon,
                              gamma=0.99, alpha=0.5, lr=0.2, n_iters=250)

    # 3) Print results
    print("\nLearned state rewards r_hat(s):")
    for s in range(S):
        print(f"s={s:2d}: {theta[s]: .3f}")

    # Show probability of actions at start
    print("\nPolicy at start state (IRL):", pi_irl[start_state])
    print("Policy at start state (Expert):", pi_expert[start_state])

    env.close()

if __name__ == "__main__":
    main()
