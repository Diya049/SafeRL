"""
Method 1 — Lagrangian Constrained Q-Learning
=============================================
Core idea
---------
Standard Q-learning maximises reward.  Lagrangian RL adds a *constraint*:
the average safety-violation cost C(π) must stay ≤ cost_limit d.

We solve this with the *Lagrangian relaxation* of the constrained MDP:

    max_π  min_{λ≥0}  J(π) - λ · (C(π) - d)

    λ  is the *dual variable* (penalty multiplier).  It is updated online:
        λ ← clip( λ + λ_lr · (C̄ - d),  0,  λ_max )

    where C̄ is a running estimate of the average constraint cost.

The agent learns two Q-tables:
  • Q_r  — tracks expected reward
  • Q_c  — tracks expected constraint cost

The action chosen is:
    a* = argmax_a  [ Q_r(s,a)  -  λ · Q_c(s,a) ]

Safety events (hazard, human collision, near-human) contribute to the cost
signal used to update Q_c and the dual variable.
"""

import numpy as np
import random
import collections
from config import *
from env import GridWorld


# ─── helpers ────────────────────────────────────────────────────────────────

def cost_from_info(info: dict, done: bool) -> float:
    """Binary cost: 1 if ANY safety constraint was violated this step."""
    return float(info.get("hazard") or info.get("collision") or info.get("distance"))


# ─── agent ──────────────────────────────────────────────────────────────────

class LagrangianAgent:
    def __init__(self, alpha=ALPHA, gamma=GAMMA,
                 eps_start=EPSILON_START, eps_end=EPSILON_END,
                 eps_decay=EPSILON_DECAY,
                 cost_limit=LAG_COST_LIMIT,
                 lambda_lr=LAG_LAMBDA_LR,
                 lambda_init=LAG_LAMBDA_INIT,
                 lambda_max=LAG_LAMBDA_MAX):

        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

        self.cost_limit = cost_limit
        self.lambda_lr  = lambda_lr
        self.lambda_max = lambda_max
        self.lam        = lambda_init          # dual variable λ

        # Q-tables: state → dict{action: value}
        self.Q_r = collections.defaultdict(lambda: np.zeros(5))
        self.Q_c = collections.defaultdict(lambda: np.zeros(5))

        # running average of cost for dual update
        self._cost_ema = 0.0
        self._ema_alpha = 0.05

    # ── action selection ────────────────────────────────────────────────────

    def lagrangian_q(self, state):
        """Combined Q value used for action selection."""
        return self.Q_r[state] - self.lam * self.Q_c[state]

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        return int(np.argmax(self.lagrangian_q(state)))

    # ── learning ─────────────────────────────────────────────────────────────

    def update(self, state, action, reward, cost, next_state, done):
        # --- reward Q-table ---
        best_next_r = 0.0 if done else np.max(self.Q_r[next_state])
        td_r = reward + self.gamma * best_next_r - self.Q_r[state][action]
        self.Q_r[state][action] += self.alpha * td_r

        # --- cost Q-table ---
        best_next_c = 0.0 if done else np.max(self.Q_c[next_state])
        td_c = cost + self.gamma * best_next_c - self.Q_c[state][action]
        self.Q_c[state][action] += self.alpha * td_c
 
        # --- dual variable update ---
        # update EMA first
        self._cost_ema = (1 - self._ema_alpha) * self._cost_ema + self._ema_alpha * cost

        # update lambda
        self.lam = np.clip(
            self.lam + self.lambda_lr * (self._cost_ema - self.cost_limit),
            0.0, self.lambda_max
        )

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


# ─── training loop ──────────────────────────────────────────────────────────

def train(num_episodes=NUM_EPISODES, max_steps=MAX_STEPS,
          agent_kwargs=None, seed=42):
    random.seed(seed); np.random.seed(seed)

    env   = GridWorld()
    agent = LagrangianAgent(**(agent_kwargs or {}))

    history = {
        "episode":        [],
        "reward":         [],
        "violations":     [],       # number of unsafe steps
        "violation_rate": [],       # violations / steps
        "lambda":         [],
        "steps":          [],
        "success":        [],
        "best_frames":    None,     # rendered frames of best episode
        "best_reward":    -np.inf,
        "best_log":       [],
    }

    for ep in range(num_episodes):
        state    = env.reset()
        total_r  = 0.0
        viols    = 0
        ep_steps = 0
        frames   = [] if LOG_BEST else None
        ep_log   = []

        for step in range(max_steps):
            action              = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            cost                = cost_from_info(info, done)

            agent.update(state, action, reward, cost, next_state, done)

            total_r  += reward
            viols    += int(cost > 0)
            ep_steps += 1
            state     = next_state

            if LOG_BEST:
                frames.append(env.render().copy())
                ep_log.append({
                    "step":   step,
                    "action": action,
                    "reward": reward,
                    "cost":   cost,
                    "lambda": agent.lam,
                    "info":   info,
                    "pos":    list(env.agent_pos),
                })

            if done:
                break

        agent.decay_epsilon()

        vrate = viols / ep_steps if ep_steps > 0 else 0.0
        success = tuple(env.agent_pos) == env.goal_pos

        history["episode"].append(ep)
        history["reward"].append(total_r)
        history["violations"].append(viols)
        history["violation_rate"].append(vrate)
        history["lambda"].append(agent.lam)
        history["steps"].append(ep_steps)
        history["success"].append(int(success))

        if total_r > history["best_reward"]:
            history["best_reward"]  = total_r
            history["best_frames"]  = frames
            history["best_log"]     = ep_log
            history["best_episode"] = ep

        if (ep + 1) % EVAL_EVERY == 0:
            recent = history["reward"][-EVAL_EVERY:]
            rviol  = history["violation_rate"][-EVAL_EVERY:]
            print(f"[Lagrangian] ep {ep+1:4d} | "
                  f"avg_r={np.mean(recent):8.2f} | "
                  f"avg_vrate={np.mean(rviol):.3f} | "
                  f"λ={agent.lam:.10f} | ε={agent.epsilon:.3f}")

    return history, agent


# ─── run standalone ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    history, agent = train()
    print(f"\nBest episode: {history['best_episode']}  "
          f"reward={history['best_reward']:.2f}")
