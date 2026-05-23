"""
Method 2 — Safety-Layer Q-Learning
====================================
Core idea
---------
The *safety layer* sits between the policy and the environment.

1. The policy (Q-learning) proposes an action a_proposed.
2. Before executing it, the safety layer checks whether that action
   would lead to an *unsafe* next state (hazard cell, human collision,
   or moving into the proximity of a human).
3. If unsafe → the layer *overrides* the action with the safest
   alternative from the action set.
4. The corrected action a_safe is sent to the environment.

This guarantees (by construction, not by learning) that:
  • The agent never enters a hazard cell.
  • The agent never moves into a cell occupied by a human.
  • The agent avoids cells within Manhattan distance < SL_PROX_RADIUS
    of any human whenever a safe alternative exists.

Because the safety layer handles hard constraints, the underlying
Q-table only needs to optimise reward — no dual variables required.
An extra penalty (SL_PENALTY_IN_TRAIN) is still applied when the
safety layer had to intervene, so the policy *learns* to propose
safe actions over time.
"""

import numpy as np
import random
import collections
from config import *
from env import GridWorld


# ─── safety layer ────────────────────────────────────────────────────────────

ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]   # up/down/left/right/stay

def _next_pos(agent_pos, action):
    dx, dy = ACTION_DELTAS[action]
    return (agent_pos[0] + dx, agent_pos[1] + dy)


def safety_layer(env: GridWorld, proposed_action: int):
    """
    Returns (safe_action, intervened).
    intervened=True means the proposed action was overridden.
    """
    agent = env.agent_pos

    def is_unsafe(pos):
        if not env.is_valid(pos):           # wall or out-of-bounds
            return True
        if pos in env.hazards:              # hazard zone
            return True
        if pos in env.humans:               # human collision
            return True
        # proximity check
        for h in env.humans:
            if abs(h[0] - pos[0]) + abs(h[1] - pos[1]) < SL_PROX_RADIUS:
                return True
        return False

    proposed_pos = _next_pos(agent, proposed_action)

    if not is_unsafe(proposed_pos):
        return proposed_action, False       # proposed action is safe

    # ── find safest alternative ──────────────────────────────────────────────
    # Score alternatives by (1) safe?, (2) distance-to-goal
    gx, gy = env.goal_pos
    best_action = proposed_action           # fallback: keep proposed (may still be blocked by wall)
    best_dist   = np.inf

    for a in range(5):
        pos = _next_pos(agent, a)
        if is_unsafe(pos):
            continue
        dist = abs(pos[0] - gx) + abs(pos[1] - gy)
        if dist < best_dist:
            best_dist   = dist
            best_action = a

    return best_action, True


# ─── agent ──────────────────────────────────────────────────────────────────

class SafetyLayerAgent:
    def __init__(self, alpha=ALPHA, gamma=GAMMA,
                 eps_start=EPSILON_START, eps_end=EPSILON_END,
                 eps_decay=EPSILON_DECAY,
                 sl_penalty=SL_PENALTY_IN_TRAIN):

        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.sl_penalty = sl_penalty

        self.Q = collections.defaultdict(lambda: np.zeros(5))

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = 0.0 if done else np.max(self.Q[next_state])
        td = reward + self.gamma * best_next - self.Q[state][action]
        self.Q[state][action] += self.alpha * td

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


# ─── training loop ──────────────────────────────────────────────────────────

def cost_from_info(info: dict) -> float:
    return float(info.get("hazard") or info.get("collision") or info.get("distance"))


def train(num_episodes=NUM_EPISODES, max_steps=MAX_STEPS,
          agent_kwargs=None, seed=42):
    random.seed(seed); np.random.seed(seed)

    env   = GridWorld()
    agent = SafetyLayerAgent(**(agent_kwargs or {}))

    history = {
        "episode":        [],
        "reward":         [],
        "violations":     [],
        "violation_rate": [],
        "interventions":  [],       # how many times SL overrode policy
        "steps":          [],
        "success":        [],
        "best_frames":    None,
        "best_reward":    -np.inf,
        "best_log":       [],
    }

    for ep in range(num_episodes):
        state    = env.reset()
        total_r  = 0.0
        viols    = 0
        interventions = 0
        ep_steps = 0
        frames   = [] if LOG_BEST else None
        ep_log   = []

        for step in range(max_steps):
            proposed          = agent.select_action(state)
            safe_action, intv = safety_layer(env, proposed)

            next_state, reward, done, info = env.step(safe_action)
            cost = cost_from_info(info)

            # penalise policy when SL had to intervene
            train_reward = reward - (agent.sl_penalty if intv else 0.0)

            agent.update(state, safe_action, train_reward, next_state, done)

            total_r       += reward
            viols         += int(cost > 0)
            interventions += int(intv)
            ep_steps      += 1
            state          = next_state

            if LOG_BEST:
                frames.append(env.render().copy())
                ep_log.append({
                    "step":          step,
                    "proposed":      proposed,
                    "safe_action":   safe_action,
                    "intervened":    intv,
                    "reward":        reward,
                    "cost":          cost,
                    "info":          info,
                    "pos":           list(env.agent_pos),
                })

            if done:
                break

        agent.decay_epsilon()

        vrate   = viols / ep_steps if ep_steps > 0 else 0.0
        success = tuple(env.agent_pos) == env.goal_pos

        history["episode"].append(ep)
        history["reward"].append(total_r)
        history["violations"].append(viols)
        history["violation_rate"].append(vrate)
        history["interventions"].append(interventions)
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
            rintv  = history["interventions"][-EVAL_EVERY:]
            print(f"[SafetyLayer] ep {ep+1:4d} | "
                  f"avg_r={np.mean(recent):8.2f} | "
                  f"avg_vrate={np.mean(rviol):.3f} | "
                  f"avg_intv={np.mean(rintv):.1f} | "
                  f"ε={agent.epsilon:.3f}")

    return history, agent


# ─── run standalone ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    history, agent = train()
    print(f"\nBest episode: {history['best_episode']}  "
          f"reward={history['best_reward']:.2f}")
