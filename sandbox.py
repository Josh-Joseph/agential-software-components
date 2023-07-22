"""Brute force policy search for a reinforcement learning environment with error bars."""

import itertools
from typing import List, Tuple, Callable
from copy import deepcopy
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm


class DiscreteBoundedSpace(spaces.Discrete):
    """A bounded discrete space."""

    def __init__(self, n, low, high):
        super().__init__(n)
        self.low = low
        self.high = high
        self.values = set(range(self.low, self.high + 1))

    def sample(self):
        return np.random.randint(self.low, self.high + 1)

    def contains(self, x):
        return x in self.values


class OneDimensionalEnv(gym.Env):
    """Custom 1-D environment that follows gym interface."""

    def __init__(
        self,
            num_states: int = 10,
            num_actions: int = 3,
            max_steps: int = 20,
            reward_goal: int = 100,
            reward_step: int = -1
    ) -> None:
        """Initialize the environment.

        Args:
            num_states: The number of states in the environment.
            num_actions: The number of actions in the environment.
            max_steps: The maximum number of steps in an episode.
            reward_goal: The reward for reaching the goal.
            reward_step: The reward for taking a step.
        """
        super().__init__()

        assert num_states % 2 != 0, "num_states must be odd"
        self.state_space = DiscreteBoundedSpace(
            num_states, -1 * (num_states - 1) // 2, (num_states - 1) // 2)
        self.goal_space = DiscreteBoundedSpace(
            num_states, -1 * (num_states - 1) // 2, (num_states - 1) // 2)
        assert num_actions % 2 != 0, "num_states must be odd"
        # -1: left, 0: stay, 1: right
        self.action_space = DiscreteBoundedSpace(
            num_actions, -1 * (num_actions - 1) // 2, (num_actions - 1) // 2)  

        self.max_steps = max_steps
        self.reward_goal = reward_goal
        self.reward_step = reward_step

        self.current_step = 0
        self.current_position = 0
        self.goal_position = 0
        self.terminated = False
        self.truncated = False
        self.reset()

    def step(self, action: int) -> Tuple[int, int, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: The action to take.

        Returns:
            A tuple containing the next state, reward, terminated, truncated, and info.
        """
        assert not self.terminated, "Current episode has terminated"
        assert not self.truncated, "Current episode has been truncated"
        assert self.action_space.contains(action), "Invalid action"
        self.current_position += action
        # Ensure the agent stays within bounds
        self.current_position = np.clip(
            self.current_position,
            self.state_space.low,
            self.state_space.high
        )
        assert self.state_space.contains(self.current_position), "Invalid state"
        self.current_step += 1

        if self.current_position == self.goal_position:
            reward = self.reward_goal
            self.terminated = True
        else:
            reward = self.reward_step
            if self.current_step >= self.max_steps:
                self.truncated = True
        info = {}
        return (self.current_position,
                reward,
                self.terminated,
                self.truncated,
                info)

    def reset(self) -> int:
        """Reset the environment for a new episode.

        Returns:
            A tuple of the initial state and info.
        """
        self.current_step = 0
        self.current_position = self.state_space.sample()  # Start random
        self.goal_position = self.goal_space.sample()  # Goal random
        self.terminated = False
        self.truncated = False

        # Ensure the start and goal are not the same
        while self.current_position == self.goal_position:
            self.current_position = self.state_space.sample()

        info = {}
        return self.current_position, info

    def reset_to_specific_start(self, goal_state: int, start_state: int) -> int:
        """Reset the environment for a new episode to a specific start and goal.

        Returns:
            A tuple of the initial state and info.
        """
        self.current_step = 0
        self.current_position = start_state
        self.goal_position = goal_state
        self.terminated = False
        self.truncated = False

        info = {}
        return self.current_position, info


def evaluate_policy(
        env: gym.Env,
        policy: dict[tuple[int, int], int],
        n_episodes: int = 100
) -> float:
    """Evaluate a policy using sampling.

    Args:
        env: The environment to evaluate the policy in.
        policy: The policy to evaluate.
        n_episodes: The number of episodes to evaluate the policy over.

    Returns:
        The fraction of successful episodes.
    """
    run_successes = 0
    for _ in range(n_episodes):
        run_success = 0
        done = False
        state, info = env.reset()
        while not done:
            action = policy[(env.goal_position, env.current_position)]
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                run_success = 1
                done = True
            elif truncated:
                done = True
        run_successes += run_success
    return run_successes / n_episodes


def brute_force_evaluate_policy(
        env: gym.Env,
        policy: dict[tuple[int, int], int]
) -> float:
    """Brute force evaluate a policy

    Args:
        env: The environment to evaluate the policy in.
        policy: The policy to evaluate.

    Returns:
        The fraction of successful episodes.
    """
    run_successes = 0
    for goal_state in env.goal_space.values:
        for start_state in env.state_space.values:
            run_success = 0
            done = False
            state, info = env.reset_to_specific_start(goal_state, start_state)
            while not done:
                action = policy[(env.goal_position, env.current_position)]
                state, reward, terminated, truncated, info = env.step(action)
                if terminated:
                    run_success = 1
                    done = True
                elif truncated:
                    done = True
            run_successes += run_success
    return run_successes / (env.goal_space.n * env.state_space.n)


def optimize_policy(
        env: gym.Env,
        policies: List[List[int]],
        success_condition: Callable[[int, bool, int], bool]) -> Tuple[
        List[tuple[float, float, float]], tuple[float, float, float]]:
    """Find the best policy using brute force optimization."""

    best_policy_scores = []
    best_policy_score = (-1, -1, -1)

    for policy in tqdm(policies, desc='Evaluating policies', total=len(policies)):
        policy_score = evaluate_policy(env, policy, success_condition)
        if policy_score[0] > best_policy_score[0]:
            best_policy_score = policy_score
        best_policy_scores.append(best_policy_score)

    return best_policy_scores, best_policy_score


def sample_a_discrete_policy(env: gym.Env) -> dict[tuple[int, int], int]:
    """Sample a discrete policy.
    
    Args:
        env: The environment to sample the policy for.
        
    Returns:
        A dictionary mapping (goal, state) to action.
    """
    possible_goals = env.goal_space.values
    possible_states = env.state_space.values
    possible_goals_x_states = list(itertools.product(
        possible_goals, possible_states))
    policy = {}
    for goal_and_state in possible_goals_x_states:
        policy[goal_and_state] = env.action_space.sample()
    return policy

def purturb_policy(policy: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    """Purturb a discrete policy.
    
    Args:
        policy: The policy to purturb.
        
    Returns:
        The purturbed policy.
    """
    purturbed_policy = deepcopy(policy)
    key_to_purturb = random.choice(list(purturbed_policy.keys()))
    purturbed_policy[key_to_purturb] = env.action_space.sample()
    return purturbed_policy


def plot_agency_over_time(performance_history: list[int, float]) -> None:
    """Plot agency over time.

    Args:
        performance_history: A list of tuples containing the iteration and policy performance.
    """
    # Unpack the performance history into two lists
    iterations, policy_performances = zip(*performance_history)

    fig = go.Figure(data=go.Scatter(x=list(iterations), 
                                    y=list(policy_performances), 
                                    mode='markers'))

    fig.update_layout(title='Agency over Iterations',
                      xaxis_title='Iteration',
                      yaxis_title='Agency')

    fig.show()


if __name__ == '__main__':
    env = OneDimensionalEnv(num_states=5, num_actions=3)
    performance_history = []
    policy = None
    policy_performance = -np.inf
    for i in range(1000):
        if policy is None:
            proposed_policy = sample_a_discrete_policy(env)
        else:
            proposed_policy = purturb_policy(policy)
        proposed_policy_performance = brute_force_evaluate_policy(
            env, proposed_policy)
        if proposed_policy_performance >= policy_performance:
            policy = proposed_policy
            policy_performance = proposed_policy_performance
        performance_history.append([i, policy_performance])
        print(f"Iteration={i}; Policy performance={policy_performance}")
    plot_agency_over_time(performance_history)
