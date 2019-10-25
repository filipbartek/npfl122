#!/usr/bin/env python3

import logging
import sys


class GridWorld:
    # States in the gridworld are the following:
    # 0 1 2 3
    # 4 x 5 6
    # 7 8 9 10

    # The rewards are +1 in state 3 and -100 in state 6

    # Actions are ↑ → ↓ ←; with probability 80% they are performed as requested,
    # with 10% move 90° CCW is performed, with 10% move 90° CW is performed.
    states = 11

    actions = ["↑", "→", "↓", "←"]

    @staticmethod
    def step(state, action):
        return [GridWorld._step(0.8, state, action),
                GridWorld._step(0.1, state, (action + 1) % 4),
                GridWorld._step(0.1, state, (action + 3) % 4)]

    @staticmethod
    def _step(probability, state, action):
        if state >= 5: state += 1
        x, y = state % 4, state // 4
        offset_x = -1 if action == 3 else action == 1
        offset_y = -1 if action == 0 else action == 2
        new_x, new_y = x + offset_x, y + offset_y
        if not (new_x >= 4 or new_x < 0 or new_y >= 3 or new_y < 0 or (new_x == 1 and new_y == 1)):
            state = new_x + 4 * new_y
        if state >= 5: state -= 1
        return [probability, +1 if state == 3 else -100 if state == 6 else 0, state]


def action_value_function(value_function, s, a, gamma):
    """
    Estimate value of action `a` taken from state `s`.

    See [npfl122-02.pdf](https://ufal.mff.cuni.cz/~straka/courses/npfl122/1920/slides.pdf/npfl122-02.pdf) slide 5.
    """
    assert gamma >= 0
    assert gamma <= 1
    result = 0
    for probability, reward, new_state in GridWorld.step(s, a):
        assert probability >= 0
        assert probability <= 1
        assert new_state >= 0
        assert new_state < len(value_function)
        result += probability * (reward + gamma * value_function[new_state])
    return result


# Iterate to implement iterative policy evaluation.
def improve_value_function_once(value_function, policy, gamma):
    output_value_function = [0] * GridWorld.states
    for s in range(GridWorld.states):
        # The policy is deterministic: it always chooses action `policy[s]` in state `s`.
        output_value_function[s] = action_value_function(value_function, s, policy[s], gamma)
    return output_value_function


def iterative_policy_evaluation(policy, iterations, gamma, value_function=None):
    """
    Evaluate policy by iterative policy evaluation algorithm.

    See [npfl122-02.pdf](https://ufal.mff.cuni.cz/~straka/courses/npfl122/1920/slides.pdf/npfl122-02.pdf) slide 22.
    """
    if value_function is None:
        value_function = [0] * GridWorld.states
    iteration_converged = None
    for iteration in range(iterations):
        value_function_new = improve_value_function_once(value_function, policy, gamma)
        if value_function_new == value_function:
            iteration_converged = iteration
            break
        value_function = value_function_new
    if iteration_converged is None:
        logging.debug(f'Iterative policy evaluation finished without convergence after {iterations} iterations.')
    else:
        logging.debug(f'Iterative policy evaluation converged after {iteration_converged} iterations.')
    return value_function


def greedy_policy(value_function, gamma):
    """
    Greedily choose a new policy based on a value function.

    AKA policy improvement.

    See [npfl122-02.pdf](https://ufal.mff.cuni.cz/~straka/courses/npfl122/1920/slides.pdf/npfl122-02.pdf) slide 27, 23.
    """
    output_policy = [0] * GridWorld.states
    for s in range(GridWorld.states):
        best_expected_return = None
        best_a = None
        for a in range(len(GridWorld.actions)):
            expected_return = action_value_function(value_function, s, a, gamma)
            if best_expected_return is None or expected_return > best_expected_return:
                best_expected_return = expected_return
                best_a = a
        output_policy[s] = best_a
    return output_policy


def policy_iteration(steps, iterations, gamma):
    """
    Produce policy and value function using policy iteration algorithm.

    See [npfl122-02.pdf](https://ufal.mff.cuni.cz/~straka/courses/npfl122/1920/slides.pdf/npfl122-02.pdf) slide 27.
    """
    # Start with zero value function and "go North" policy
    policy = [0] * GridWorld.states
    value_function = [0] * GridWorld.states
    policy_converged = False
    for step in range(steps):
        # We baseline the value function with the one we got from the previous policy evaluation.
        value_function_new = iterative_policy_evaluation(policy, iterations, gamma, value_function)
        policy_new = greedy_policy(value_function_new, gamma)
        assert not policy_converged or policy_new == policy
        if not policy_converged and policy_new == policy:
            logging.info(f'Policy converged after {step} steps.')
            policy_converged = True
        policy = policy_new
        if value_function_new == value_function:
            logging.info(f'Value function converged after {step} steps.')
            assert policy_converged
            break
        value_function = value_function_new
    return policy, value_function


def print_results(policy, value_function, file=sys.stdout):
    for l in range(3):
        for c in range(4):
            state = l * 4 + c
            if state >= 5: state -= 1
            print("        " if l == 1 and c == 1 else "{:-8.2f}".format(value_function[state]), end="", file=file)
            print(" " if l == 1 and c == 1 else GridWorld.actions[policy[state]], end="", file=file)
        print(file=file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default=10, type=int, help="Number of policy evaluation/improvements to perform.")
    parser.add_argument("--iterations", default=1, type=int, help="Number of iterations in policy evaluation step.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor.")
    args = parser.parse_args()

    # Implement policy iteration algorithm, with `args.steps` steps of
    # policy evaluation/policy improvement. During policy evaluation, use the
    # current value function and perform `args.iterations` applications of the
    # Bellman equation. Perform the policy evaluation synchronously (i.e., do
    # not overwrite the current value function when computing its improvement).

    policy, value_function = policy_iteration(args.steps, args.iterations, args.gamma)
    print_results(policy, value_function)
