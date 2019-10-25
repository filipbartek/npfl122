import io
from unittest import TestCase

import policy_iteration


class TestPolicyIteration(TestCase):
    def test_policy_iteration(self):
        for steps, iterations, gamma, expected_filename in [(0, 0, 1, 'test_policy_iteration_initial.txt'),
                                                            (0, 0, 0, 'test_policy_iteration_initial.txt'),
                                                            (4, 4, 1, 'test_policy_iteration_assignment.txt')]:
            with self.subTest(steps=steps, iterations=iterations, gamma=gamma):
                policy, value_function = policy_iteration.policy_iteration(steps, iterations, gamma)
                with io.StringIO() as actual, open(expected_filename) as expected_file:
                    policy_iteration.print_results(policy, value_function, file=actual)
                    self.assertEqual(actual.getvalue(), expected_file.read())
