import unittest
import numpy as np
from qsml import QuantumRegister, grover_search
from qsml.resources import state_vector_bytes, recommend_qubit_limit


class TestQSML(unittest.TestCase):
    def test_bell_state(self):
        q = QuantumRegister(2).h(0).cnot(0, 1)
        np.testing.assert_allclose(q.probabilities(), [0.5, 0, 0, 0.5])
        self.assertTrue(q.validate())

    def test_gate_x(self):
        q = QuantumRegister(1).x(0)
        np.testing.assert_allclose(q.probabilities(), [0, 1])

    def test_memory_estimates(self):
        self.assertEqual(state_vector_bytes(1), 32)
        self.assertEqual(state_vector_bytes(10), 16384)

    def test_recommendation(self):
        self.assertEqual(recommend_qubit_limit(16 * 1024, 0.25, 3), 6)

    def test_grover_finds_marked_state(self):
        result = grover_search(3, ["101"], shots=100, seed=3)
        self.assertEqual(result.top_candidates[0][0], "101")
        self.assertGreater(result.measurements.get("101", 0), 90)

    def test_grover_multiple_marked(self):
        result = grover_search(3, ["001", "110"], shots=1000, seed=4)
        self.assertGreater(result.probabilities["001"], 0.45)
        self.assertGreater(result.probabilities["110"], 0.45)

    def test_bad_marked_state(self):
        with self.assertRaises(ValueError):
            grover_search(3, ["11"], shots=10)

    def test_bad_qubit_count(self):
        with self.assertRaises(ValueError):
            state_vector_bytes(0)


if __name__ == "__main__":
    unittest.main()
