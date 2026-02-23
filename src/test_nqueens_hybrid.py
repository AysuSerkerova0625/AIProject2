import time
import unittest


import src.nqueens_hybrid as nq


def is_valid_solution(board):
    """Check N-Queens validity for board[row] = col."""
    n = len(board)
    if sorted(board) != list(range(n)):
        return False

    cols = set()
    d1 = set()
    d2 = set()

    for r, c in enumerate(board):
        if c in cols:
            return False
        if (r - c) in d1:
            return False
        if (r + c) in d2:
            return False
        cols.add(c)
        d1.add(r - c)
        d2.add(r + c)

    return True


class TestHybridNQueens(unittest.TestCase):

    def test_diagonal_index_ranges(self):
        n = 50
        for r in range(n):
            for c in range(n):
                self.assertTrue(0 <= nq.d1(r, c, n) < 2 * n - 1)
                self.assertTrue(0 <= nq.d2(r, c) < 2 * n - 1)

    def test_conflicts_zero_on_valid_solution(self):
        # Known valid solution for n=10
        board = [0, 2, 5, 7, 9, 4, 8, 1, 3, 6]
        cnt = nq.C(len(board), board)

        for r in range(len(board)):
            self.assertEqual(cnt.row_conf(board, r), 0)

        self.assertEqual(cnt.conflicted_rows(board), [])

    def test_solve_small_n(self):
        n = 10
        sol = nq.solve(n=n, max_steps=200_000, restarts=20, seed=42)
        self.assertEqual(len(sol), n)
        self.assertTrue(is_valid_solution(sol))

    def test_runtime_medium_n(self):
        n = 200
        start = time.perf_counter()
        sol = nq.solve(n=n, max_steps=250_000, restarts=30, seed=1)
        elapsed = time.perf_counter() - start

        self.assertTrue(is_valid_solution(sol))
        print(f"\nRuntime n={n}: {elapsed:.3f} seconds")

    def test_runtime_large_n(self):
        n = 1000
        start = time.perf_counter()
        sol = nq.solve(n=n, max_steps=250_000, restarts=30, seed=7)
        elapsed = time.perf_counter() - start

        self.assertTrue(is_valid_solution(sol))
        print(f"\nRuntime n={n}: {elapsed:.3f} seconds")


if __name__ == "__main__":
    unittest.main(verbosity=2)
