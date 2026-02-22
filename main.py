#!/usr/bin/env python3
"""
Hybrid N-Queens (10 <= n <= 1000)

Fast core: swap-based MIN-CONFLICTS (keeps 1 queen per column always)

CSP parts (kept lightweight so n=1000 still works):
- MRV: choose the conflicted row that has the smallest set of good swap options
- LCV: choose the swap that causes the least conflicts
- AC-3: run REAL AC-3, but only on a small LOCAL subset (sample) to keep it fast
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from typing import Deque, List, Optional, Set, Tuple


# ---------- I/O ----------

def read_board(path: str) -> List[int]:
    cols: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if "//" in line:
                line = line.split("//", 1)[0].strip()
            if line:
                cols.append(int(line))
    return cols


def write_board(board: List[int]) -> None:
    for c in board:
        print(c)


# ---------- Fast conflict counters (O(1) checks/updates) ----------

def d1(r: int, c: int, n: int) -> int:  # main diag index
    return (r - c) + (n - 1)


def d2(r: int, c: int) -> int:          # anti diag index
    return r + c


class C:
    """Counts for columns + diagonals so conflicts are O(1)."""

    def __init__(self, n: int, board: List[int]) -> None:
        self.n = n
        self.col = [0] * n
        self.di1 = [0] * (2 * n - 1)
        self.di2 = [0] * (2 * n - 1)
        for r, c in enumerate(board):
            self.col[c] += 1
            self.di1[d1(r, c, n)] += 1
            self.di2[d2(r, c)] += 1

    def conflicts(self, r: int, c: int) -> int:
        n = self.n
        return (self.col[c] - 1) + (self.di1[d1(r, c, n)] - 1) + (self.di2[d2(r, c)] - 1)

    def row_conf(self, board: List[int], r: int) -> int:
        return self.conflicts(r, board[r])


    # This function returns all rows whose queens currently violate constraints and must be repaired by the min-conflicts algorithm.
    def conflicted_rows(self, board: List[int]) -> List[int]:
        return [r for r in range(self.n) if self.row_conf(board, r) > 0]

    def rem(self, r: int, c: int) -> None:
        n = self.n
        self.col[c] -= 1
        self.di1[d1(r, c, n)] -= 1
        self.di2[d2(r, c)] -= 1

    def add(self, r: int, c: int) -> None:
        n = self.n
        self.col[c] += 1
        self.di1[d1(r, c, n)] += 1
        self.di2[d2(r, c)] += 1


# ---------- Local AC-3 (real queue-based, but only on a small sample) ----------


# This function checks the N-Queens constraint: two queens are compatible if they are not in the same column and not on the same diagonal.

def ok(i: int, ci: int, j: int, cj: int) -> bool:
    return (ci != cj) and (abs(ci - cj) != abs(i - j))


def ac3_local(rows: List[int], domains: List[Set[int]]) -> bool:
    """
    AC-3 (LOCAL): make the sampled subproblem arc-consistent.
    rows[t] is the real row number for local variable t.
    """
    m = len(rows)
# This line initializes the AC-3 queue with all directed arcs between distinct variables in the local CSP so every domain can be revised for consistency.
    q: Deque[Tuple[int, int]] = deque(
        (a, b) for a in range(m) for b in range(m) if a != b)

    def revise(a: int, b: int) -> bool:
        ra, rb = rows[a], rows[b]
        Da, Db = domains[a], domains[b]
        bad = [va for va in Da if not any(ok(ra, va, rb, vb) for vb in Db)]
        if not bad:
            return False
        for va in bad:
            Da.remove(va)
        return True

    while q:
        a, b = q.popleft()
        if revise(a, b):
            if not domains[a]:
                return False
            for k in range(m):
                if k != a and k != b:
                    q.append((k, a))
    return True


# ---------- Min-conflicts + MRV + LCV + local AC-3 ----------

# The input file is only used to determine n. We use a random permutation as the initial assignment, which is standard for min-conflicts and ensures one queen per column.
def rand_perm(n: int) -> List[int]:
    board = list(range(n))
    random.shuffle(board)
    return board

# ---Min-conflicts---

# swap_score temporarily removes two queens, estimates how many conflicts they would have if swapped using constant-time counters, and returns this as the min-conflicts score.
def swap_score(board: List[int], cnt: C, r1: int, r2: int) -> int:
    """How many conflicts the two queens would have AFTER swapping their columns (smaller is better)."""
    n = cnt.n
    c1, c2 = board[r1], board[r2]

    cnt.rem(r1, c1)
    cnt.rem(r2, c2)
    s1 = cnt.col[c2] + cnt.di1[d1(r1, c2, n)] + cnt.di2[d2(r1, c2)]
    s2 = cnt.col[c1] + cnt.di1[d1(r2, c1, n)] + cnt.di2[d2(r2, c1)]
    cnt.add(r1, c1)
    cnt.add(r2, c2)

    return s1 + s2


# ---LCV---
# We compute a conflict score for swapping row r with every other row r2, sort by (score, r2) so ties prefer smaller r2, and take the first k rows as the best swap partners.

def best_partners(board: List[int], cnt: C, r: int, k: int) -> List[int]:
    """Pick best k swap partners for row r (LCV candidates)."""
    n = cnt.n
    arr: List[Tuple[int, int]] = []
    for r2 in range(n):
        if r2 == r:
            continue
        arr.append((swap_score(board, cnt, r, r2), r2))
    arr.sort(key=lambda t: (t[0], t[1]))
    return [r2 for _, r2 in arr[:k]]




# ---MRV---
def pick_row_mrv(board: List[int], cnt: C, k: int, sample: int) -> Optional[Tuple[int, List[int]]]:
    """
    MRV: among conflicted rows, choose the row with the smallest number of good swap options.
    (We sample conflicted rows to keep it fast.)
    """
    bad = cnt.conflicted_rows(board)
    if not bad:
        return None
    if len(bad) > sample:
        bad = random.sample(bad, sample)

    best_r = bad[0]
    best_cands = best_partners(board, cnt, best_r, k)
    best_mrv = len(best_cands)

# This loop scans conflicted rows and selects the one with the smallest number of good swap options, implementing the MRV heuristic.
    for r in bad[1:]:
        cands = best_partners(board, cnt, r, k)
        mrv = len(cands)
        if mrv < best_mrv:
            best_r, best_cands, best_mrv = r, cands, mrv

    return best_r, best_cands


def local_domains(board: List[int], cnt: C, rows: List[int], dom_k: int) -> List[Set[int]]:
    """
    Build small domains for local AC-3:
    for each local row, try its current column + some random columns,
    keep best dom_k by diagonal conflict estimate.
    """
    n = cnt.n
    out: List[Set[int]] = []
    tries = max(20, dom_k * 3)

    for r in rows:
        cols = {board[r]}
        for _ in range(min(n, tries)):
            cols.add(random.randrange(n))

        scored = []
        for c in cols:
            scored.append((cnt.di1[d1(r, c, n)] + cnt.di2[d2(r, c)], c))
        scored.sort(key=lambda t: (t[0], t[1]))
        out.append(set(c for _, c in scored[:dom_k]))

    return out


def choose_swap_lcv(board: List[int], cnt: C, r: int, cands: List[int],
                    local_rows: int, dom_k: int) -> int:
    """
    LCV: choose the swap that causes the least conflicts.
    Plus: reject swaps that fail local AC-3 (penalty).
    """
    n = cnt.n
    best_r2 = cands[0]
    best_key = (10**18, best_r2)

    for r2 in cands:
        base = swap_score(board, cnt, r, r2)

        # try swap temporarily
        c1, c2 = board[r], board[r2]
        cnt.rem(r, c1)
        cnt.rem(r2, c2)
        board[r], board[r2] = c2, c1
        cnt.add(r, board[r])
        cnt.add(r2, board[r2])

        # AC-3 on a small local subset
        bad = cnt.conflicted_rows(board)
        pool = [x for x in bad if x not in {r, r2}]
        random.shuffle(pool)
        S = [r, r2] + pool[: max(0, local_rows - 2)]
        doms = local_domains(board, cnt, S, dom_k)
        ok_local = ac3_local(S, doms)

        # undo swap
        cnt.rem(r, board[r])
        cnt.rem(r2, board[r2])
        board[r], board[r2] = c1, c2
        cnt.add(r, board[r])
        cnt.add(r2, board[r2])

        penalty = 0 if ok_local else 10**9
        key = (base + penalty, r2)
        if key < best_key:
            best_key = key
            best_r2 = r2
# return the partner row r2 with the smallest rank
    return best_r2


def apply_swap(board: List[int], cnt: C, r1: int, r2: int) -> None:
    c1, c2 = board[r1], board[r2]
    cnt.rem(r1, c1)
    cnt.rem(r2, c2)
    board[r1], board[r2] = c2, c1
    cnt.add(r1, board[r1])
    cnt.add(r2, board[r2])


def solve(n: int, max_steps: int, restarts: int, seed: Optional[int]) -> List[int]:
    if seed is not None:
        random.seed(seed)

    # small defaults (good for n up to 1000)
    cand_k = 20       # how many swap candidates per row
    row_sample = 35   # how many conflicted rows to sample for MRV
    local_rows = 30   # local subset size for AC-3
    dom_k = 10        # domain size per local row

    for _ in range(restarts):
        board = rand_perm(n)
        cnt = C(n, board)

        for _ in range(max_steps):
            # ---- MRV ----
            picked = pick_row_mrv(board, cnt, cand_k, row_sample)
            if picked is None:
                return board
            r, cands = picked

            # ---- LCV + local AC-3 ----
            r2 = choose_swap_lcv(board, cnt, r, cands, local_rows, dom_k)
            apply_swap(board, cnt, r, r2)

    raise RuntimeError("No solution found. Increase max_steps or restarts.")


# ---------- Main ----------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hybrid N-Queens: Min-Conflicts + MRV + LCV + local AC-3")
    ap.add_argument("input_file", help="File with n lines. Used to get n.")
    ap.add_argument("--max-steps", type=int, default=250_000)
    ap.add_argument("--restarts", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    n = len(read_board(args.input_file))
    if not (10 <= n <= 1000):
        raise ValueError(f"n must be between 10 and 1000, got {n}")

    sol = solve(n, args.max_steps, args.restarts, args.seed)
    write_board(sol)


if __name__ == "__main__":
    main()
