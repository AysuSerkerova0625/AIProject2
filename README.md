## How to run
``` 
git clone https://github.com/AysuSerkerova0625/AIProject2.git
cd src
python3 -m venv venv
source venv/bin/activate
python3 nqueens_hybrid.py n-queen.txt
```

## How to run test cases

``` 
python3 -m unittest -v test_nqueens_hybrid.py
```
### The tests check:
- correctness of diagonal and conflict calculations
- that the solver returns a valid N-Queens solution
- runtime performance for medium and large board sizes

<br>

# Task Definition
The task is to solve the N-Queens problem, which involves placing N queens on an N × N chessboard so that no two queens attack each other (no shared row, column, or diagonal).

The goal is to efficiently find a valid solution for large values of N (10 ≤ N ≤ 1000) using a hybrid approach based on Min-Conflicts and constraint satisfaction heuristics.

<br>

# Code explanation

This project implements a hybrid N-Queens solver that combines local search with constraint-satisfaction ideas.
- Min-Conflicts
The main algorithm. Queens are repaired by swapping columns to reduce conflicts.
- MRV (Minimum Remaining Values)
Among conflicting rows, the algorithm chooses the row with the fewest good swap options.
- LCV (Least Constraining Value)
Chooses the swap that introduces the smallest number of new conflicts.
- Local AC-3
A lightweight arc-consistency check applied only to a small subset of rows to keep the algorithm fast.
- Constant-time conflict counting
Column and diagonal conflicts are tracked using counters, allowing O(1) conflict checks.

The solver is designed to work efficiently for board sizes from 10 up to 1000.