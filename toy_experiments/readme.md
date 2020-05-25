# Experiments on Toy MOO problems

### Implementation of problems
The `problems` module contains the toy MOO problems, for which the Pareto front in the objective space is available.
- `toy_biobjective.py`: two objectives
- `toy_triobjective.py`: three objectives

Apart from these two, `simulation.py` implements a many objective toy problem.

### Implementation of solvers
The `solvers` module contains four different solvers:
1. Linear Sclarization: `linscalar.py`
2. MGDA based MOO: `moo_mtl.py` and `min_norm_solvers_numpy.py`
3. Pareto MTL: 
	- cpu: `pmtl.py` and `min_norm_solvers_numpy.py`
	- gpu: `pmtl_gpu.py` and `min_norm_solvers.py`
4. EPO Search: `epo_search.py` and `epo_lp.py`

## Experiments in the Main Paper

### Comparison of four Solvers
`compare_solvers.py` compares the spread of Pareto optimal solutions, and its precision for different preference vectors.

### Comparison of different initializations
`compare_init.py` compares between PMTL and EPO search with 
- `init_type="easy"`, where initialization is near the solution, and 
-  `init_type="hard"`, where initialization is far from the solution.

### Restricted Descent vs. Relaxed Descent
`compare_descent.py` compares between the two types of descent. For *Restricted descent*, the additional constraint in the linear programming is implemented in `epo_lp.py`.

## Experiments in the Supplementary paper

### What happens when EPO does not exist?
`empty_epo.py` uses preference vectors for which EPO does not exist for a biobjectivce problem.

### Tracing the Pareto Front with EPO Search
`trace_pf.py` traces the Pareto front of a triobjective problem, by sequencially finding the EPOs of different preference vectors.

### PMTL vs. EPO Search on Many Objective Problem
`simulation.py` conducts a synthetic experiment to see how both solvers scale with the number of objectives. The results are stored in `simulation.pkl` file, and visualized using `simulation_vis.py`.
