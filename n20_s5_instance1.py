import matplotlib.pyplot as plt
from nsga2 import NSGA2Solver
from spea2 import SPEA2Solver
from moead import MOEADSolver

# --- Configuration du problème (à partir de l'instance) ---
n_jobs = 20
n_stages = 5

max_skilled_welders = [3, 3, 3]
max_ordinary_welders = [5, 5, 5]
max_robots = [4, 4, 4] 

processing_times = {
   # Task 0
    (0, 0): (8, 4),   (0, 1): (16, 8),  (0, 2): (16, 12), (0, 3): (10, 6),  (0, 4): (12, 5),
    # Task 1
    (1, 0): (10, 0),  (1, 1): (8, 0),   (1, 2): (6, 0),   (1, 3): (6, 0),   (1, 4): (9, 0),
    # Task 2
    (2, 0): (12, 0),  (2, 1): (12, 0),  (2, 2): (8, 0),   (2, 3): (8, 0),   (2, 4): (7, 0),
    # Task 3
    (3, 0): (9, 3),   (3, 1): (11, 6),  (3, 2): (14, 9),  (3, 3): (7, 4),   (3, 4): (10, 3),
    # Task 4
    (4, 0): (15, 0),  (4, 1): (10, 0),  (4, 2): (9, 0),   (4, 3): (12, 0),  (4, 4): (6, 0),
    # Task 5
    (5, 0): (7, 2),   (5, 1): (13, 7),  (5, 2): (11, 8),  (5, 3): (9, 5),   (5, 4): (13, 4),
    # Task 6
    (6, 0): (11, 0),  (6, 1): (9, 0),   (6, 2): (7, 0),   (6, 3): (10, 0),  (6, 4): (8, 0),
    # Task 7
    (7, 0): (14, 4),  (7, 1): (12, 6),  (7, 2): (10, 7),  (7, 3): (11, 3),  (7, 4): (15, 6),
    # Task 8
    (8, 0): (6, 0),   (8, 1): (7, 0),   (8, 2): (9, 0),   (8, 3): (13, 0),  (8, 4): (11, 0),
    # Task 9
    (9, 0): (10, 3),  (9, 1): (15, 8),  (9, 2): (12, 9),  (9, 3): (8, 4),   (9, 4): (14, 5),
    # Task 10
    (10, 0): (13, 0), (10, 1): (11, 0), (10, 2): (10, 0), (10, 3): (7, 0),  (10, 4): (9, 0),
    # Task 11
    (11, 0): (8, 2),  (11, 1): (14, 7), (11, 2): (13, 8), (11, 3): (10, 5), (11, 4): (12, 4),
    # Task 12
    (12, 0): (9, 0),  (12, 1): (8, 0),  (12, 2): (11, 0), (12, 3): (12, 0), (12, 4): (7, 0),
    # Task 13
    (13, 0): (12, 3), (13, 1): (10, 6), (13, 2): (15, 9), (13, 3): (9, 4),  (13, 4): (11, 3),
    # Task 14
    (14, 0): (11, 0), (14, 1): (13, 0), (14, 2): (12, 0), (14, 3): (6, 0),  (14, 4): (10, 0),
    # Task 15
    (15, 0): (7, 2),  (15, 1): (9, 7),  (15, 2): (14, 8), (15, 3): (12, 5), (15, 4): (8, 4),
    # Task 16
    (16, 0): (10, 0), (16, 1): (6, 0),  (16, 2): (8, 0),  (16, 3): (11, 0), (16, 4): (13, 0),
    # Task 17
    (17, 0): (13, 3), (17, 1): (11, 6), (17, 2): (9, 9),  (17, 3): (14, 4), (17, 4): (7, 3),
    # Task 18
    (18, 0): (8, 0),  (18, 1): (10, 0), (18, 2): (7, 0),  (18, 3): (15, 0), (18, 4): (12, 0),
    # Task 19
    (19, 0): (11, 2), (19, 1): (12, 7), (19, 2): (13, 8), (19, 3): (6, 5),  (19, 4): (9, 4)
}

P_run = [1.0] * n_stages
P_sb  = [0.5] * n_stages

complex_operations = {(0, j) for j in range(n_jobs)}

# --- Appel des algorithmes ---

# NSGA-II
nsga2_solver = NSGA2Solver(
    pop_size=100,
    max_gen=200,
    crossover_rate=0.8,  # Nom correct du paramètre
    mutation_rate=0.1
)
nsga2_solver.n_jobs = n_jobs
nsga2_solver.n_stages = n_stages
nsga2_solver.max_skilled_welders = max_skilled_welders
nsga2_solver.max_ordinary_welders = max_ordinary_welders
nsga2_solver.max_robots = max_robots
nsga2_solver.processing_times = processing_times
nsga2_solver.P_run = P_run
nsga2_solver.P_sb = P_sb
nsga2_solver.complex_operations = complex_operations
nsga2_pop = nsga2_solver.solve()
all_fronts = nsga2_solver.non_dominated_sort(nsga2_pop)
nsga2_front = all_fronts[1] 

# SPEA2
spea2_solver = SPEA2Solver(
    pop_size=100,
    max_gen=200,
    crossover_rate=0.8,  # Nom correct du paramètre
    mutation_rate=0.1 
)
spea2_solver.n_jobs = n_jobs
spea2_solver.n_stages = n_stages
spea2_solver.max_skilled_welders = max_skilled_welders
spea2_solver.max_ordinary_welders = max_ordinary_welders
spea2_solver.max_robots = max_robots
spea2_solver.processing_times = processing_times
spea2_solver.P_run = P_run
spea2_solver.P_sb = P_sb
spea2_solver.complex_operations = complex_operations
spea2_front = spea2_solver.solve()

# MOEA/D
moead_solver = MOEADSolver(
        pop_size=100,
        max_gen=200,
        crossover_rate=0.8,
        mutation_rate=0.1,
        neighborhood=10
)
moead_solver.n_jobs = n_jobs
moead_solver.n_stages = n_stages
moead_solver.max_skilled_welders = max_skilled_welders
moead_solver.max_ordinary_welders = max_ordinary_welders
moead_solver.max_robots = max_robots
moead_solver.processing_times = processing_times
moead_solver.P_run = P_run
moead_solver.P_sb = P_sb
moead_solver.complex_operations = complex_operations
moead_pop = moead_solver.solve()
moead_front = [
    sol for sol in moead_pop
    if not any(
        other != sol and
        all(o <= s for o, s in zip(other.objectives, sol.objectives)) and
        any(o < s for o, s in zip(other.objectives, sol.objectives))
        for other in moead_pop
    )
]

# === Function to extract points ===
def get_points(front):
    return [sol.objectives[0] for sol in front], [sol.objectives[1] for sol in front]

m_nsga2, tec_nsga2 = get_points(nsga2_front)
m_spea2, tec_spea2 = get_points(spea2_front)
m_moead, tec_moead = get_points(moead_front)

# === Styles ===
colors = {"NSGA-II": "red", "SPEA2": "blue", "MOEA/D": "magenta"}
markers = {"NSGA-II": "o", "SPEA2": "s", "MOEA/D": "^"}
metric_style = {'linewidth': 2, 'markersize': 8, 'alpha': 0.7}

# ==================== Figure 1 : Pareto Fronts comparison ====================
plt.figure(figsize=(8, 6))
plt.title("Comparison of Pareto Fronts", fontsize=14)

plt.scatter(m_nsga2, tec_nsga2, color=colors["NSGA-II"], marker=markers["NSGA-II"], edgecolors='black', label="NSGA-II")
plt.scatter(m_spea2, tec_spea2, color=colors["SPEA2"], marker=markers["SPEA2"], edgecolors='black', label="SPEA2")
plt.scatter(m_moead, tec_moead, color=colors["MOEA/D"], marker=markers["MOEA/D"], edgecolors='black', label="MOEA/D")

plt.xlabel("Makespan", fontsize=12)
plt.ylabel("TEC", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==================== Figure 2 : Hypervolume ====================
plt.figure(figsize=(8, 6))
plt.title("Hypervolume Evolution", fontsize=14)

plt.plot(nsga2_solver.hv_log, marker='o', color=colors["NSGA-II"], label='NSGA-II', **metric_style)
plt.plot(spea2_solver.hv_log, marker='o', color=colors["SPEA2"], label='SPEA2', **metric_style)
plt.plot(moead_solver.hv_log, marker='o', color=colors["MOEA/D"], label='MOEA/D', **metric_style)

plt.xlabel("Generations (per 5)", fontsize=12)
plt.ylabel("HV Value", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==================== Figure 3 : Generational Distance (GD) ====================
plt.figure(figsize=(8, 6))
plt.title("Generational Distance (GD) Evolution", fontsize=14)

plt.plot(nsga2_solver.gd_log, marker='s', color=colors["NSGA-II"], label='NSGA-II', **metric_style)
plt.plot(spea2_solver.gd_log, marker='s', color=colors["SPEA2"], label='SPEA2', **metric_style)
plt.plot(moead_solver.gd_log, marker='s', color=colors["MOEA/D"], label='MOEA/D', **metric_style)

plt.xlabel("Generations (per 5)", fontsize=12)
plt.ylabel("GD Value", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==================== Final Results ====================
best_hv_nsga2 = max(nsga2_solver.hv_log)
best_gd_nsga2 = min(nsga2_solver.gd_log)

best_hv_spea2 = max(spea2_solver.hv_log)
best_gd_spea2 = min(spea2_solver.gd_log)

best_hv_moead = max(moead_solver.hv_log)
best_gd_moead = min(moead_solver.gd_log)

print("\n" + "="*60)
print("Final Performance Comparison".center(60))
print("="*60)
print(f"{'Algorithm':<12} | {'Best HV':<12} | {'Best GD':<12}")
print("-"*60)
print(f"{'NSGA-II':<12} | {best_hv_nsga2:^12.4f} | {best_gd_nsga2:^12.4f}")
print(f"{'SPEA2':<12} | {best_hv_spea2:^12.4f} | {best_gd_spea2:^12.4f}")
print(f"{'MOEA/D':<12} | {best_hv_moead:^12.4f} | {best_gd_moead:^12.4f}")
print("="*60 + "\n")