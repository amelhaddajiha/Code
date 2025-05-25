import matplotlib.pyplot as plt
from nsga2 import NSGA2Solver
from spea2 import SPEA2Solver
from moead import MOEADSolver

# --- Configuration du problème (à partir de l'instance) ---
n_jobs = 20
n_stages = 10

max_skilled_welders = [2] * n_stages
max_ordinary_welders = [2] * n_stages
max_robots = [4] * n_stages

processing_times = {
    (0, 0): (12, 6), (0, 1): (8, 3), (0, 2): (15, 9), (0, 3): (7, 2), (0, 4): (10, 5),
    (0, 5): (13, 7), (0, 6): (9, 4), (0, 7): (11, 6), (0, 8): (16, 8), (0, 9): (14, 7),
    (1, 0): (9, 1), (1, 1): (11, 2), (1, 2): (7, 1), (1, 3): (13, 3), (1, 4): (6, 1),
    (1, 5): (10, 2), (1, 6): (8, 1), (1, 7): (12, 3), (1, 8): (14, 2), (1, 9): (7, 1),
    (2, 0): (7, 2), (2, 1): (13, 1), (2, 2): (9, 2), (2, 3): (11, 1), (2, 4): (15, 3),
    (2, 5): (8, 1), (2, 6): (12, 2), (2, 7): (6, 1), (2, 8): (10, 2), (2, 9): (14, 1),
    (3, 0): (10, 5), (3, 1): (6, 3), (3, 2): (12, 7), (3, 3): (8, 4), (3, 4): (14, 9),
    (3, 5): (7, 2), (3, 6): (11, 6), (3, 7): (9, 5), (3, 8): (13, 8), (3, 9): (16, 10),
    (4, 0): (11, 1), (4, 1): (7, 2), (4, 2): (14, 1), (4, 3): (9, 2), (4, 4): (6, 1),
    (4, 5): (12, 3), (4, 6): (8, 1), (4, 7): (10, 2), (4, 8): (13, 1), (4, 9): (15, 3),
    (5, 0): (8, 4), (5, 1): (14, 7), (5, 2): (6, 2), (5, 3): (10, 5), (5, 4): (12, 8),
    (5, 5): (7, 3), (5, 6): (9, 6), (5, 7): (15, 9), (5, 8): (11, 7), (5, 9): (13, 8),
    (6, 0): (13, 2), (6, 1): (9, 1), (6, 2): (11, 3), (6, 3): (7, 1), (6, 4): (15, 2),
    (6, 5): (6, 1), (6, 6): (10, 3), (6, 7): (12, 1), (6, 8): (8, 2), (6, 9): (14, 1),
    (7, 0): (6, 3), (7, 1): (12, 6), (7, 2): (10, 5), (7, 3): (14, 8), (7, 4): (8, 4),
    (7, 5): (11, 7), (7, 6): (7, 2), (7, 7): (13, 9), (7, 8): (15, 10), (7, 9): (9, 4),
    (8, 0): (15, 1), (8, 1): (8, 2), (8, 2): (10, 1), (8, 3): (12, 3), (8, 4): (7, 1),
    (8, 5): (9, 2), (8, 6): (14, 1), (8, 7): (6, 2), (8, 8): (11, 1), (8, 9): (13, 3),
    (9, 0): (11, 6), (9, 1): (7, 2), (9, 2): (13, 8), (9, 3): (9, 4), (9, 4): (15, 10),
    (9, 5): (6, 3), (9, 6): (10, 5), (9, 7): (12, 7), (9, 8): (8, 4), (9, 9): (14, 9),
    (10, 0): (8, 1), (10, 1): (12, 3), (10, 2): (14, 2), (10, 3): (7, 1), (10, 4): (9, 3),
    (10, 5): (11, 2), (10, 6): (15, 1), (10, 7): (6, 2), (10, 8): (10, 3), (10, 9): (13, 1),
    (11, 0): (14, 7), (11, 1): (10, 5), (11, 2): (7, 2), (11, 3): (12, 8), (11, 4): (9, 4),
    (11, 5): (16, 9), (11, 6): (6, 3), (11, 7): (11, 6), (11, 8): (8, 4), (11, 9): (13, 7),
    (12, 0): (6, 2), (12, 1): (10, 1), (12, 2): (13, 3), (12, 3): (8, 1), (12, 4): (11, 2),
    (12, 5): (15, 1), (12, 6): (7, 3), (12, 7): (9, 1), (12, 8): (12, 2), (12, 9): (14, 1),
    (13, 0): (12, 7), (13, 1): (8, 4), (13, 2): (11, 6), (13, 3): (15, 9), (13, 4): (7, 2),
    (13, 5): (10, 5), (13, 6): (14, 8), (13, 7): (6, 3), (13, 8): (9, 5), (13, 9): (13, 7),
    (14, 0): (9, 1), (14, 1): (15, 3), (14, 2): (11, 2), (14, 3): (6, 1), (14, 4): (13, 3),
    (14, 5): (7, 1), (14, 6): (10, 2), (14, 7): (12, 1), (14, 8): (8, 3), (14, 9): (14, 2),
    (15, 0): (7, 3), (15, 1): (11, 6), (15, 2): (14, 9), (15, 3): (9, 4), (15, 4): (13, 7),
    (15, 5): (8, 2), (15, 6): (12, 5), (15, 7): (10, 3), (15, 8): (16, 10), (15, 9): (6, 1),
    (16, 0): (10, 2), (16, 1): (14, 1), (16, 2): (8, 3), (16, 3): (12, 1), (16, 4): (7, 2),
    (16, 5): (11, 1), (16, 6): (13, 3), (16, 7): (9, 1), (16, 8): (15, 2), (16, 9): (6, 1),
    (17, 0): (16, 8), (17, 1): (9, 4), (17, 2): (7, 2), (17, 3): (11, 6), (17, 4): (13, 7),
    (17, 5): (6, 3), (17, 6): (10, 5), (17, 7): (14, 9), (17, 8): (8, 4), (17, 9): (12, 7),
    (18, 0): (11, 1), (18, 1): (13, 3), (18, 2): (9, 2), (18, 3): (15, 1), (18, 4): (8, 3),
    (18, 5): (10, 2), (18, 6): (12, 1), (18, 7): (7, 3), (18, 8): (14, 2), (18, 9): (6, 1),
    (19, 0): (8, 4), (19, 1): (10, 6), (19, 2): (12, 8), (19, 3): (6, 3), (19, 4): (14, 9),
    (19, 5): (9, 5), (19, 6): (7, 2), (19, 7): (11, 7), (19, 8): (15, 10), (19, 9): (13, 8),
}

P_run = [1.0] * n_stages
P_sb  = [0.5] * n_stages

complex_operations = {(3, j) for j in range(n_stages)}

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
nsga2_front = nsga2_solver.non_dominated_sort(nsga2_pop)[0]

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

# === Fonction pour extraire les points ===
def get_points(front):
    return [sol.objectives[0] for sol in front], [sol.objectives[1] for sol in front]

m_nsga2, tec_nsga2 = get_points(nsga2_front)
m_spea2, tec_spea2 = get_points(spea2_front)
m_moead, tec_moead = get_points(moead_front)

# === Normalisation HV et GD entre 0 et 1 ===
def normalize(metric_list, max_val=None):
    if max_val is None:
        max_val = max(metric_list)
    return [val / max_val if max_val != 0 else 0 for val in metric_list]

# Normalisation HV globale
all_hv = nsga2_solver.hv_log + spea2_solver.hv_log + moead_solver.hv_log
hv_max = max(all_hv)

nsga2_solver.hv_log = normalize(nsga2_solver.hv_log, hv_max)
spea2_solver.hv_log = normalize(spea2_solver.hv_log, hv_max)
moead_solver.hv_log = normalize(moead_solver.hv_log, hv_max)

# Normalisation GD globale
all_gd = nsga2_solver.gd_log + spea2_solver.gd_log + moead_solver.gd_log
gd_max = max(all_gd)

nsga2_solver.gd_log = normalize(nsga2_solver.gd_log, gd_max)
spea2_solver.gd_log = normalize(spea2_solver.gd_log, gd_max)
moead_solver.gd_log = normalize(moead_solver.gd_log, gd_max)

# === Styles ===
colors = {"NSGA-II": "red", "SPEA2": "blue", "MOEA/D": "magenta"}
markers = {"NSGA-II": "o", "SPEA2": "s", "MOEA/D": "^"}
metric_style = {'linewidth': 2, 'markersize': 8, 'alpha': 0.7}

# ==================== Figure 1 : Pareto Fronts ====================
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
plt.title("Normalized Hypervolume Evolution", fontsize=14)

plt.plot(nsga2_solver.hv_log, marker='o', color=colors["NSGA-II"], label='NSGA-II', **metric_style)
plt.plot(spea2_solver.hv_log, marker='o', color=colors["SPEA2"], label='SPEA2', **metric_style)
plt.plot(moead_solver.hv_log, marker='o', color=colors["MOEA/D"], label='MOEA/D', **metric_style)

plt.xlabel("Generations (per 5)", fontsize=12)
plt.ylabel("Normalized HV (0–1)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==================== Figure 3 : Generational Distance (GD) ====================
plt.figure(figsize=(8, 6))
plt.title("Normalized Generational Distance (GD) Evolution", fontsize=14)

plt.plot(nsga2_solver.gd_log, marker='s', color=colors["NSGA-II"], label='NSGA-II', **metric_style)
plt.plot(spea2_solver.gd_log, marker='s', color=colors["SPEA2"], label='SPEA2', **metric_style)
plt.plot(moead_solver.gd_log, marker='s', color=colors["MOEA/D"], label='MOEA/D', **metric_style)

plt.xlabel("Generations (per 5)", fontsize=12)
plt.ylabel("Normalized GD (0–1)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==================== Résultats finaux ====================
best_hv_nsga2 = max(nsga2_solver.hv_log)
best_gd_nsga2 = min(nsga2_solver.gd_log)

best_hv_spea2 = max(spea2_solver.hv_log)
best_gd_spea2 = min(spea2_solver.gd_log)

best_hv_moead = max(moead_solver.hv_log)
best_gd_moead = min(moead_solver.gd_log)

print("\n" + "="*60)
print("Final Normalized Performance Comparison".center(60))
print("="*60)
print(f"{'Algorithm':<12} | {'Best HV':<12} | {'Best GD':<12}")
print("-"*60)
print(f"{'NSGA-II':<12} | {best_hv_nsga2:^12.4f} | {best_gd_nsga2:^12.4f}")
print(f"{'SPEA2':<12} | {best_hv_spea2:^12.4f} | {best_gd_spea2:^12.4f}")
print(f"{'MOEA/D':<12} | {best_hv_moead:^12.4f} | {best_gd_moead:^12.4f}")
print("="*60 + "\n")

