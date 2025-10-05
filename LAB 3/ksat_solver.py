import random
import time
import sys
import matplotlib.pyplot as plt

sys.setrecursionlimit(2000)

def generate_k_sat(k, m, n):
    if k > n:
        raise ValueError("k cannot be greater than n")
    
    formula = []
    variables = list(range(1, n + 1))
    
    for _ in range(m):
        chosen_vars = random.sample(variables, k)
        clause = [var if random.choice([True, False]) else -var for var in chosen_vars]
        formula.append(clause)
    
    return formula

def heuristic_satisfied_clauses(formula, assignment):
    """Counts number of satisfied clauses (heuristic 1)"""
    satisfied_count = 0
    for clause in formula:
        if any((literal > 0 and assignment[abs(literal)-1]) or
               (literal < 0 and not assignment[abs(literal)-1]) for literal in clause):
            satisfied_count += 1
    return satisfied_count

def heuristic_unsatisfied_penalty(formula, assignment):
    """Penalizes unsatisfied clauses (heuristic 2)"""
    unsatisfied_count = 0
    for clause in formula:
        if not any((literal > 0 and assignment[abs(literal)-1]) or
                   (literal < 0 and not assignment[abs(literal)-1]) for literal in clause):
            unsatisfied_count += 1
    return -unsatisfied_count  # negative for maximization

def hill_climbing_solver(formula, n, heuristic_func, max_iterations=10000):
    start_time = time.time()
    assignment = [random.choice([True, False]) for _ in range(n)]
    score = heuristic_func(formula, assignment)
    path_length = 1
    states_visited = 1
    num_clauses = len(formula)

    for _ in range(max_iterations):
        neighbor = list(assignment)
        flip_index = random.randint(0, n-1)
        neighbor[flip_index] = not neighbor[flip_index]
        states_visited += 1
        neighbor_score = heuristic_func(formula, neighbor)
        if neighbor_score >= score:
            assignment = neighbor
            score = neighbor_score
            path_length += 1
        if heuristic_func(formula, assignment) == num_clauses:
            break

    end_time = time.time()
    penetrance = path_length / states_visited if states_visited > 0 else 0
    return {"solution_found": heuristic_func(formula, assignment)==num_clauses,
            "satisfied_clauses": sum(1 for clause in formula if any(
                (lit>0 and assignment[abs(lit)-1]) or (lit<0 and not assignment[abs(lit)-1])
            for lit in clause)),
            "time": end_time - start_time,
            "penetrance": penetrance}

def beam_search_solver(formula, n, heuristic_func, beam_width=3, max_iterations=100):
    start_time = time.time()
    beam = [[random.choice([True, False]) for _ in range(n)] for _ in range(beam_width)]
    states_visited = beam_width
    num_clauses = len(formula)

    for i in range(max_iterations):
        all_neighbors = []
        for assignment in beam:
            score = heuristic_func(formula, assignment)
            if score == num_clauses:
                end_time = time.time()
                return {"solution_found": True,
                        "satisfied_clauses": sum(1 for clause in formula if any(
                            (lit>0 and assignment[abs(lit)-1]) or (lit<0 and not assignment[abs(lit)-1])
                        for lit in clause)),
                        "time": end_time-start_time,
                        "penetrance": (i+1)/states_visited}
            for flip_index in range(n):
                neighbor = list(assignment)
                neighbor[flip_index] = not neighbor[flip_index]
                all_neighbors.append(neighbor)
        states_visited += len(all_neighbors)
        unique_neighbors = [list(x) for x in set(tuple(x) for x in all_neighbors)]
        unique_neighbors.sort(key=lambda a: heuristic_func(formula, a), reverse=True)
        beam = unique_neighbors[:beam_width]

    end_time = time.time()
    best_assignment = beam[0] if beam else [False]*n
    final_score = heuristic_func(formula, best_assignment)
    penetrance = max_iterations / states_visited if states_visited>0 else 0
    return {"solution_found": final_score==num_clauses,
            "satisfied_clauses": sum(1 for clause in formula if any(
                (lit>0 and best_assignment[abs(lit)-1]) or (lit<0 and not best_assignment[abs(lit)-1])
            for lit in clause)),
            "time": end_time-start_time,
            "penetrance": penetrance}

# Variable Neighborhood Descent
def neighborhood_flip_one(assignment, n):
    neighbor = list(assignment)
    idx = random.randint(0,n-1)
    neighbor[idx] = not neighbor[idx]
    return neighbor

def neighborhood_flip_two(assignment, n):
    neighbor = list(assignment)
    idx1, idx2 = random.sample(range(n),2)
    neighbor[idx1] = not neighbor[idx1]
    neighbor[idx2] = not neighbor[idx2]
    return neighbor

def neighborhood_swap_two(assignment, n):
    neighbor = list(assignment)
    idx1, idx2 = random.sample(range(n),2)
    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
    return neighbor

def vnd_solver(formula, n, heuristic_func, max_iterations=10000):
    start_time = time.time()
    neighborhoods = [neighborhood_flip_one, neighborhood_flip_two, neighborhood_swap_two]
    assignment = [random.choice([True, False]) for _ in range(n)]
    score = heuristic_func(formula, assignment)
    path_length = 1
    states_visited = 1
    i = 0
    num_clauses = len(formula)

    while i < max_iterations:
        k = 0
        while k < len(neighborhoods):
            neighbor = neighborhoods[k](assignment, n)
            states_visited += 1
            i += 1
            neighbor_score = heuristic_func(formula, neighbor)
            if neighbor_score >= score:
                assignment = neighbor
                score = neighbor_score
                path_length += 1
                k = 0
            else:
                k += 1
            if score == num_clauses: break
        if score == num_clauses: break

    end_time = time.time()
    penetrance = path_length / states_visited if states_visited>0 else 0
    return {"solution_found": score==num_clauses,
            "satisfied_clauses": sum(1 for clause in formula if any(
                (lit>0 and assignment[abs(lit)-1]) or (lit<0 and not assignment[abs(lit)-1])
            for lit in clause)),
            "time": end_time-start_time,
            "penetrance": penetrance}

def plot_results(all_results, test_cases, heuristic_name):
    algorithms = list(all_results[0].keys())
    x_labels = [m for m,n in test_cases]

    # Execution Time
    plt.figure(figsize=(10,6))
    for alg in algorithms:
        plt.plot(x_labels, [res[alg]['time'] for res in all_results], marker='o', label=alg)
    plt.title(f'Execution Time vs Clauses ({heuristic_name})')
    plt.xlabel('Number of Clauses')
    plt.ylabel('Time (s)')
    plt.legend(); plt.grid(True); plt.show()

    # Penetrance
    plt.figure(figsize=(10,6))
    for alg in algorithms:
        plt.plot(x_labels, [res[alg]['penetrance'] for res in all_results], marker='o', label=alg)
    plt.title(f'Penetrance vs Clauses ({heuristic_name})')
    plt.xlabel('Number of Clauses')
    plt.ylabel('Penetrance')
    plt.legend(); plt.grid(True); plt.show()

    # Solution Quality
    plt.figure(figsize=(10,6))
    for alg in algorithms:
        plt.plot(x_labels, [(res[alg]['satisfied_clauses']/m)*100 for res,(m,n) in zip(all_results,test_cases)], marker='o', label=alg)
    plt.title(f'Solution Quality (%) vs Clauses ({heuristic_name})')
    plt.xlabel('Number of Clauses')
    plt.ylabel('Clauses Satisfied (%)')
    plt.ylim(0,105)
    plt.legend(); plt.grid(True); plt.show()

if __name__ == "__main__":
    K = 3
    test_cases = [(40,10),(100,20),(200,50),(425,100)]
    heuristics = [("Satisfied Clauses", heuristic_satisfied_clauses),
                  ("Unsatisfied Penalty", heuristic_unsatisfied_penalty)]

    for heuristic_name, heuristic_func in heuristics:
        print("\n"+"="*50)
        print(f"Running all solvers using heuristic: {heuristic_name}")
        print("="*50)
        all_results = []

        for m,n in test_cases:
            print(f"\nGenerating problem with {m} clauses and {n} variables")
            formula = generate_k_sat(K, m, n)
            results = {}
            results['Hill-Climbing'] = hill_climbing_solver(formula, n, heuristic_func)
            results['Beam-Search (w=3)'] = beam_search_solver(formula, n, heuristic_func, beam_width=3)
            results['Beam-Search (w=4)'] = beam_search_solver(formula, n, heuristic_func, beam_width=4)
            results['VND'] = vnd_solver(formula, n, heuristic_func)
            all_results.append(results)

            # Print table
            print(f"{'Algorithm':<20} | {'Solution Found?':<16} | {'Satisfied Clauses':<20} | {'Time (s)':<10} | {'Penetrance':<15}")
            print("-"*90)
            for name,res in results.items():
                found_str = "Yes" if res['solution_found'] else "No"
                satisfied_str = f"{res['satisfied_clauses']}/{m}"
                print(f"{name:<20} | {found_str:<16} | {satisfied_str:<20} | {res['time']:.4f}    | {res['penetrance']:.6f}")

        # Ploted line graphs for this heuristic
        plot_results(all_results, test_cases, heuristic_name)
