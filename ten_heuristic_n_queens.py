import heapq
import time
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# کلاس وضعیت برای الگوریتم A*
# -----------------------------------------------------------------------------
class State:
    def __init__(self, board, cost=0, parent=None, action=None):
        self.board = board.copy() # وضعیت فعلی صفحه (لیستی از موقعیت سطر وزیران در هر ستون)
        self.cost = cost          # g(n): هزینه واقعی از شروع تا این وضعیت (تعداد وزیران قرار داده شده)
        self.parent = parent      # گره پدر در درخت جستجو
        self.action = action      # عملی که منجر به این وضعیت شده (موقعیت وزیر قرار داده شده)

    # برای مقایسه در صف اولویت heapq
    def __lt__(self, other):
        # این تابع صرفا برای کارکرد صحیح heapq است و f_value در خود حلقه A* مقایسه می شود
        return self.cost < other.cost

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return self.board == other.board

    def __hash__(self):
        # برای استفاده در closed_set
        return hash(tuple(self.board))

# -----------------------------------------------------------------------------
# توابع هیوریستیک (h(n))
# board: لیستی که وضعیت جزئی یا کامل صفحه را نشان می‌دهد.
# board[i] = -1 یعنی ستون i هنوز وزیری ندارد.
# -----------------------------------------------------------------------------

def h1_attacking_pairs(board):
    n = len(board)
    threats = 0
    placed_queens_pos = []
    for col, row in enumerate(board):
        if row != -1:
            placed_queens_pos.append((col, row))

    for i in range(len(placed_queens_pos)):
        for j in range(i + 1, len(placed_queens_pos)):
            col1, row1 = placed_queens_pos[i]
            col2, row2 = placed_queens_pos[j]
            if row1 == row2 or abs(row1 - row2) == abs(col1 - col2):
                threats += 1
    return threats

def h2_min_conflicts_for_next_queen(board):
    n = len(board)
    next_col = -1
    for i in range(n):
        if board[i] == -1:
            next_col = i
            break
    if next_col == -1: # همه وزیران قرار گرفته‌اند
        return h1_attacking_pairs(board) # اگر کامل است، تعداد تهدیدهای فعلی را برگردان

    min_conflicts_for_this_col = float('inf')
    possible_moves_in_col = 0
    for r_next in range(n):
        current_conflicts = 0
        # بررسی تعارض با وزیران قبلی
        for c_prev in range(next_col):
            r_prev = board[c_prev]
            if r_prev == r_next or abs(r_prev - r_next) == abs(c_prev - next_col):
                current_conflicts += 1
        min_conflicts_for_this_col = min(min_conflicts_for_this_col, current_conflicts)
        possible_moves_in_col +=1

    if possible_moves_in_col == 0 : # این حالت نباید رخ دهد اگر next_col معتبر باشد
         return float('inf')
    return min_conflicts_for_this_col


def h3_max_future_freedom(board):
    n = len(board)
    next_col_to_fill = -1
    for i in range(n):
        if board[i] == -1:
            next_col_to_fill = i
            break
    if next_col_to_fill == -1: # Board is full
        return h1_attacking_pairs(board)

    threatened_count = 0 # تعداد خانه‌های تهدید شده در ستون‌های باقی‌مانده
    remaining_cols = n - next_col_to_fill
    
    # ایجاد یک کپی از صفحه برای بررسی تهدیدها بدون تغییر وضعیت اصلی
    temp_board_for_checking = [[False for _ in range(n)] for _ in range(n)]

    # نشانه‌گذاری خانه‌های تهدید شده توسط وزیران فعلی
    for c_placed in range(next_col_to_fill):
        r_placed = board[c_placed]
        for r_target in range(n): # تهدید سطری
            if not temp_board_for_checking[r_target][c_placed]: # فقط برای ستون‌های آتی
                 for c_future in range(next_col_to_fill, n):
                      if r_target == r_placed:
                           temp_board_for_checking[r_target][c_future] = True
        for c_target in range(n): # تهدید ستونی (برای ستون‌های آتی معنی ندارد چون در هر ستون یک وزیر است)
            pass
        # تهدید قطری
        for r_target in range(n):
            for c_target in range(next_col_to_fill, n):
                if abs(r_placed - r_target) == abs(c_placed - c_target):
                    temp_board_for_checking[r_target][c_target] = True
    
    safe_cells_in_future = 0
    for r_ in range(n):
        for c_ in range(next_col_to_fill, n):
            if not temp_board_for_checking[r_][c_]:
                safe_cells_in_future += 1
    
    # می‌خواهیم خانه‌های امن را حداکثر کنیم، پس هزینه، معکوس آن است
    # یا (تعداد کل خانه‌های ممکن در آینده - خانه‌های امن)
    max_possible_future_cells = n * remaining_cols
    return max_possible_future_cells - safe_cells_in_future

def h4_threat_density(board):
    n = len(board)
    next_col_to_fill = -1
    num_placed_queens = 0
    for i in range(n):
        if board[i] == -1 and next_col_to_fill == -1:
            next_col_to_fill = i
        if board[i] != -1:
            num_placed_queens +=1

    if num_placed_queens == n: # Board is full
        return h1_attacking_pairs(board)
    if next_col_to_fill == -1 and num_placed_queens < n: # should not happen, means error in logic
        return float('inf')


    threat_counts_in_future_cols = [[0 for _ in range(n)] for _ in range(n)]
    
    # محاسبه تعداد تهدیدها برای هر خانه در ستون‌های آتی از وزیران قرار داده شده
    for c_placed in range(num_placed_queens): # Iterate only over placed queens
        if board[c_placed] == -1: continue # Skip if this col was skipped for some reason
        r_placed = board[c_placed]
        
        for c_future in range(next_col_to_fill if next_col_to_fill !=-1 else c_placed + 1, n):
            for r_future in range(n):
                # تهدید سطری
                if r_placed == r_future:
                    threat_counts_in_future_cols[r_future][c_future] += 1
                # تهدید قطری
                if abs(r_placed - r_future) == abs(c_placed - c_future):
                    threat_counts_in_future_cols[r_future][c_future] += 1
    
    total_threat_potential = 0
    num_future_cells = 0
    for r_ in range(n):
        for c_ in range(next_col_to_fill if next_col_to_fill !=-1 else n, n): # iterate from next_col_to_fill or end if full
            total_threat_potential += threat_counts_in_future_cols[r_][c_]
            num_future_cells +=1
            
    if num_future_cells == 0:
        return 0 # No future cells to evaluate, or board is full and handled
    return total_threat_potential / num_future_cells


def h5_mean_distance_from_center(board):
    n = len(board)
    center_r = (n - 1) / 2.0
    center_c = (n - 1) / 2.0 # ستون‌ها هم از 0 تا n-1 هستند
    
    total_manhattan_distance = 0
    placed_count = 0
    for c, r in enumerate(board):
        if r != -1:
            total_manhattan_distance += abs(r - center_r) + abs(c - center_c)
            placed_count += 1
            
    if placed_count == 0:
        return (n-1) # Max possible average distance if only one queen can be placed at corner
    
    # اگر هنوز همه وزیران قرار نگرفته‌اند، می‌توانیم برای بقیه هم یک تخمین اضافه کنیم
    # مثلا فرض کنیم بقیه هم همین میانگین فاصله را خواهند داشت
    estimated_remaining_distance = ((n-1) / 2.0) * (n - placed_count) # میانگین فاصله یک خانه تا مرکز
    
    # هدف کمینه کردن این فاصله است
    return (total_manhattan_distance + estimated_remaining_distance) / n


def h6_queen_dispersion(board): # (Maximize, so return negative or Max - value)
    n = len(board)
    placed_queens_pos = []
    for col, row in enumerate(board):
        if row != -1:
            placed_queens_pos.append((col, row))

    if len(placed_queens_pos) < 2:
        return n * np.sqrt(2*(n-1)**2) # Max possible sum of distances if queens are at opposite corners

    sum_distances = 0
    num_pairs = 0
    for i in range(len(placed_queens_pos)):
        for j in range(i + 1, len(placed_queens_pos)):
            c1, r1 = placed_queens_pos[i]
            c2, r2 = placed_queens_pos[j]
            distance = np.sqrt((c1 - c2)**2 + (r1 - r2)**2)
            sum_distances += distance
            num_pairs +=1
            
    # Heuristic should estimate remaining cost.
    # If we want to maximize dispersion, A* needs a cost to minimize.
    # So, (Max_possible_sum_distances - current_sum_distances)
    # Max_possible_sum_distances is hard to define for partial board.
    # Let's use 1/dispersion or a large_number - dispersion
    if sum_distances == 0 :
         return n * n # A large number if no dispersion yet
    return 1.0 / (sum_distances / max(1,num_pairs) + 1e-6) # Add small epsilon to avoid division by zero, +1 to make it positive

def h7_quadrant_balance(board):
    n = len(board)
    mid_r, mid_c = n // 2, n // 2
    
    q_counts = [0, 0, 0, 0] # Q1: top-left, Q2: top-right, Q3: bottom-left, Q4: bottom-right
    placed_count = 0
    
    for c, r in enumerate(board):
        if r != -1:
            placed_count += 1
            if r < mid_r and c < mid_c: q_counts[0] += 1
            elif r < mid_r and c >= mid_c: q_counts[1] += 1
            elif r >= mid_r and c < mid_c: q_counts[2] += 1
            else: q_counts[3] += 1 # r >= mid_r and c >= mid_c
            
    if placed_count == 0:
        return n # Max possible deviation if no queens
        
    ideal_per_q = placed_count / 4.0
    deviation = sum(abs(count - ideal_per_q) for count in q_counts)
    
    # Estimate for remaining queens
    remaining_queens = n - placed_count
    # Assume remaining queens could be placed to minimize deviation,
    # but this is complex. A simpler h is just current deviation.
    # Or, project that remaining queens will also deviate.
    # For A*, we need an underestimate of cost (or overestimate of "goodness" if we flip).
    # Let's return current deviation, A* will try to minimize it.
    return deviation


def h8_sum_of_squared_distances(board): # (Maximize, so return negative or Max - value)
    n = len(board)
    placed_queens_pos = []
    for col, row in enumerate(board):
        if row != -1:
            placed_queens_pos.append((col, row))

    if len(placed_queens_pos) < 2:
        return n * (2 * (n-1)**2) # Max possible sum of squared distances

    sum_sq_distances = 0
    num_pairs = 0
    for i in range(len(placed_queens_pos)):
        for j in range(i + 1, len(placed_queens_pos)):
            c1, r1 = placed_queens_pos[i]
            c2, r2 = placed_queens_pos[j]
            sq_distance = (c1 - c2)**2 + (r1 - r2)**2
            sum_sq_distances += sq_distance
            num_pairs +=1
            
    # Similar to H6, return 1/value or Max_val - value
    if sum_sq_distances == 0:
        return n*n*n # A large number
    return 1.0 / (sum_sq_distances / max(1,num_pairs) + 1e-6)


def h9_diagonal_interference_weighted(board):
    n = len(board)
    threats = 0
    placed_queens_pos = []
    for col, row in enumerate(board):
        if row != -1:
            placed_queens_pos.append((col, row))

    for i in range(len(placed_queens_pos)):
        for j in range(i + 1, len(placed_queens_pos)):
            col1, row1 = placed_queens_pos[i]
            col2, row2 = placed_queens_pos[j]
            if row1 == row2: # Row conflict
                threats += 1
            if abs(row1 - row2) == abs(col1 - col2): # Diagonal conflict
                threats += 2 # Weight diagonal conflicts more
    return threats

def h10_combined(board):
    n = len(board)
    if n == 0: return 0

    # Weights (example, can be tuned)
    w1, w2, w3, w4, w5, w6, w7, w8, w9 = 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15

    # Max possible values for normalization (approximate)
    max_h1 = n * (n - 1) / 2
    max_h2 = n -1 # Max conflicts for one queen in a col
    max_h3 = n * n # Max (total cells - safe cells)
    max_h4 = n # Max avg threat density (if all cells are threatened by n-1 queens)
    max_h5 = n # Max avg manhattan distance
    max_h6_inv = n*n # Inverse of min dispersion (large number)
    max_h7 = n # Max deviation
    max_h8_inv = n*n*n # Inverse of min sum_sq_dist
    max_h9 = (n * (n - 1) / 2) * 2 # Max weighted threats

    # Calculate and normalize (0 to 1, where 0 is better)
    # Ensure denominators are not zero
    v1 = h1_attacking_pairs(board) / max(1, max_h1)
    v2 = h2_min_conflicts_for_next_queen(board) / max(1, max_h2)
    v3 = h3_max_future_freedom(board) / max(1, max_h3)
    v4 = h4_threat_density(board) / max(1, max_h4)
    v5 = h5_mean_distance_from_center(board) / max(1, max_h5)
    v6 = h6_queen_dispersion(board) # Already inverted, smaller is better
    v7 = h7_quadrant_balance(board) / max(1, max_h7)
    v8 = h8_sum_of_squared_distances(board) # Already inverted
    v9 = h9_diagonal_interference_weighted(board) / max(1, max_h9)
    
    # Some heuristics like h6 and h8 are already inverted (smaller is better after inversion)
    # We want to minimize the combined heuristic
    combined_h = (w1*v1 + w2*v2 + w3*v3 + w4*v4 + w5*v5 + 
                   w6*v6 + w7*v7 + w8*v8 + w9*v9)
    return combined_h

# -----------------------------------------------------------------------------
# توابع کمکی برای A*
# -----------------------------------------------------------------------------
def is_safe_partial(board, r, c_new):
    # بررسی آیا قرار دادن وزیر در (r, c_new) با وزیران قبلی در ستون‌های c < c_new تداخل دارد
    for c_prev in range(c_new):
        r_prev = board[c_prev]
        if r_prev == r or abs(r_prev - r) == abs(c_prev - c_new):
            return False
    return True

def is_complete_and_valid_solution(board):
    n = len(board)
    if -1 in board: # اگر هنوز همه‌ی وزیران قرار نگرفته‌اند
        return False
    return h1_attacking_pairs(board) == 0 # اگر کامل است، باید هیچ تهدیدی نباشد

# -----------------------------------------------------------------------------
# الگوریتم A*
# -----------------------------------------------------------------------------
def solve_n_queens_astar(n, heuristic_func):
    initial_board = [-1] * n
    # (f_value, tie_breaker (e.g. current_steps), state_object)
    # tie_breaker is important for heapq when f_values are equal
    open_set = [(heuristic_func(initial_board), 0, State(initial_board, 0))] 
    heapq.heapify(open_set)
    
    closed_set = set() # Store hash of boards
    
    g_costs = {tuple(initial_board): 0} # Store g(n) for visited states
    
    
    steps = 0 # تعداد گره‌های بسط داده شده

    while open_set:
        f_val, _, current_s_obj = heapq.heappop(open_set)
        current_board_tuple = tuple(current_s_obj.board)

        if current_board_tuple in closed_set:
            continue
        closed_set.add(current_board_tuple)
        steps += 1 # گره از صف برداشته و پردازش می‌شود

        if is_complete_and_valid_solution(current_s_obj.board):
            return current_s_obj.board, steps

        # پیدا کردن اولین ستون خالی برای قرار دادن وزیر بعدی
        next_col = -1
        try:
            next_col = current_s_obj.board.index(-1)
        except ValueError: # No -1 found, board is full (but not valid if we are here)
            continue 
        
        # تولید جانشین‌ها
        for r in range(n):
            if is_safe_partial(current_s_obj.board, r, next_col):
                new_board = current_s_obj.board[:] # Create a copy
                new_board[next_col] = r
                
                new_g_cost = current_s_obj.cost + 1 # g(n) is number of queens placed
                new_board_tuple = tuple(new_board)

                if new_board_tuple in closed_set and new_g_cost >= g_costs.get(new_board_tuple, float('inf')):
                    continue
                
                h_val = heuristic_func(new_board)
                new_f_val = new_g_cost + h_val
                
                # Add to open_set if new path is better or not visited
                if new_g_cost < g_costs.get(new_board_tuple, float('inf')):
                    g_costs[new_board_tuple] = new_g_cost
                    heapq.heappush(open_set, (new_f_val, steps, State(new_board, new_g_cost, current_s_obj, (next_col,r))))
                    
    return None, steps # No solution found

# -----------------------------------------------------------------------------
# توابع ارزیابی و رسم نمودار
# -----------------------------------------------------------------------------
def print_solution_board(board_solution, n):
    if board_solution:
        print(f"Solution for N={n}: {board_solution}")
        for r in range(n):
            line = ""
            for c in range(n):
                if board_solution[c] == r:
                    line += "Q "
                else:
                    line += ". "
            print(line)
        print("-" * 2 * n)
    else:
        print(f"No solution found for N={n} with this heuristic/limit.")

def evaluate_heuristics(heuristics_dict, n_values_list):
    results_steps = {name: [] for name in heuristics_dict.keys()}
    results_times = {name: [] for name in heuristics_dict.keys()}

    for n_val in n_values_list:
        print(f"\n--- Evaluating for N = {n_val} ---")
        for h_name, h_func in heuristics_dict.items():
            print(f"  Running Heuristic: {h_name}...")
            start_time = time.time()
            solution, steps = solve_n_queens_astar(n_val, h_func)
            end_time = time.time()
            elapsed_time = end_time - start_time

            results_steps[h_name].append(steps)
            results_times[h_name].append(elapsed_time)
            
            print(f"    Solution: {'Found' if solution else 'Not Found'}, Steps: {steps}, Time: {elapsed_time:.4f}s")
            # if solution: print_solution_board(solution, n_val) # Uncomment to print boards
            
    return results_steps, results_times

def plot_results(data_dict, n_values_list, title, ylabel, filename_suffix):
    plt.figure(figsize=(12, 7))
    for h_name, values in data_dict.items():
        plt.plot(n_values_list, values, marker='o', linestyle='-', label=h_name)
    
    plt.xlabel("N (Board Size)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(n_values_list)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1)) # Legend outside
    plt.grid(True)
    plt.tight_layout(rect=[0,0,0.85,1]) # Adjust layout to make space for legend
    plt.savefig(f"n_queens_{filename_suffix}_comparison.png")
    plt.show()
    plt.close()

def plot_bar_results(data_dict, n_values_list, title, ylabel, filename_suffix):
    num_heuristics = len(data_dict)
    num_n_values = len(n_values_list)
    
    bar_width = 0.8 / num_heuristics 
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    indices = np.arange(num_n_values)
    
    for i, (h_name, values) in enumerate(data_dict.items()):
        ax.bar(indices + i * bar_width, values, bar_width, label=h_name)
        
    ax.set_xlabel("N (Board Size)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(indices + bar_width * (num_heuristics - 1) / 2)
    ax.set_xticklabels(n_values_list)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax.grid(True, axis='y')
    fig.tight_layout(rect=[0,0,0.85,1])
    plt.savefig(f"n_queens_{filename_suffix}_bar_comparison.png")
    plt.show()
    plt.close()

# -----------------------------------------------------------------------------
# اجرای اصلی
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # تعریف دیکشنری هیوریستیک‌ها
    heuristics_to_test = {
        'H1: Attacking Pairs': h1_attacking_pairs,
        'H2: Min Conflicts Next': h2_min_conflicts_for_next_queen,
        'H3: Max Future Freedom': h3_max_future_freedom,
        'H4: Threat Density': h4_threat_density,
        'H5: Mean Dist Center': h5_mean_distance_from_center,
        'H6: Queen Dispersion (Inv)': h6_queen_dispersion,
        'H7: Quadrant Balance': h7_quadrant_balance,
        'H8: Sum Sq Dist (Inv)': h8_sum_of_squared_distances,
        'H9: Weighted Diag Interference': h9_diagonal_interference_weighted,
        'H10: Combined': h10_combined
    }

    # اندازه‌های صفحه برای آزمایش
    n_values_to_run = [4, 6, 8, 10] # For larger N, it can be very slow

    # ارزیابی
    all_steps, all_times = evaluate_heuristics(heuristics_to_test, n_values_to_run)

    # چاپ جداول خلاصه (شبیه به چیزی که در تصویر فرستادید)
    print("\n--- Summary Table: Steps ---")
    print(f"{'Heuristic':<30} | {'N=4':<5} | {'N=6':<5} | {'N=8':<5} | {'N=10':<6} | {'Avg':<7} | Min | Max")
    print("-" * 80)
    for h_name in heuristics_to_test.keys():
        steps = all_steps[h_name]
        avg_steps = np.mean(steps) if steps else 0
        min_steps = min(steps) if steps else 0
        max_steps = max(steps) if steps else 0
        print(f"{h_name:<30} | {steps[0]:<5} | {steps[1]:<5} | {steps[2]:<5} | {steps[3]:<6} | {avg_steps:<7.2f} | {min_steps} | {max_steps}")

    print("\n--- Summary Table: Time (s) ---")
    print(f"{'Heuristic':<30} | {'N=4':<5} | {'N=6':<5} | {'N=8':<5} | {'N=10':<6} | {'Avg':<7} | Min | Max")
    print("-" * 80)
    for h_name in heuristics_to_test.keys():
        times = all_times[h_name]
        avg_times = np.mean(times) if times else 0
        min_times = min(times) if times else 0
        max_times = max(times) if times else 0
        print(f"{h_name:<30} | {times[0]:<5.2f} | {times[1]:<5.2f} | {times[2]:<5.2f} | {times[3]:<6.2f} | {avg_times:<7.2f} | {min_times:.2f} | {max_times:.2f}")


    # رسم نمودارها
    plot_results(all_steps, n_values_to_run, "N-Queens: Steps Comparison", "Number of Steps (Expanded Nodes)", "steps")
    plot_results(all_times, n_values_to_run, "N-Queens: Execution Time Comparison", "Time (seconds)", "time")
    plot_bar_results(all_steps, n_values_to_run, "N-Queens: Steps Bar Comparison", "Number of Steps (Expanded Nodes)", "steps_bar")
    plot_bar_results(all_times, n_values_to_run, "N-Queens: Execution Time Bar Comparison", "Time (seconds)", "time_bar")

    print("\nEvaluation complete. Check for PNG files for plots.")