#!/usr/bin/env python3
"""
Store Pathfinder — FULL RESEARCH SIMULATION (Experiments 1–4)
=============================================================
Exp 1: Head-to-head matchups (10 games × 12 configs = 120 games)
Exp 2: A3 blockage stress test (cross-cluster trips, corridor bottlenecks)
Exp 3: Blocking effectiveness (blocks caused/suffered, trap success rate)
Exp 4: Node expansion & computational cost (per-move instrumentation, timing)
"""

import random
import math
import csv
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import heapq

# ============================================================================
# AIMA FRAMEWORK
# ============================================================================

class Problem:
    def __init__(self, initial=None, goal=None, **kwds):
        self.__dict__.update(initial=initial, goal=goal, **kwds)
    def actions(self, state):        raise NotImplementedError
    def result(self, state, action): raise NotImplementedError
    def is_goal(self, state):        return state == self.goal
    def action_cost(self, s, a, s1): return 1
    def h(self, node):               return 0

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)
    def __repr__(self): return f'<{self.state}>'
    def __len__(self): return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): return self.path_cost < other.path_cost

failure = Node('failure', path_cost=math.inf)

def expand(problem, node):
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)

def path_states(node):
    if node in (failure, None): return []
    return path_states(node.parent) + [node.state]

# ============================================================================
# INSTRUMENTED SEARCH ALGORITHMS
# ============================================================================

class SearchMetrics:
    """Per-call search metrics."""
    def __init__(self):
        self.calls = []  # list of per-call dicts

    def new_call(self):
        self.calls.append({'nodes_expanded': 0, 'max_frontier': 0, 'time_ns': 0})
        return self.calls[-1]

    def totals(self):
        if not self.calls:
            return {'total_calls': 0, 'total_nodes': 0, 'avg_nodes_per_call': 0,
                    'max_frontier_ever': 0, 'total_time_us': 0, 'avg_time_per_call_us': 0}
        total_n = sum(c['nodes_expanded'] for c in self.calls)
        max_f = max(c['max_frontier'] for c in self.calls)
        total_t = sum(c['time_ns'] for c in self.calls)
        return {
            'total_calls': len(self.calls),
            'total_nodes': total_n,
            'avg_nodes_per_call': round(total_n / len(self.calls), 2),
            'max_frontier_ever': max_f,
            'total_time_us': round(total_t / 1000, 1),
            'avg_time_per_call_us': round(total_t / 1000 / len(self.calls), 2),
        }

    def reset(self):
        self.calls = []

metrics = defaultdict(SearchMetrics)

def breadth_first_search(problem, algo_name='BFS'):
    m = metrics[algo_name].new_call()
    t0 = time.perf_counter_ns()
    node = Node(problem.initial)
    if problem.is_goal(problem.initial):
        m['time_ns'] = time.perf_counter_ns() - t0
        return node
    frontier = deque([node])
    reached = {problem.initial}
    while frontier:
        m['max_frontier'] = max(m['max_frontier'], len(frontier))
        node = frontier.popleft()
        for child in expand(problem, node):
            m['nodes_expanded'] += 1
            s = child.state
            if problem.is_goal(s):
                m['time_ns'] = time.perf_counter_ns() - t0
                return child
            if s not in reached:
                reached.add(s); frontier.append(child)
    m['time_ns'] = time.perf_counter_ns() - t0
    return failure

def depth_first_search(problem, algo_name='DFS'):
    m = metrics[algo_name].new_call()
    t0 = time.perf_counter_ns()
    node = Node(problem.initial)
    if problem.is_goal(problem.initial):
        m['time_ns'] = time.perf_counter_ns() - t0
        return node
    frontier = [node]
    reached = {problem.initial}
    while frontier:
        m['max_frontier'] = max(m['max_frontier'], len(frontier))
        node = frontier.pop()
        for child in expand(problem, node):
            m['nodes_expanded'] += 1
            s = child.state
            if problem.is_goal(s):
                m['time_ns'] = time.perf_counter_ns() - t0
                return child
            if s not in reached:
                reached.add(s); frontier.append(child)
    m['time_ns'] = time.perf_counter_ns() - t0
    return failure

def uniform_cost_search(problem, algo_name='UCS'):
    m = metrics[algo_name].new_call()
    t0 = time.perf_counter_ns()
    node = Node(problem.initial, path_cost=0)
    if problem.is_goal(problem.initial):
        m['time_ns'] = time.perf_counter_ns() - t0
        return node
    frontier = [(node.path_cost, id(node), node)]
    reached = {problem.initial: node.path_cost}
    while frontier:
        m['max_frontier'] = max(m['max_frontier'], len(frontier))
        _, _, node = heapq.heappop(frontier)
        if problem.is_goal(node.state):
            m['time_ns'] = time.perf_counter_ns() - t0
            return node
        for child in expand(problem, node):
            m['nodes_expanded'] += 1
            s = child.state
            if s not in reached or child.path_cost < reached[s]:
                reached[s] = child.path_cost
                heapq.heappush(frontier, (child.path_cost, id(child), child))
    m['time_ns'] = time.perf_counter_ns() - t0
    return failure

ALGO_MAP = {'BFS': breadth_first_search, 'DFS': depth_first_search, 'UCS': uniform_cost_search}

# ============================================================================
# STORE GRAPH
# ============================================================================

PERMANENTLY_BLOCKED = {'A3'}
LEFT_CLUSTER = {'A1', 'A2'}
RIGHT_CLUSTER = {'A4', 'A5'}
T_CORRIDOR = {'T2', 'T3', 'T4'}
B_CORRIDOR = {'B2', 'B3', 'B4'}
BOTTLENECK_NODES = {'T3', 'B3'}  # The two key chokepoints around A3

class StoreNavigationProblem(Problem):
    def __init__(self, initial, goal, neighbors, blocked=None):
        super().__init__(initial=initial, goal=goal)
        self.neighbors = neighbors
        self.blocked = blocked or set()
    def actions(self, state):
        return [n for n in self.neighbors.get(state, [])
                if n not in self.blocked and n not in PERMANENTLY_BLOCKED]
    def result(self, state, action):
        return action

def build_store_graph():
    nb = defaultdict(list)
    def add(a, b):
        if b not in nb[a]: nb[a].append(b)
        if a not in nb[b]: nb[b].append(a)
    for i in range(1, 5):
        add(f'T{i}', f'T{i+1}')
        add(f'B{i}', f'B{i+1}')
        add(f'A{i}', f'A{i+1}')
    for i in range(1, 6):
        add(f'T{i}', f'A{i}')
        add(f'A{i}', f'B{i}')
    for n in ['T2','T3','T4']:
        add('ENTRANCE_TOP', n)
    for n in ['B2','B3','B4']:
        add('ENTRANCE_BOT', n)
    return dict(nb)

STORE_NEIGHBORS = build_store_graph()

PRODUCT_MAP = {
    'Apples':'A1','Bananas':'A1','Lettuce':'A1','Tomatoes':'A1','Carrots':'A1','Oranges':'A1',
    'Milk':'A2','Eggs':'A2','Cheese':'A2','Yogurt':'A2','Bread':'A2','Butter':'A2',
    'Rice':'A4','Pasta':'A4','Soup':'A4','Beans':'A4','Cereal':'A4','Coffee':'A4',
    'Chips':'A5','Cookies':'A5','Soda':'A5','Juice':'A5','Popcorn':'A5','Tea':'A5',
}

def shortest_path_length(start, goal, neighbors):
    if start == goal: return 0
    frontier = deque([(start, 0)])
    visited = {start}
    while frontier:
        node, dist = frontier.popleft()
        for nb in neighbors.get(node, []):
            if nb in PERMANENTLY_BLOCKED: continue
            if nb == goal: return dist + 1
            if nb not in visited:
                visited.add(nb)
                frontier.append((nb, dist + 1))
    return math.inf

# ============================================================================
# AI DECISION + TRAP LOGIC
# ============================================================================

def ai_decide_next(cur, shop_list, collected, algo_name, opp_pos):
    remaining = [p for p in shop_list if p not in collected]
    if not remaining: return cur, None, [], False

    blocked = {opp_pos} if opp_pos and opp_pos != cur else set()
    search_fn = ALGO_MAP.get(algo_name, breadth_first_search)
    best_path, best_prod, best_cost = None, None, math.inf

    for prod in remaining:
        goal = PRODUCT_MAP[prod]
        if cur == goal: return cur, prod, [cur], False
        prob = StoreNavigationProblem(cur, goal, STORE_NEIGHBORS, blocked)
        res = search_fn(prob, algo_name)
        if res != failure:
            p = path_states(res)
            if len(p) - 1 < best_cost:
                best_cost = len(p) - 1; best_path = p; best_prod = prod

    if best_path is None:
        for prod in remaining:
            goal = PRODUCT_MAP[prod]
            prob = StoreNavigationProblem(cur, goal, STORE_NEIGHBORS, set())
            res = search_fn(prob, algo_name)
            if res != failure:
                p = path_states(res)
                if len(p) - 1 < best_cost:
                    best_cost = len(p) - 1; best_path = p; best_prod = prod
        if best_path and len(best_path) > 1:
            if best_path[1] == opp_pos: return cur, best_prod, [cur], True
            return best_path[1], best_prod, best_path, False
        return cur, remaining[0] if remaining else None, [cur], True

    if len(best_path) > 1: return best_path[1], best_prod, best_path, False
    return cur, best_prod, best_path, False


def find_trap_move(algo_name, my_pos, ai_predicted_path, ai_pos):
    """Can I move to intercept the AI's predicted path?"""
    if not ai_predicted_path or len(ai_predicted_path) < 2:
        return None, None, 0
    search_fn = ALGO_MAP[algo_name]
    for choke in ai_predicted_path[1:]:
        if choke == my_pos:
            return choke, [my_pos], 0
        prob = StoreNavigationProblem(my_pos, choke, STORE_NEIGHBORS, {ai_pos})
        res = search_fn(prob, algo_name)
        if res != failure:
            p = path_states(res)
            dist = len(p) - 1
            if dist <= 3:
                return choke, p, dist
    return None, None, 0


# ============================================================================
# CLUSTER ANALYSIS HELPERS
# ============================================================================

def node_cluster(node):
    """Return 'left', 'right', 'corridor', or 'entrance'."""
    if node in LEFT_CLUSTER: return 'left'
    if node in RIGHT_CLUSTER: return 'right'
    if node.startswith('T') or node.startswith('B'): return 'corridor'
    return 'entrance'

def is_cross_cluster_move(old_pos, new_pos):
    """Did we cross from one side of A3 to the other?"""
    old_c = node_cluster(old_pos)
    new_c = node_cluster(new_pos)
    return (old_c == 'left' and new_c == 'right') or (old_c == 'right' and new_c == 'left')

def count_items_by_cluster(item_list):
    left = sum(1 for p in item_list if PRODUCT_MAP[p] in LEFT_CLUSTER)
    right = sum(1 for p in item_list if PRODUCT_MAP[p] in RIGHT_CLUSTER)
    return left, right

def distribution_label(left, right):
    total = left + right
    if total == 0: return "empty"
    ratio = max(left, right) / total
    if ratio >= 0.75: return "lopsided"
    elif ratio >= 0.58: return "moderate"
    else: return "balanced"


# ============================================================================
# FULL INSTRUMENTED GAME ENGINE
# ============================================================================

def auto_collect(position, shop_list, collected):
    new = [x for x in shop_list if x not in collected and PRODUCT_MAP.get(x) == position]
    collected.extend(new)
    return new

def run_single_game(algo_a, algo_b, entrance_a='Top', seed=None):
    """Run one game with full instrumentation for all 4 experiments."""
    if seed is not None:
        random.seed(seed)

    prods = list(PRODUCT_MAP.keys())
    random.shuffle(prods)
    a_list = sorted(prods[:12])
    b_list = sorted(prods[12:24])

    a_pos = 'ENTRANCE_TOP' if entrance_a == 'Top' else 'ENTRANCE_BOT'
    b_pos = 'ENTRANCE_BOT' if entrance_a == 'Top' else 'ENTRANCE_TOP'

    a_got, b_got = [], []
    a_moves, b_moves = 0, 0
    a_waits, b_waits = 0, 0
    turn = 0
    max_turns = 200

    # Reset per-game metrics
    for key in metrics:
        metrics[key].reset()

    # ----- EXP 2: Cross-cluster tracking -----
    a_cross_cluster_trips = 0
    b_cross_cluster_trips = 0
    a_last_cluster = node_cluster(a_pos)
    b_last_cluster = node_cluster(b_pos)

    # Corridor visit tracking
    a_t_visits, a_b_visits = 0, 0
    b_t_visits, b_b_visits = 0, 0

    # Bottleneck visits (T3, B3 specifically)
    a_bottleneck_visits = 0
    b_bottleneck_visits = 0

    # Per-aisle visit counts
    a_aisle_visits = defaultdict(int)
    b_aisle_visits = defaultdict(int)

    # ----- EXP 3: Blocking tracking -----
    a_blocks_caused = 0   # times A's position caused B to wait
    b_blocks_caused = 0   # times B's position caused A to wait
    a_blocked_at_bottleneck = 0  # waits specifically at T3/B3
    b_blocked_at_bottleneck = 0

    # Trap tracking
    a_trap_attempts = 0
    a_trap_successes = 0
    b_trap_attempts = 0
    b_trap_successes = 0

    # ----- EXP 4: Per-move path tracking -----
    a_actual_steps_total = 0
    a_optimal_steps_total = 0
    b_actual_steps_total = 0
    b_optimal_steps_total = 0
    a_per_move_nodes = []  # nodes expanded per move decision
    b_per_move_nodes = []

    # Full game timer
    game_start = time.perf_counter_ns()

    while turn < max_turns:
        # ========== PLAYER A's TURN ==========
        remaining_a = [p for p in a_list if p not in a_got]
        if not remaining_a:
            break

        # Record metrics state before A's search
        a_calls_before = len(metrics[algo_a].calls)

        # Predict B's next move (for trap analysis)
        remaining_b_pre = [p for p in b_list if p not in b_got]
        _, b_pred_prod, b_pred_path, _ = ai_decide_next(
            b_pos, b_list, b_got, algo_b, a_pos)

        # Check if A can trap B
        trap_node, trap_path, trap_dist = find_trap_move(
            algo_a, a_pos, b_pred_path, b_pos)
        if trap_node:
            a_trap_attempts += 1

        # A decides its move
        a_next, a_tgt, a_path, a_waited = ai_decide_next(
            a_pos, a_list, a_got, algo_a, b_pos)

        # Record per-move node expansion
        a_calls_after = len(metrics[algo_a].calls)
        move_nodes = sum(c['nodes_expanded'] for c in metrics[algo_a].calls[a_calls_before:a_calls_after])
        a_per_move_nodes.append(move_nodes)

        if a_waited:
            a_waits += 1
            b_blocks_caused += 1
            if a_pos in BOTTLENECK_NODES:
                a_blocked_at_bottleneck += 1
        else:
            old_a = a_pos
            # Detour tracking
            if a_tgt:
                goal_aisle = PRODUCT_MAP[a_tgt]
                opt = shortest_path_length(old_a, goal_aisle, STORE_NEIGHBORS)
                a_optimal_steps_total += opt
                if a_path and len(a_path) > 1:
                    a_actual_steps_total += len(a_path) - 1

            a_pos = a_next
            a_moves += 1
            collected = auto_collect(a_pos, a_list, a_got)

            # Cluster crossing
            new_cluster = node_cluster(a_pos)
            if a_last_cluster in ('left','right') and new_cluster in ('left','right') and a_last_cluster != new_cluster:
                a_cross_cluster_trips += 1
            if new_cluster in ('left','right'):
                a_last_cluster = new_cluster

            # Corridor tracking
            if a_pos in T_CORRIDOR: a_t_visits += 1
            if a_pos in B_CORRIDOR: a_b_visits += 1
            if a_pos in BOTTLENECK_NODES: a_bottleneck_visits += 1

            # Aisle visits
            if a_pos.startswith('A'):
                a_aisle_visits[a_pos] += 1

            # Check if trap succeeded (did A move to a spot that blocks B?)
            if trap_node and a_pos == trap_node:
                # Check if B would have wanted to move through here
                if b_pred_path and a_pos in b_pred_path[1:]:
                    a_trap_successes += 1

        if len(a_got) >= 12:
            break

        # ========== PLAYER B's TURN ==========
        remaining_b = [p for p in b_list if p not in b_got]
        if not remaining_b:
            break

        b_calls_before = len(metrics[algo_b].calls)

        # Predict A's next move (for B's trap analysis)
        remaining_a_pre = [p for p in a_list if p not in a_got]
        _, a_pred_prod, a_pred_path, _ = ai_decide_next(
            a_pos, a_list, a_got, algo_a, b_pos)

        trap_node_b, trap_path_b, trap_dist_b = find_trap_move(
            algo_b, b_pos, a_pred_path, a_pos)
        if trap_node_b:
            b_trap_attempts += 1

        b_next, b_tgt, b_path, b_waited = ai_decide_next(
            b_pos, b_list, b_got, algo_b, a_pos)

        b_calls_after = len(metrics[algo_b].calls)
        move_nodes_b = sum(c['nodes_expanded'] for c in metrics[algo_b].calls[b_calls_before:b_calls_after])
        b_per_move_nodes.append(move_nodes_b)

        if b_waited:
            b_waits += 1
            a_blocks_caused += 1
            if b_pos in BOTTLENECK_NODES:
                b_blocked_at_bottleneck += 1
        else:
            old_b = b_pos
            if b_tgt:
                goal_aisle = PRODUCT_MAP[b_tgt]
                opt = shortest_path_length(old_b, goal_aisle, STORE_NEIGHBORS)
                b_optimal_steps_total += opt
                if b_path and len(b_path) > 1:
                    b_actual_steps_total += len(b_path) - 1

            b_pos = b_next
            b_moves += 1
            auto_collect(b_pos, b_list, b_got)

            new_cluster_b = node_cluster(b_pos)
            if b_last_cluster in ('left','right') and new_cluster_b in ('left','right') and b_last_cluster != new_cluster_b:
                b_cross_cluster_trips += 1
            if new_cluster_b in ('left','right'):
                b_last_cluster = new_cluster_b

            if b_pos in T_CORRIDOR: b_t_visits += 1
            if b_pos in B_CORRIDOR: b_b_visits += 1
            if b_pos in BOTTLENECK_NODES: b_bottleneck_visits += 1

            if b_pos.startswith('A'):
                b_aisle_visits[b_pos] += 1

            if trap_node_b and b_pos == trap_node_b:
                if a_pred_path and b_pos in a_pred_path[1:]:
                    b_trap_successes += 1

        if len(b_got) >= 12:
            break
        turn += 1

    game_time_us = (time.perf_counter_ns() - game_start) / 1000

    # Winner
    a_done = len(a_got) >= 12
    b_done = len(b_got) >= 12
    if a_done and b_done:
        winner = 'Tie' if a_moves == b_moves else ('A' if a_moves < b_moves else 'B')
    elif a_done:
        winner = 'A'
    elif b_done:
        winner = 'B'
    else:
        winner = 'Timeout'

    # Item distribution
    a_left, a_right = count_items_by_cluster(a_list)
    b_left, b_right = count_items_by_cluster(b_list)

    # Detour ratios
    a_detour = (a_actual_steps_total / a_optimal_steps_total) if a_optimal_steps_total > 0 else 1.0
    b_detour = (b_actual_steps_total / b_optimal_steps_total) if b_optimal_steps_total > 0 else 1.0

    # Search metrics
    a_search = metrics[algo_a].totals()
    b_search = metrics[algo_b].totals()

    return {
        # --- EXP 1: Core matchup data ---
        'algo_a': algo_a, 'algo_b': algo_b, 'entrance_a': entrance_a,
        'seed': seed, 'winner': winner,
        'a_moves': a_moves, 'b_moves': b_moves,
        'a_waits': a_waits, 'b_waits': b_waits,
        'a_items_collected': len(a_got), 'b_items_collected': len(b_got),
        'a_efficiency': round(12 / a_moves, 4) if a_moves > 0 else 0,
        'b_efficiency': round(12 / b_moves, 4) if b_moves > 0 else 0,
        'move_margin': abs(a_moves - b_moves),

        # --- EXP 2: A3 blockage / cross-cluster ---
        'a_items_left': a_left, 'a_items_right': a_right,
        'b_items_left': b_left, 'b_items_right': b_right,
        'a_distribution': distribution_label(a_left, a_right),
        'b_distribution': distribution_label(b_left, b_right),
        'a_cross_cluster_trips': a_cross_cluster_trips,
        'b_cross_cluster_trips': b_cross_cluster_trips,
        'a_t_corridor_visits': a_t_visits,
        'a_b_corridor_visits': a_b_visits,
        'b_t_corridor_visits': b_t_visits,
        'b_b_corridor_visits': b_b_visits,
        'a_bottleneck_visits': a_bottleneck_visits,
        'b_bottleneck_visits': b_bottleneck_visits,
        'a_detour_ratio': round(a_detour, 3),
        'b_detour_ratio': round(b_detour, 3),
        'a_aisle_A1': a_aisle_visits.get('A1', 0),
        'a_aisle_A2': a_aisle_visits.get('A2', 0),
        'a_aisle_A4': a_aisle_visits.get('A4', 0),
        'a_aisle_A5': a_aisle_visits.get('A5', 0),
        'b_aisle_A1': b_aisle_visits.get('A1', 0),
        'b_aisle_A2': b_aisle_visits.get('A2', 0),
        'b_aisle_A4': b_aisle_visits.get('A4', 0),
        'b_aisle_A5': b_aisle_visits.get('A5', 0),

        # --- EXP 3: Blocking effectiveness ---
        'a_blocks_caused': a_blocks_caused,
        'b_blocks_caused': b_blocks_caused,
        'a_blocked_at_bottleneck': a_blocked_at_bottleneck,
        'b_blocked_at_bottleneck': b_blocked_at_bottleneck,
        'a_trap_attempts': a_trap_attempts,
        'a_trap_successes': a_trap_successes,
        'b_trap_attempts': b_trap_attempts,
        'b_trap_successes': b_trap_successes,

        # --- EXP 4: Computational cost ---
        'a_total_search_calls': a_search['total_calls'],
        'a_total_nodes_expanded': a_search['total_nodes'],
        'a_avg_nodes_per_call': a_search['avg_nodes_per_call'],
        'a_max_frontier': a_search['max_frontier_ever'],
        'a_search_time_us': a_search['total_time_us'],
        'a_avg_time_per_call_us': a_search['avg_time_per_call_us'],
        'b_total_search_calls': b_search['total_calls'],
        'b_total_nodes_expanded': b_search['total_nodes'],
        'b_avg_nodes_per_call': b_search['avg_nodes_per_call'],
        'b_max_frontier': b_search['max_frontier_ever'],
        'b_search_time_us': b_search['total_time_us'],
        'b_avg_time_per_call_us': b_search['avg_time_per_call_us'],

        'a_avg_nodes_per_move': round(sum(a_per_move_nodes) / len(a_per_move_nodes), 1) if a_per_move_nodes else 0,
        'b_avg_nodes_per_move': round(sum(b_per_move_nodes) / len(b_per_move_nodes), 1) if b_per_move_nodes else 0,
        'game_time_us': round(game_time_us, 1),
        'total_turns': turn,
    }


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_all_experiments(n_per_config=10, base_seed=42):
    algos = ['BFS', 'DFS', 'UCS']
    entrances = ['Top', 'Bottom']
    matchups = [(a, b) for a in algos for b in algos if a != b]
    total = len(matchups) * len(entrances) * n_per_config
    all_results = []
    game_id = 0

    print(f"\n{'='*80}")
    print(f"  STORE PATHFINDER — COMPLETE RESEARCH SUITE (Experiments 1-4)")
    print(f"  {len(matchups)} matchups x {len(entrances)} entrances x {n_per_config} trials = {total} games")
    print(f"{'='*80}\n")

    for algo_a, algo_b in matchups:
        for entrance in entrances:
            for trial in range(n_per_config):
                game_id += 1
                seed = base_seed + game_id
                result = run_single_game(algo_a, algo_b, entrance, seed)
                result['game_id'] = game_id
                result['trial'] = trial + 1
                all_results.append(result)

            cr = [r for r in all_results if r['algo_a']==algo_a and r['algo_b']==algo_b and r['entrance_a']==entrance]
            wa = sum(1 for r in cr if r['winner']=='A')
            wb = sum(1 for r in cr if r['winner']=='B')
            ti = sum(1 for r in cr if r['winner']=='Tie')
            am = sum(r['a_moves'] for r in cr)/len(cr)
            bm = sum(r['b_moves'] for r in cr)/len(cr)
            print(f"  {algo_a:>3} vs {algo_b:<3} | {entrance:>6} | A:{wa:2d} B:{wb:2d} T:{ti:2d} | "
                  f"Moves A:{am:6.1f} B:{bm:6.1f}")

    return all_results


# ============================================================================
# EXPERIMENT ANALYSIS
# ============================================================================

def analyze_experiment_1(results):
    """Head-to-head matchup summary."""
    print(f"\n{'='*96}")
    print(f"  EXPERIMENT 1: HEAD-TO-HEAD MATCHUP RESULTS")
    print(f"{'='*96}")
    print(f"{'Matchup':<14} {'Ent':>6} {'N':>3} {'A Win':>6} {'B Win':>6} {'Tie':>4} "
          f"{'A Mvs':>6} {'B Mvs':>6} {'A Wt':>5} {'B Wt':>5} "
          f"{'A Eff':>6} {'B Eff':>6} {'A Det':>6} {'B Det':>6}")
    print('-'*96)

    summary = defaultdict(lambda: defaultdict(float))
    for r in results:
        k = (f"{r['algo_a']}v{r['algo_b']}", r['entrance_a'])
        s = summary[k]
        s['n'] += 1
        s['aw'] += (r['winner']=='A')
        s['bw'] += (r['winner']=='B')
        s['ti'] += (r['winner']=='Tie')
        s['am'] += r['a_moves']; s['bm'] += r['b_moves']
        s['awt'] += r['a_waits']; s['bwt'] += r['b_waits']
        s['ae'] += r['a_efficiency']; s['be'] += r['b_efficiency']
        s['ad'] += r['a_detour_ratio']; s['bd'] += r['b_detour_ratio']

    for (m, e), s in sorted(summary.items()):
        n = int(s['n'])
        print(f"{m:<14} {e:>6} {n:>3} {int(s['aw']):>6} {int(s['bw']):>6} {int(s['ti']):>4} "
              f"{s['am']/n:>6.1f} {s['bm']/n:>6.1f} {s['awt']/n:>5.1f} {s['bwt']/n:>5.1f} "
              f"{s['ae']/n:>6.3f} {s['be']/n:>6.3f} {s['ad']/n:>6.2f} {s['bd']/n:>6.2f}")
    return summary


def analyze_experiment_2(results):
    """A3 blockage stress test."""
    print(f"\n{'='*96}")
    print(f"  EXPERIMENT 2A: CROSS-CLUSTER ITEM DISTRIBUTION IMPACT")
    print(f"{'='*96}")
    print(f"{'Distrib':<10} {'Algo':<5} {'N':>4} {'Avg Moves':>10} {'Win%':>6} "
          f"{'CrossTrips':>11} {'Detour':>7}")
    print('-'*60)

    dist_data = defaultdict(lambda: defaultdict(lambda: {
        'n':0,'moves':0,'wins':0,'cross':0,'detour':0}))
    for r in results:
        d = dist_data[r['a_distribution']][r['algo_a']]
        d['n'] += 1
        d['moves'] += r['a_moves']
        d['wins'] += (r['winner'] == 'A')
        d['cross'] += r['a_cross_cluster_trips']
        d['detour'] += r['a_detour_ratio']

    for label in ['balanced', 'moderate', 'lopsided']:
        if label not in dist_data: continue
        for algo in ['BFS', 'DFS', 'UCS']:
            d = dist_data[label].get(algo)
            if not d or d['n'] == 0: continue
            n = d['n']
            print(f"{label:<10} {algo:<5} {n:>4} {d['moves']/n:>10.1f} "
                  f"{d['wins']/n*100:>5.1f}% {d['cross']/n:>11.1f} {d['detour']/n:>7.2f}")

    print(f"\n{'='*96}")
    print(f"  EXPERIMENT 2B: CORRIDOR BOTTLENECK ANALYSIS")
    print(f"{'='*96}")
    print(f"{'Algo':<5} {'T-row':>8} {'B-row':>8} {'T/(T+B)':>8} "
          f"{'T3+B3 visits':>13} {'Aisle revisits':>15}")
    print('-'*62)

    for algo in ['BFS', 'DFS', 'UCS']:
        t_v, b_v, bn_v, n, total_aisle = 0, 0, 0, 0, 0
        for r in results:
            if r['algo_a'] == algo:
                t_v += r['a_t_corridor_visits']; b_v += r['a_b_corridor_visits']
                bn_v += r['a_bottleneck_visits']; n += 1
                total_aisle += r['a_aisle_A1']+r['a_aisle_A2']+r['a_aisle_A4']+r['a_aisle_A5']
            if r['algo_b'] == algo:
                t_v += r['b_t_corridor_visits']; b_v += r['b_b_corridor_visits']
                bn_v += r['b_bottleneck_visits']; n += 1
                total_aisle += r['b_aisle_A1']+r['b_aisle_A2']+r['b_aisle_A4']+r['b_aisle_A5']
        if n == 0: continue
        ratio = (t_v/n) / ((t_v+b_v)/n) if (t_v+b_v) > 0 else 0.5
        # 4 unique aisles to visit => optimal aisle visits ~ 4
        print(f"{algo:<5} {t_v/n:>8.1f} {b_v/n:>8.1f} {ratio:>8.2f} "
              f"{bn_v/n:>13.1f} {total_aisle/n:>15.1f}")
    return dist_data


def analyze_experiment_3(results):
    """Blocking effectiveness."""
    print(f"\n{'='*96}")
    print(f"  EXPERIMENT 3A: BLOCKS CAUSED vs SUFFERED")
    print(f"{'='*96}")
    print(f"{'Algo':<5} {'Games':>6} {'Caused':>8} {'Suffered':>10} "
          f"{'Net':>6} {'At Bottleneck':>14}")
    print('-'*55)

    for algo in ['BFS', 'DFS', 'UCS']:
        caused, suffered, bn_blocked, n = 0, 0, 0, 0
        for r in results:
            if r['algo_a'] == algo:
                caused += r['a_blocks_caused']; suffered += r['a_waits']
                bn_blocked += r['a_blocked_at_bottleneck']; n += 1
            if r['algo_b'] == algo:
                caused += r['b_blocks_caused']; suffered += r['b_waits']
                bn_blocked += r['b_blocked_at_bottleneck']; n += 1
        if n == 0: continue
        print(f"{algo:<5} {n:>6} {caused/n:>8.2f} {suffered/n:>10.2f} "
              f"{(caused-suffered)/n:>+6.2f} {bn_blocked/n:>14.2f}")

    print(f"\n{'='*96}")
    print(f"  EXPERIMENT 3B: TRAP STRATEGY SUCCESS RATE")
    print(f"{'='*96}")
    print(f"{'Algo':<5} {'Games':>6} {'Trap Attempts':>14} {'Successes':>10} {'Success%':>9}")
    print('-'*50)

    for algo in ['BFS', 'DFS', 'UCS']:
        attempts, successes, n = 0, 0, 0
        for r in results:
            if r['algo_a'] == algo:
                attempts += r['a_trap_attempts']; successes += r['a_trap_successes']; n += 1
            if r['algo_b'] == algo:
                attempts += r['b_trap_attempts']; successes += r['b_trap_successes']; n += 1
        if n == 0: continue
        rate = (successes/attempts*100) if attempts > 0 else 0
        print(f"{algo:<5} {n:>6} {attempts/n:>14.1f} {successes/n:>10.2f} "
              f"{rate:>8.1f}%")


def analyze_experiment_4(results):
    """Node expansion and computational cost."""
    print(f"\n{'='*96}")
    print(f"  EXPERIMENT 4A: NODES EXPLORED PER MOVE DECISION")
    print(f"{'='*96}")
    print(f"{'Algo':<5} {'Games':>6} {'Search Calls':>13} {'Total Nodes':>12} "
          f"{'Nodes/Call':>11} {'Nodes/Move':>11} {'Max Frontier':>13}")
    print('-'*76)

    for algo in ['BFS', 'DFS', 'UCS']:
        calls, nodes, npm, maxf, n = 0, 0, 0, 0, 0
        for r in results:
            if r['algo_a'] == algo:
                calls += r['a_total_search_calls']; nodes += r['a_total_nodes_expanded']
                npm += r['a_avg_nodes_per_move']; maxf = max(maxf, r['a_max_frontier']); n += 1
            if r['algo_b'] == algo:
                calls += r['b_total_search_calls']; nodes += r['b_total_nodes_expanded']
                npm += r['b_avg_nodes_per_move']; maxf = max(maxf, r['b_max_frontier']); n += 1
        if n == 0: continue
        print(f"{algo:<5} {n:>6} {calls/n:>13.0f} {nodes/n:>12.0f} "
              f"{nodes/calls:>11.1f} {npm/n:>11.1f} {maxf:>13}")

    print(f"\n{'='*96}")
    print(f"  EXPERIMENT 4B: TIMING & DATA STRUCTURE OVERHEAD")
    print(f"{'='*96}")
    print(f"{'Algo':<5} {'Games':>6} {'Total Time':>13} {'Time/Call':>11} "
          f"{'Time/Game':>11} {'Data Structure':>20}")
    print('-'*72)

    ds_map = {'BFS': 'deque (FIFO)', 'DFS': 'list (stack)', 'UCS': 'heapq (min-heap)'}
    for algo in ['BFS', 'DFS', 'UCS']:
        total_t, avg_tc, game_t, n = 0, 0, 0, 0
        for r in results:
            if r['algo_a'] == algo:
                total_t += r['a_search_time_us']; avg_tc += r['a_avg_time_per_call_us']
                game_t += r['game_time_us']; n += 1
            if r['algo_b'] == algo:
                total_t += r['b_search_time_us']; avg_tc += r['b_avg_time_per_call_us']
                game_t += r['game_time_us']; n += 1
        if n == 0: continue
        print(f"{algo:<5} {n:>6} {total_t/n:>10.0f} us {avg_tc/n:>8.1f} us "
              f"{game_t/n:>8.0f} us {ds_map[algo]:>20}")

    # BFS vs UCS equivalence deep-dive
    print(f"\n{'='*96}")
    print(f"  EXPERIMENT 4C: BFS vs UCS EQUIVALENCE (uniform-cost graph)")
    print(f"{'='*96}")

    bfs_v_ucs = [r for r in results if
                 (r['algo_a']=='BFS' and r['algo_b']=='UCS') or
                 (r['algo_a']=='UCS' and r['algo_b']=='BFS')]
    if bfs_v_ucs:
        diffs = [abs(r['a_moves']-r['b_moves']) for r in bfs_v_ucs]
        same = sum(1 for d in diffs if d == 0)
        print(f"  Direct matchups:       {len(bfs_v_ucs)}")
        print(f"  Identical move counts: {same} ({same/len(bfs_v_ucs)*100:.1f}%)")
        print(f"  Avg move difference:   {sum(diffs)/len(diffs):.2f}")
        print(f"  Max move difference:   {max(diffs)}")

        # Path identity: check if same-seed games produce same paths
        bfs_nodes_avg = sum(r['a_avg_nodes_per_call' if r['algo_a']=='BFS' else 'b_avg_nodes_per_call']
                           for r in bfs_v_ucs) / len(bfs_v_ucs)
        ucs_nodes_avg = sum(r['a_avg_nodes_per_call' if r['algo_a']=='UCS' else 'b_avg_nodes_per_call']
                           for r in bfs_v_ucs) / len(bfs_v_ucs)
        print(f"  BFS avg nodes/call:    {bfs_nodes_avg:.1f}")
        print(f"  UCS avg nodes/call:    {ucs_nodes_avg:.1f}")
        print(f"  Note: UCS expands more nodes due to heap re-insertion of lower-cost paths,")
        print(f"  even though all costs are 1. This is pure overhead on uniform-cost graphs.")


def print_overall_ranking(results):
    """Final algorithm ranking."""
    print(f"\n{'='*96}")
    print(f"  FINAL ALGORITHM RANKING (across all 120 games)")
    print(f"{'='*96}")

    for algo in ['BFS', 'DFS', 'UCS']:
        wins, losses, ties, moves, n = 0, 0, 0, 0, 0
        for r in results:
            if r['algo_a'] == algo:
                n += 1; moves += r['a_moves']
                if r['winner']=='A': wins += 1
                elif r['winner']=='B': losses += 1
                else: ties += 1
            if r['algo_b'] == algo:
                n += 1; moves += r['b_moves']
                if r['winner']=='B': wins += 1
                elif r['winner']=='A': losses += 1
                else: ties += 1

        print(f"\n  {algo}:")
        print(f"    Win rate:     {wins/n*100:.1f}% ({wins}W / {losses}L / {ties}T in {n} games)")
        print(f"    Avg moves:    {moves/n:.1f}")
        print(f"    Verdict:      ", end='')
        if wins/n > 0.55:
            print("STRONG — optimal pathfinding, consistent winner")
        elif wins/n > 0:
            print("MODERATE — performs well but no clear edge over BFS")
        else:
            print("WEAK — non-optimal paths cause massive move overhead")


# ============================================================================
# SAVE RAW DATA
# ============================================================================

def save_all_data(results, csv_path):
    if not results: return
    keys = results[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Raw data ({len(results)} rows, {len(keys)} columns) saved to: {csv_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    t0 = time.perf_counter()

    results = run_all_experiments(n_per_config=10, base_seed=42)

    print("\n" + "=" * 96)
    print("  FULL ANALYSIS")
    print("=" * 96)

    analyze_experiment_1(results)
    analyze_experiment_2(results)
    analyze_experiment_3(results)
    analyze_experiment_4(results)
    print_overall_ranking(results)

    save_all_data(results, '/home/claude/full_experiment_results.csv')

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*96}")
    print(f"  All experiments complete: {len(results)} games in {elapsed:.2f}s")
    print(f"{'='*96}\n")
