# ============================================================================
# STORE PATHFINDER - INTERACTIVE TURN-BASED GAME (v4.0)
# ============================================================================
# You (Player A) vs AI (Player B) using AIMA Search Algorithms
# 5 aisles, 30 products, 15 items each, banana-leaf blocked lanes
# ============================================================================

import streamlit as st
import random
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import heapq

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Store Pathfinder Race", page_icon="🛒", layout="wide")

# ============================================================================
# AIMA FRAMEWORK (Problem, Node, expand, path_states)
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
# SEARCH ALGORITHMS (AIMA)
# ============================================================================

def breadth_first_search(problem):
    node = Node(problem.initial)
    if problem.is_goal(problem.initial): return node
    frontier = deque([node])
    reached = {problem.initial}
    while frontier:
        node = frontier.popleft()
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s): return child
            if s not in reached:
                reached.add(s); frontier.append(child)
    return failure

def depth_first_search(problem):
    node = Node(problem.initial)
    if problem.is_goal(problem.initial): return node
    frontier = [node]
    reached = {problem.initial}
    while frontier:
        node = frontier.pop()
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s): return child
            if s not in reached:
                reached.add(s); frontier.append(child)
    return failure

def uniform_cost_search(problem):
    node = Node(problem.initial, path_cost=0)
    if problem.is_goal(problem.initial): return node
    frontier = [(node.path_cost, id(node), node)]
    reached = {problem.initial: node.path_cost}
    while frontier:
        _, _, node = heapq.heappop(frontier)
        if problem.is_goal(node.state): return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s]:
                reached[s] = child.path_cost
                heapq.heappush(frontier, (child.path_cost, id(child), child))
    return failure

ALGO_MAP = {'BFS': breadth_first_search, 'DFS': depth_first_search, 'UCS': uniform_cost_search}

# ============================================================================
# STORE NAVIGATION PROBLEM (blocks opponent node)
# ============================================================================

class StoreNavigationProblem(Problem):
    def __init__(self, initial, goal, neighbors, blocked=None):
        super().__init__(initial=initial, goal=goal)
        self.neighbors = neighbors
        self.blocked = blocked or set()
    def actions(self, state):
        return [n for n in self.neighbors.get(state, []) if n not in self.blocked]
    def result(self, state, action):
        return action

# ============================================================================
# STORE GRAPH
# ============================================================================
#     [ENTRANCE TOP]
#          |
#   T1--T2--T3--T4--T5       Top corridor
#   |   |   |   |   |
#   A1--A2--A3--A4--A5       Aisle mids (products)
#   |   |   |   |   |
#   B1--B2--B3--B4--B5       Bottom corridor
#          |
#     [ENTRANCE BOT]

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
AISLE_NODES = [f'A{i}' for i in range(1, 6)]

# ============================================================================
# PRODUCTS
# ============================================================================

PRODUCT_MAP = {
    'Apples':'A1','Bananas':'A1','Lettuce':'A1','Tomatoes':'A1','Carrots':'A1','Oranges':'A1',
    'Milk':'A2','Eggs':'A2','Cheese':'A2','Yogurt':'A2','Bread':'A2','Butter':'A2',
    'Chicken':'A3','Beef':'A3','Salmon':'A3','Shrimp':'A3','Pork':'A3','Turkey':'A3',
    'Rice':'A4','Pasta':'A4','Soup':'A4','Beans':'A4','Cereal':'A4','Coffee':'A4',
    'Chips':'A5','Cookies':'A5','Soda':'A5','Juice':'A5','Popcorn':'A5','Tea':'A5',
}

AISLE_CAT = {
    'A1':'🥬 Produce','A2':'🥛 Dairy','A3':'🥩 Meat','A4':'🥫 Pantry','A5':'🍿 Snacks',
}
AISLE_COL = {'A1':'#4CAF50','A2':'#2196F3','A3':'#F44336','A4':'#FF9800','A5':'#9C27B0'}
NODE_FRIENDLY = {
    'ENTRANCE_TOP':'🚪 Top Entrance','ENTRANCE_BOT':'🚪 Bottom Entrance',
    'T1':'T1 (top-left)','T2':'T2','T3':'T3 (top-center)','T4':'T4','T5':'T5 (top-right)',
    'A1':'A1 🥬 Produce','A2':'A2 🥛 Dairy','A3':'A3 🥩 Meat','A4':'A4 🥫 Pantry','A5':'A5 🍿 Snacks',
    'B1':'B1 (bot-left)','B2':'B2','B3':'B3 (bot-center)','B4':'B4','B5':'B5 (bot-right)',
}

def products_in_aisle(a):
    return [p for p, loc in PRODUCT_MAP.items() if loc == a]

# ============================================================================
# SVG MAP with banana-leaf blocked lanes
# ============================================================================

def render_svg(a_pos, b_pos, a_collected, b_collected, a_list, b_list):
    W, H = 740, 580
    cx = {1:110, 2:235, 3:360, 4:485, 5:610}
    ent_top_y, t_y, a_y, b_y, ent_bot_y = 38, 115, 270, 425, 500

    pos = {'ENTRANCE_TOP':(cx[3],ent_top_y), 'ENTRANCE_BOT':(cx[3],ent_bot_y)}
    for i in range(1,6):
        pos[f'T{i}'] = (cx[i], t_y)
        pos[f'A{i}'] = (cx[i], a_y)
        pos[f'B{i}'] = (cx[i], b_y)

    s = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" '
         f'style="background:#0d1117;border-radius:16px;border:1px solid #30363d;">']

    # Title
    s.append(f'<text x="{W//2}" y="22" text-anchor="middle" fill="#7d8590" '
             f'font-size="12" font-family="sans-serif">MLC Busy Local Grocery Store</text>')

    # ---- Aisle vertical bars ----
    for i in range(1,6):
        x = cx[i]; col = AISLE_COL[f'A{i}']
        s.append(f'<rect x="{x-40}" y="{t_y-20}" width="80" height="{b_y-t_y+40}" '
                 f'rx="12" fill="{col}" opacity="0.08" stroke="{col}" stroke-width="1"/>')
        # Category
        s.append(f'<text x="{x}" y="{a_y+52}" text-anchor="middle" fill="{col}" '
                 f'font-size="10" font-weight="600" font-family="sans-serif">{AISLE_CAT[f"A{i}"]}</text>')
        # Product list below aisle
        for pi, pr in enumerate(products_in_aisle(f'A{i}')):
            yy = a_y + 64 + pi*12
            got_a = pr in a_collected; got_b = pr in b_collected
            need_a = pr in a_list and not got_a; need_b = pr in b_list and not got_b
            fl = '#484f58'
            if got_a: fl='#FFD600'
            elif got_b: fl='#E040FB'
            elif need_a: fl='#e3b341'
            elif need_b: fl='#bc8cff'
            mark = '✓ ' if (got_a or got_b) else ''
            s.append(f'<text x="{x}" y="{yy}" text-anchor="middle" fill="{fl}" '
                     f'font-size="8" font-family="sans-serif">{mark}{pr}</text>')

    # ---- Edges ----
    drawn = set()
    for nd, nbrs in STORE_NEIGHBORS.items():
        if nd not in pos: continue
        x1,y1 = pos[nd]
        for nb in nbrs:
            if nb not in pos: continue
            edge = tuple(sorted([nd,nb]))
            if edge in drawn: continue
            drawn.add(edge)
            x2,y2 = pos[nb]

            # Check if this edge leads to/from a blocked node (b_pos)
            is_blocked_edge = (nd == b_pos or nb == b_pos) and nd != a_pos and nb != a_pos
            stroke_col = '#2d333b'
            dash = '5,4'
            sw = 2
            if is_blocked_edge:
                stroke_col = '#2d6a2e'
                sw = 6
                dash = '0'
            s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                     f'stroke="{stroke_col}" stroke-width="{sw}" stroke-dasharray="{dash}" '
                     f'stroke-linecap="round"/>')
            # Banana leaves on blocked edges
            if is_blocked_edge:
                mx, my = (x1+x2)//2, (y1+y2)//2
                for ox, oy in [(-8,-6),(6,4),(0,-12),(0,10),(-5,8),(7,-8)]:
                    lx, ly = mx+ox, my+oy
                    s.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" font-size="11">🍃</text>')

    # ---- Entrance labels ----
    for ey, label in [(ent_top_y,'🚪 ENTRANCE TOP'), (ent_bot_y,'🚪 ENTRANCE BOT')]:
        ex = cx[3]
        s.append(f'<rect x="{ex-68}" y="{ey-13}" width="136" height="26" '
                 f'rx="8" fill="#238636" opacity="0.2"/>')
        s.append(f'<text x="{ex}" y="{ey+4}" text-anchor="middle" fill="#3fb950" '
                 f'font-size="11" font-weight="bold" font-family="sans-serif">{label}</text>')

    # ---- Nodes ----
    for nd, (x,y) in pos.items():
        if nd.startswith('ENTRANCE'): continue
        is_aisle = nd.startswith('A')
        r = 22 if is_aisle else 17
        fill = AISLE_COL.get(nd,'#161b22') if is_aisle else '#161b22'
        stroke = '#fff' if is_aisle else '#30363d'

        # Banana leaf overlay if this is AI's position
        if nd == b_pos:
            # Big banana leaf cluster
            s.append(f'<circle cx="{x}" cy="{y}" r="{r+10}" fill="#1a4d1a" opacity="0.5"/>')
            for ox, oy in [(-14,-14),(14,-10),(-10,14),(12,12),(0,-18),(0,16),(-16,2),(16,0)]:
                s.append(f'<text x="{x+ox}" y="{y+oy}" text-anchor="middle" font-size="14">🍌</text>')
            s.append(f'<text x="{x}" y="{y-r-8}" text-anchor="middle" fill="#f85149" '
                     f'font-size="9" font-weight="bold" font-family="sans-serif">🚫 BLOCKED</text>')

        s.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="2"/>')
        s.append(f'<text x="{x}" y="{y+4}" text-anchor="middle" fill="white" '
                 f'font-size="10" font-weight="bold" font-family="sans-serif">{nd}</text>')

    # ---- Players ----
    def draw_p(p, color, lbl, ox):
        if p in pos:
            px, py = pos[p]; px += ox
            s.append(f'<circle cx="{px}" cy="{py}" r="20" fill="none" '
                     f'stroke="{color}" stroke-width="2" opacity="0.4">'
                     f'<animate attributeName="r" values="18;22;18" dur="1.5s" repeatCount="indefinite"/>'
                     f'</circle>')
            s.append(f'<circle cx="{px}" cy="{py}" r="14" fill="{color}" stroke="#fff" stroke-width="2.5"/>')
            s.append(f'<text x="{px}" y="{py+5}" text-anchor="middle" fill="#000" '
                     f'font-size="12" font-weight="bold" font-family="sans-serif">{lbl}</text>')

    if a_pos == b_pos:
        draw_p(a_pos,'#FFD600','A',-18); draw_p(b_pos,'#E040FB','B',18)
    else:
        draw_p(a_pos,'#FFD600','A',0); draw_p(b_pos,'#E040FB','B',0)

    # Legend
    ly = H - 18
    s.append(f'<text x="15" y="{ly}" fill="#7d8590" font-size="10" font-family="sans-serif">'
             f'🟡 You (A)  |  🟣 AI (B)  |  🍌🍃 Banana leaf = blocked by AI  |  '
             f'Yellow text = your items  |  Purple text = AI items</text>')

    s.append('</svg>')
    return '\n'.join(s)

# ============================================================================
# AI LOGIC
# ============================================================================

def ai_decide_next(cur, shop_list, collected, algo_name, opp_pos):
    """AI picks next target and uses AIMA search to navigate (avoids opponent)."""
    remaining = [p for p in shop_list if p not in collected]
    if not remaining: return cur, None, [], False

    blocked = {opp_pos} if opp_pos and opp_pos != cur else set()
    search_fn = ALGO_MAP.get(algo_name, breadth_first_search)
    best_path, best_prod, best_cost = None, None, math.inf

    for prod in remaining:
        goal = PRODUCT_MAP[prod]
        if cur == goal: return cur, prod, [cur], False
        prob = StoreNavigationProblem(cur, goal, STORE_NEIGHBORS, blocked)
        res = search_fn(prob)
        if res != failure:
            p = path_states(res)
            if len(p)-1 < best_cost:
                best_cost = len(p)-1; best_path = p; best_prod = prod

    if best_path is None:
        for prod in remaining:
            goal = PRODUCT_MAP[prod]
            prob = StoreNavigationProblem(cur, goal, STORE_NEIGHBORS, set())
            res = search_fn(prob)
            if res != failure:
                p = path_states(res)
                if len(p)-1 < best_cost:
                    best_cost = len(p)-1; best_path = p; best_prod = prod
        if best_path and len(best_path)>1:
            if best_path[1] == opp_pos: return cur, best_prod, [cur], True
            return best_path[1], best_prod, best_path, False
        return cur, remaining[0] if remaining else None, [cur], True

    if len(best_path)>1: return best_path[1], best_prod, best_path, False
    return cur, best_prod, best_path, False

# ============================================================================
# STRATEGY ADVISOR: your two helper algorithms recommend moves
# ============================================================================

ALGO_DESC = {
    'BFS': 'Breadth-First Search (explores level-by-level, guaranteed shortest path)',
    'DFS': 'Depth-First Search (dives deep first, may find creative detours)',
    'UCS': 'Uniform Cost Search (expands lowest-cost node, optimal path)',
}
ALGO_SHORT = {'BFS': 'Shortest Path', 'DFS': 'Deep Dive', 'UCS': 'Optimal Cost'}
ALGO_ICON = {'BFS': '🔵', 'DFS': '🟢', 'UCS': '🟠'}

def predict_ai_move(gs):
    """Predict what AI will do next turn using its own algorithm."""
    ai_remaining = [p for p in gs['b_list'] if p not in gs['b_got']]
    if not ai_remaining:
        return None, None, []
    _, prod, path, _ = ai_decide_next(
        gs['b_pos'], gs['b_list'], gs['b_got'], gs['algo'], gs['a_pos'])
    return prod, PRODUCT_MAP.get(prod) if prod else None, path or []

def algo_recommend(algo_name, my_pos, my_remaining, ai_pos):
    """Use a specific AIMA algorithm to recommend the best next product + path for you."""
    search_fn = ALGO_MAP[algo_name]
    best_path, best_prod, best_cost = None, None, math.inf
    blocked = {ai_pos} if ai_pos else set()

    for prod in my_remaining:
        goal = PRODUCT_MAP[prod]
        if my_pos == goal:
            return prod, goal, [my_pos], 0, "You're already here! Collect it."
        prob = StoreNavigationProblem(my_pos, goal, STORE_NEIGHBORS, blocked)
        res = search_fn(prob)
        if res != failure:
            p = path_states(res)
            cost = len(p) - 1
            if cost < best_cost:
                best_cost = cost; best_path = p; best_prod = prod

    if best_prod:
        goal_aisle = PRODUCT_MAP[best_prod]
        return best_prod, goal_aisle, best_path, best_cost, None
    return None, None, [], 0, "No reachable products found."

def find_trap_move(algo_name, my_pos, ai_predicted_path, ai_pos):
    """Use an algorithm to find if you can intercept/block the AI's predicted path."""
    if not ai_predicted_path or len(ai_predicted_path) < 2:
        return None, None, 0

    search_fn = ALGO_MAP[algo_name]
    # Try to reach each node on AI's path (prioritize earlier nodes = bigger block)
    for choke in ai_predicted_path[1:]:
        if choke == my_pos:
            return choke, [my_pos], 0
        prob = StoreNavigationProblem(my_pos, choke, STORE_NEIGHBORS, {ai_pos})
        res = search_fn(prob)
        if res != failure:
            p = path_states(res)
            dist = len(p) - 1
            if dist <= 3:
                return choke, p, dist
    return None, None, 0

def get_full_recommendations(gs):
    """Build complete recommendations from the two algorithms Player A has access to."""
    ai_algo = gs['algo']
    my_algos = [a for a in ['BFS', 'DFS', 'UCS'] if a != ai_algo]
    my_pos = gs['a_pos']
    ai_pos = gs['b_pos']
    my_remaining = [p for p in gs['a_list'] if p not in gs['a_got']]

    # Predict AI
    ai_prod, ai_aisle, ai_path = predict_ai_move(gs)

    # Build recommendation for each of your two algos
    recs = {}
    for algo in my_algos:
        rec = {'algo': algo}

        # Best product to go for
        prod, aisle, path, cost, note = algo_recommend(algo, my_pos, my_remaining, ai_pos)
        rec['product'] = prod
        rec['aisle'] = aisle
        rec['path'] = path or []
        rec['cost'] = cost
        rec['note'] = note

        # Can we trap the AI?
        trap_node, trap_path, trap_dist = find_trap_move(algo, my_pos, ai_path, ai_pos)
        rec['trap_node'] = trap_node
        rec['trap_path'] = trap_path or []
        rec['trap_dist'] = trap_dist

        # Count how many items at the recommended aisle
        if aisle:
            rec['items_at_aisle'] = [p for p in my_remaining if PRODUCT_MAP.get(p) == aisle]
        else:
            rec['items_at_aisle'] = []

        recs[algo] = rec

    # Shared aisles
    ai_remaining = [p for p in gs['b_list'] if p not in gs['b_got']]
    my_needed = set(PRODUCT_MAP[p] for p in my_remaining)
    ai_needed = set(PRODUCT_MAP[p] for p in ai_remaining)
    contested = sorted(my_needed & ai_needed)

    return {
        'my_algos': my_algos,
        'ai_prod': ai_prod,
        'ai_aisle': ai_aisle,
        'ai_path': ai_path,
        'recs': recs,
        'contested': contested,
        'ai_remaining': ai_remaining,
    }

# ============================================================================
# SESSION STATE
# ============================================================================

def init_game(entrance, algo_b, player_a_mode='manual'):
    prods = list(PRODUCT_MAP.keys()); random.shuffle(prods)
    st.session_state.gs = {
        'active': True, 'turn': 0,
        'a_pos': 'ENTRANCE_TOP' if entrance=='Top' else 'ENTRANCE_BOT',
        'b_pos': 'ENTRANCE_BOT' if entrance=='Top' else 'ENTRANCE_TOP',
        'a_list': sorted(prods[:15]), 'b_list': sorted(prods[15:30]),
        'a_got': [], 'b_got': [],
        'a_moves': 0, 'b_moves': 0, 'a_waits': 0, 'b_waits': 0,
        'algo': algo_b, 'entrance': entrance,
        'a_mode': player_a_mode,  # 'manual', 'BFS', 'DFS', 'UCS', or 'auto_best'
        'log': [], 'game_over': False, 'winner': None,
        'last_ai_action': None, 'ai_target': None, 'ai_path': [],
    }

def auto_collect(p, slist, got):
    new = [x for x in slist if x not in got and PRODUCT_MAP.get(x)==p]
    got.extend(new); return new

def compute_player_a_auto_move(gs):
    """Compute the best move for Player A based on their selected algo mode."""
    mode = gs.get('a_mode', 'manual')
    if mode == 'manual':
        return None  # manual, no auto-move

    my_pos = gs['a_pos']
    ai_pos = gs['b_pos']
    my_remaining = [p for p in gs['a_list'] if p not in gs['a_got']]
    if not my_remaining:
        return None

    if mode == 'auto_best':
        # Try both available algos and pick the one with shorter path
        other_algos = [a for a in ['BFS','DFS','UCS'] if a != gs['algo']]
        best_move, best_cost = None, math.inf
        best_algo_used = None
        for algo_name in other_algos:
            prod, aisle, path, cost, _ = algo_recommend(algo_name, my_pos, my_remaining, ai_pos)
            if prod and cost < best_cost:
                best_cost = cost
                best_move = path[1] if path and len(path) > 1 else my_pos
                best_algo_used = algo_name
        if best_move and best_move != my_pos:
            return best_move, best_algo_used
        # If at the aisle already, stay (collect items)
        if any(PRODUCT_MAP.get(p) == my_pos for p in my_remaining):
            return my_pos, 'auto_best'
        return None
    else:
        # Use the specific algo selected
        algo_name = mode
        prod, aisle, path, cost, _ = algo_recommend(algo_name, my_pos, my_remaining, ai_pos)
        if prod and path and len(path) > 1:
            return path[1], algo_name
        if prod and my_pos == aisle:
            return my_pos, algo_name
        return None

def process_move(gs, choice):
    """Process player A's choice and then AI's turn."""
    cur = gs['a_pos']
    if choice == '__WAIT__':
        gs['a_waits'] += 1
        gs['log'].append(f"Turn {gs['turn']+1}: 🟡 You waited at {cur}")
    else:
        gs['a_pos'] = choice
        gs['a_moves'] += 1
        new_a = auto_collect(choice, gs['a_list'], gs['a_got'])
        msg = f"Turn {gs['turn']+1}: 🟡 You moved to {choice}"
        if new_a: msg += f"  ✅ Picked up: {', '.join(new_a)}"
        gs['log'].append(msg)

    # AI turn
    ai_next, ai_tgt, ai_full_path, ai_waited = ai_decide_next(
        gs['b_pos'], gs['b_list'], gs['b_got'], gs['algo'], gs['a_pos'])
    if ai_waited:
        gs['b_waits'] += 1
        gs['log'].append(f"Turn {gs['turn']+1}: 🟣 AI waited (path blocked by you!)")
        gs['last_ai_action'] = "Waited (blocked)"
        gs['ai_target'] = ai_tgt
        gs['ai_path'] = []
    else:
        gs['b_pos'] = ai_next; gs['b_moves'] += 1
        new_b = auto_collect(ai_next, gs['b_list'], gs['b_got'])
        msg = f"Turn {gs['turn']+1}: 🟣 AI moved to {ai_next}"
        if ai_tgt: msg += f" (targeting {ai_tgt})"
        if new_b: msg += f"  ✅ Picked up: {', '.join(new_b)}"
        gs['log'].append(msg)
        gs['last_ai_action'] = f"Moved to {ai_next}" + (f" (got {', '.join(new_b)})" if new_b else "")
        gs['ai_target'] = ai_tgt
        gs['ai_path'] = ai_full_path if ai_full_path else []

    gs['turn'] += 1
    # Check win
    if len(gs['a_got'])>=15 or len(gs['b_got'])>=15:
        gs['game_over'] = True
        a_d, b_d = len(gs['a_got'])>=15, len(gs['b_got'])>=15
        if a_d and b_d:
            gs['winner'] = 'TIE' if gs['a_moves']==gs['b_moves'] else ('You' if gs['a_moves']<gs['b_moves'] else 'AI')
        elif a_d: gs['winner'] = 'You'
        else: gs['winner'] = 'AI'


# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    section[data-testid="stSidebar"] { background: #0d1117; }
    .move-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 12px 16px; margin: 4px 0; color: #c9d1d9;
    }
    .move-card.blocked {
        background: #1c2e1c; border-color: #2ea04370;
        opacity: 0.6;
    }
    .move-card.has-items {
        border-color: #e3b341; border-width: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE
# ============================================================================
st.title("🛒 Store Pathfinder: Shopping Race")
st.caption("You (Player A) vs AI (Player B) | Manual or Algorithm-Assisted | MLC Busy Local Grocery Store")

# ============================================================================
# SETUP SCREEN
# ============================================================================
if 'gs' not in st.session_state or not st.session_state.gs.get('active'):
    st.markdown("---")
    st.header("⚙️ New Game Setup")

    st.markdown("#### 🤖 Player B (AI Opponent)")
    c1, c2 = st.columns(2)
    with c1:
        entrance = st.radio("🚪 **Your entrance:**", ["Top", "Bottom"], horizontal=True)
    with c2:
        algo_b = st.radio("**AI (Player B) algorithm:**", ["BFS", "DFS", "UCS"], horizontal=True,
                          captions=["Shortest path", "Dives deep first", "Lowest cost first"])

    st.markdown("---")
    st.markdown("#### 🧑 Player A (You)")
    other_algos = [a for a in ['BFS','DFS','UCS'] if a != algo_b]

    player_a_mode = st.radio(
        "**How do you want to play?**",
        options=["manual", other_algos[0], other_algos[1], "auto_best"],
        format_func=lambda v: {
            "manual": "🎮 Manual -- I choose every move myself (advisors show recommendations)",
            other_algos[0]: f"🤖 Use {other_algos[0]} -- Auto-play assisted by {ALGO_DESC.get(other_algos[0],other_algos[0])}",
            other_algos[1]: f"🤖 Use {other_algos[1]} -- Auto-play assisted by {ALGO_DESC.get(other_algos[1],other_algos[1])}",
            "auto_best": f"⚡ Auto Best -- Pick the smarter of {other_algos[0]} & {other_algos[1]} each turn",
        }[v],
        key="player_a_mode_select"
    )

    st.markdown("")
    if st.button("🎮  START GAME", type="primary", use_container_width=True):
        init_game(entrance, algo_b, player_a_mode)
        st.rerun()

    st.markdown("---")
    st.subheader("🏪 Store Preview")
    st.markdown(render_svg('ENTRANCE_TOP','ENTRANCE_BOT',[],[],
                           list(PRODUCT_MAP.keys())[:15],list(PRODUCT_MAP.keys())[15:]),
                unsafe_allow_html=True)

    st.markdown("---")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("""
        ### 📋 How to Play
        1. Pick Player B's algorithm and **your play mode**:
           - **Manual:** You choose every move, advisors show recommendations
           - **Use [Algo]:** That algorithm auto-picks your moves (you can override)
           - **Auto Best:** Smartest of your two algos picks each turn
        2. 🍌 **Banana leaves** block lanes where the AI stands
        3. Products auto-collect when you reach the right aisle
        4. Each player gets **15 random items** -- first to collect all wins!
        """)
    with r2:
        st.markdown("### 🗺️ Aisle Directory")
        for a in AISLE_NODES:
            st.markdown(f"**{AISLE_CAT[a]}** (`{a}`): {', '.join(products_in_aisle(a))}")
    st.stop()


# ============================================================================
# GAME SCREEN
# ============================================================================
gs = st.session_state.gs

# ---- MAP (full width, no overlap) ----
st.markdown(render_svg(gs['a_pos'], gs['b_pos'], gs['a_got'], gs['b_got'],
                       gs['a_list'], gs['b_list']), unsafe_allow_html=True)

st.markdown("---")

# ---- SCOREBOARD ROW ----
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("🟡 Your Items", f"{len(gs['a_got'])}/15")
with s2:
    st.metric("🟡 Your Moves", gs['a_moves'], delta=f"{gs['a_waits']} waits")
with s3:
    st.metric(f"🟣 AI Items ({gs['algo']})", f"{len(gs['b_got'])}/15")
with s4:
    st.metric("🟣 AI Moves", gs['b_moves'], delta=f"{gs['b_waits']} waits")

# Show last AI action
if gs['last_ai_action']:
    st.info(f"🤖 **AI's last action:** {gs['last_ai_action']}")

st.markdown("---")

# ---- AI INTEL + YOUR ALGO ADVISORS (always visible during gameplay) ----
if not gs['game_over']:
    my_remaining_items = [p for p in gs['a_list'] if p not in gs['a_got']]

    if my_remaining_items:
        advice = get_full_recommendations(gs)
        algo1, algo2 = advice['my_algos']

        with st.expander("🕵️ **AI SCOUTING + YOUR ALGORITHM ADVISORS** -- click to expand", expanded=True):

            # ---- ROW 1: AI Intel (what is the enemy doing?) ----
            st.markdown("### 🔴 AI Scouting Report")
            ai_c1, ai_c2, ai_c3 = st.columns(3)
            with ai_c1:
                st.markdown(f"**AI uses:** `{gs['algo']}` ({ALGO_DESC.get(gs['algo'],'')})")
                st.markdown(f"**AI position:** `{gs['b_pos']}`")
                if advice['ai_prod']:
                    st.markdown(f"**Predicted target:** `{advice['ai_aisle']}` "
                                f"{AISLE_CAT.get(advice['ai_aisle'],'')} for **{advice['ai_prod']}**")
            with ai_c2:
                if advice['ai_path'] and len(advice['ai_path']) > 1:
                    st.markdown(f"**Predicted path:**")
                    st.code(" → ".join(advice['ai_path']), language=None)
                ai_rem = advice['ai_remaining']
                st.markdown(f"**AI needs {len(ai_rem)} items** from: "
                            f"`{'`, `'.join(sorted(set(PRODUCT_MAP[p] for p in ai_rem)))}`")
            with ai_c3:
                st.markdown("**AI has collected:**")
                if gs['b_got']:
                    for p in gs['b_got']:
                        st.markdown(f"&nbsp;&nbsp;✅ ~~{p}~~")
                else:
                    st.caption("Nothing yet.")
                if advice['contested']:
                    st.markdown(f"⚔️ **Contested:** `{'`, `'.join(advice['contested'])}`")

            st.markdown("---")

            # ---- ROW 2: Your two algorithm advisors side by side ----
            st.markdown(f"### 🧠 Your Advisors:  {ALGO_ICON[algo1]} {algo1}  vs  {ALGO_ICON[algo2]} {algo2}")
            col_a1, col_divider, col_a2 = st.columns([5, 1, 5])

            for col, algo_name in [(col_a1, algo1), (col_a2, algo2)]:
                rec = advice['recs'][algo_name]
                with col:
                    st.markdown(f"#### {ALGO_ICON[algo_name]} {algo_name} Advisor")
                    st.caption(ALGO_SHORT[algo_name])

                    # Shopping recommendation
                    if rec['product']:
                        items_count = len(rec['items_at_aisle'])
                        st.markdown(f"🛒 **Go to `{rec['aisle']}`** "
                                    f"({AISLE_CAT.get(rec['aisle'],'')}) "
                                    f"for **{rec['product']}**")
                        if items_count > 1:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;📦 *{items_count} of your items here:* "
                                        f"{', '.join(rec['items_at_aisle'])}")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;📏 **{rec['cost']} move{'s' if rec['cost']!=1 else ''}** away")
                        if rec['path'] and len(rec['path']) > 1:
                            st.code(" → ".join(rec['path']), language=None)
                    elif rec['note']:
                        st.info(rec['note'])

                    # Trap recommendation
                    if rec['trap_node']:
                        st.markdown(f"🪤 **TRAP OPTION:** Move to **`{rec['trap_node']}`** "
                                    f"({rec['trap_dist']} move{'s' if rec['trap_dist']!=1 else ''}) "
                                    f"to block AI's path!")
                        if rec['trap_path'] and len(rec['trap_path']) > 1:
                            st.code("Trap route: " + " → ".join(rec['trap_path']), language=None)
                    else:
                        st.caption("No trap opportunity this turn.")

            with col_divider:
                st.markdown("<div style='text-align:center;color:#484f58;font-size:24px;"
                            "margin-top:80px;'>vs</div>", unsafe_allow_html=True)

            # AI weakness reminder
            st.markdown("---")
            if gs['algo'] == 'BFS':
                st.markdown("💡 **AI weakness (BFS):** Always picks shortest path. "
                            "Block corridor nodes (T2-T4, B2-B4) to force long detours.")
            elif gs['algo'] == 'DFS':
                st.markdown("💡 **AI weakness (DFS):** Dives deep along one path. "
                            "Block bottom-row nodes (B1-B5) to trap it in dead ends.")
            elif gs['algo'] == 'UCS':
                st.markdown("💡 **AI weakness (UCS):** Optimizes cost at junctions. "
                            "Block merge nodes (A2, A3, A4) where paths converge.")

st.markdown("---")

# ---- GAME OVER ----
if gs['game_over']:
    if gs['winner'] == 'You':
        st.balloons()
        st.success(f"## 🎉 YOU WIN!  \nCollected 15 items in **{gs['a_moves']} moves** "
                   f"({gs['a_waits']} waits)")
    elif gs['winner'] == 'AI':
        st.error(f"## 🤖 AI Wins!  \nCollected 15 items in **{gs['b_moves']} moves** "
                 f"using {gs['algo']}")
    else:
        st.info("## 🤝 It's a TIE!")

    if st.button("🔄 Play Again", type="primary", use_container_width=True):
        del st.session_state.gs; st.rerun()

else:
    # ---- MOVE SELECTION PANEL ----
    a_mode = gs.get('a_mode', 'manual')
    mode_labels = {
        'manual': '🎮 Manual Mode',
        'auto_best': f"⚡ Auto Best (smartest of your 2 algos)",
    }
    if a_mode not in mode_labels:
        mode_labels[a_mode] = f"🤖 Auto-Play ({a_mode})"
    mode_label = mode_labels.get(a_mode, a_mode)

    st.subheader(f"🎮 Turn {gs['turn']+1} -- {mode_label}")
    st.markdown(f"**You are at:** `{gs['a_pos']}` ({NODE_FRIENDLY.get(gs['a_pos'], gs['a_pos'])})")

    current = gs['a_pos']
    neighbors = STORE_NEIGHBORS.get(current, [])
    blocked_node = gs['b_pos']
    remaining_a = [p for p in gs['a_list'] if p not in gs['a_got']]
    needed_aisles = set(PRODUCT_MAP[p] for p in remaining_a)

    open_options = []
    blocked_options = []
    for n in neighbors:
        if n == blocked_node:
            blocked_options.append(n)
        else:
            items_here = [p for p in remaining_a if PRODUCT_MAP.get(p) == n]
            desc = NODE_FRIENDLY.get(n, n)
            if items_here:
                desc += f"  --  🛒 {len(items_here)} item(s): {', '.join(items_here)}"
            open_options.append((n, desc))

    # Show blocked lanes
    if blocked_options:
        for bn in blocked_options:
            st.markdown(
                f'<div class="move-card blocked">'
                f'🍌🍃 <b>{bn}</b> ({NODE_FRIENDLY.get(bn, bn)}) '
                f'-- <span style="color:#f85149;">BLOCKED by AI (banana leaf barrier!)</span>'
                f'</div>', unsafe_allow_html=True)

    # ========== AUTO-PLAY MODES ==========
    if a_mode != 'manual':
        auto_result = compute_player_a_auto_move(gs)

        if auto_result:
            auto_move, algo_used = auto_result
            # Show what the algo decided
            items_at = [p for p in remaining_a if PRODUCT_MAP.get(p) == auto_move]
            st.markdown(f"**{ALGO_ICON.get(algo_used, '🤖')} {algo_used} recommends:** "
                        f"Move to **`{auto_move}`** ({NODE_FRIENDLY.get(auto_move, auto_move)})")
            if items_at:
                st.markdown(f"&nbsp;&nbsp;🛒 Will pick up: **{', '.join(items_at)}**")

            # Show the full recommendation details
            if algo_used in ALGO_MAP:
                prod, aisle, path, cost, _ = algo_recommend(algo_used, current, remaining_a, blocked_node)
                if path and len(path) > 1:
                    st.code(f"{algo_used} path: " + " → ".join(path), language=None)

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if st.button(f"✅ Execute {algo_used} move → `{auto_move}`",
                             type="primary", use_container_width=True,
                             key=f"auto_exec_{gs['turn']}"):
                    process_move(gs, auto_move)
                    st.rerun()

            # Option to override
            with st.expander("🔧 Override -- choose manually instead"):
                if not open_options:
                    st.warning("All neighbors blocked! Only option is to wait.")
                    if st.button("⏳ Wait", key=f"override_wait_{gs['turn']}"):
                        process_move(gs, '__WAIT__'); st.rerun()
                else:
                    labels, values = [], []
                    for node, desc in open_options:
                        has_items = any(PRODUCT_MAP.get(p)==node for p in remaining_a)
                        prefix = "🛒 " if has_items else "➡️ "
                        labels.append(f"{prefix} **{node}**  --  {desc}")
                        values.append(node)
                    labels.append("⏳  **Wait** -- Stay at current position")
                    values.append("__WAIT__")

                    override = st.radio("Override destination:", options=values,
                                        format_func=lambda v: labels[values.index(v)],
                                        key=f"override_radio_{gs['turn']}")
                    if st.button("Override & Move", key=f"override_btn_{gs['turn']}"):
                        process_move(gs, override); st.rerun()
        else:
            st.info("Algorithm has no move recommendation. Choose manually:")
            if not open_options:
                if st.button("⏳ Wait", key=f"fallback_wait_{gs['turn']}",
                             type="primary", use_container_width=True):
                    process_move(gs, '__WAIT__'); st.rerun()
            else:
                labels, values = [], []
                for node, desc in open_options:
                    has_items = any(PRODUCT_MAP.get(p)==node for p in remaining_a)
                    labels.append(f"{'🛒' if has_items else '➡️'} **{node}**  --  {desc}")
                    values.append(node)
                labels.append("⏳  **Wait**"); values.append("__WAIT__")
                choice = st.radio("Destination:", values,
                                  format_func=lambda v: labels[values.index(v)],
                                  key=f"fallback_radio_{gs['turn']}")
                if st.button("Confirm", type="primary", key=f"fallback_btn_{gs['turn']}"):
                    process_move(gs, choice); st.rerun()

    # ========== MANUAL MODE ==========
    else:
        if not open_options:
            st.warning("All neighbors are blocked! You must wait.")
            if st.button("⏳ Wait this turn", type="primary", use_container_width=True,
                         key=f"wait_only_{gs['turn']}"):
                process_move(gs, '__WAIT__'); st.rerun()
        else:
            labels = []
            values = []
            for node, desc in open_options:
                has_items = any(PRODUCT_MAP.get(p)==node for p in remaining_a)
                prefix = "🛒 " if has_items else "➡️ "
                labels.append(f"{prefix} **{node}**  --  {desc}")
                values.append(node)
            labels.append("⏳  **Wait** -- Stay at current position")
            values.append("__WAIT__")

            choice = st.radio(
                "**Select your destination:**",
                options=values,
                format_func=lambda v: labels[values.index(v)],
                key=f"move_radio_{gs['turn']}",
                label_visibility="visible"
            )

            # Confirm button
            c_left, c_mid, c_right = st.columns([1, 2, 1])
            with c_mid:
                if choice == '__WAIT__':
                    btn_label = "⏳  Confirm: Wait"
                else:
                    items_at = [p for p in remaining_a if PRODUCT_MAP.get(p)==choice]
                    if items_at:
                        btn_label = f"✅  Move to {choice} and pick up {len(items_at)} item(s)!"
                    else:
                        btn_label = f"➡️  Move to {choice}"

                if st.button(btn_label, type="primary", use_container_width=True,
                             key=f"confirm_{gs['turn']}"):
                    process_move(gs, choice)
                    st.rerun()

# ---- BOTTOM TABS ----
st.markdown("---")
tab_mine, tab_ai, tab_log = st.tabs(["🛍️ Your Shopping List", f"🤖 AI Shopping List ({gs['algo']})", "📜 Game Log"])

with tab_mine:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Still Need")
        for p in gs['a_list']:
            if p not in gs['a_got']:
                st.markdown(f"📌 **{p}** -- Aisle `{PRODUCT_MAP[p]}` {AISLE_CAT.get(PRODUCT_MAP[p],'')}")
    with c2:
        st.markdown("#### Collected ✅")
        if gs['a_got']:
            for p in gs['a_got']:
                st.markdown(f"✅ ~~{p}~~")
        else:
            st.caption("Nothing collected yet. Start moving!")

with tab_ai:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### AI Still Needs")
        for p in gs['b_list']:
            if p not in gs['b_got']:
                st.markdown(f"📌 **{p}** -- Aisle `{PRODUCT_MAP[p]}`")
    with c2:
        st.markdown("#### AI Collected ✅")
        if gs['b_got']:
            for p in gs['b_got']:
                st.markdown(f"✅ ~~{p}~~")
        else:
            st.caption("AI hasn't collected anything yet.")

with tab_log:
    if gs['log']:
        for entry in reversed(gs['log'][-30:]):
            st.markdown(entry)
    else:
        st.info("Game just started -- make your first move!")

st.markdown("---")
col_q, col_s = st.columns([1, 4])
with col_q:
    if st.button("🚪 Quit"):
        del st.session_state.gs; st.rerun()
with col_s:
    st.caption("Store Pathfinder v4.0 | MLC Busy Local Grocery Store")
