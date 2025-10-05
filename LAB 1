from collections import deque

# Helper function: generate valid moves from current state
def get_moves(state):
    moves = []
    state = list(state)
    for i, r in enumerate(state):
        # Right-facing rabbit
        if r == 'R':
            if i + 1 < len(state) and state[i + 1] == '_':
                new_state = state.copy()
                new_state[i], new_state[i + 1] = new_state[i + 1], new_state[i]
                moves.append(''.join(new_state))
            if i + 2 < len(state) and state[i + 1] == 'L' and state[i + 2] == '_':
                new_state = state.copy()
                new_state[i], new_state[i + 2] = new_state[i + 2], new_state[i]
                moves.append(''.join(new_state))
        # Left-facing rabbit
        if r == 'L':
            if i - 1 >= 0 and state[i - 1] == '_':
                new_state = state.copy()
                new_state[i], new_state[i - 1] = new_state[i - 1], new_state[i]
                moves.append(''.join(new_state))
            if i - 2 >= 0 and state[i - 1] == 'R' and state[i - 2] == '_':
                new_state = state.copy()
                new_state[i], new_state[i - 2] = new_state[i - 2], new_state[i]
                moves.append(''.join(new_state))
    return moves

# BFS Implementation
def bfs(start, goal):
    queue = deque([[start]])
    visited = set()
    while queue:
        path = queue.popleft()
        state = path[-1]
        if state == goal:
            return path
        if state in visited:
            continue
        visited.add(state)
        for next_state in get_moves(state):
            if next_state not in visited:
                queue.append(path + [next_state])
    return None

# DFS Implementation
def dfs(start, goal):
    stack = [[start]]
    visited = set()
    while stack:
        path = stack.pop()
        state = path[-1]
        if state == goal:
            return path
        if state in visited:
            continue
        visited.add(state)
        for next_state in get_moves(state):
            if next_state not in visited:
                stack.append(path + [next_state])
    return None

# Main
if _name_ == "_main_":
    start_state = "RRR_LLL"
    goal_state = "LLL_RRR"

    bfs_path = bfs(start_state, goal_state)
    dfs_path = dfs(start_state, goal_state)

    print("BFS Path:", bfs_path)
    print("DFS Path:", dfs_path)
