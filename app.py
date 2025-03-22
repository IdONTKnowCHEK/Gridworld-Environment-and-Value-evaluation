from flask import Flask, render_template, request, jsonify
import numpy as np
import json

app = Flask(__name__)

# Constants for actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTION_SYMBOLS = ['↑', '→', '↓', '←']

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Configure Flask to use the custom JSON encoder
app.json_encoder = NumpyEncoder

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_random_policy', methods=['POST'])
def generate_random_policy():
    data = request.json
    grid_size = data['gridSize']
    obstacles = data['obstacles']
    start = data['start']
    end = data['end']
    
    # Initialize policy for completely random actions
    policy = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Generate random policy for all cells except obstacles and end
    for i in range(grid_size):
        for j in range(grid_size):
            if [i, j] in obstacles or [i, j] == end:
                policy[i][j] = None
            else:
                # Random action: 0 (up), 1 (right), 2 (down), 3 (left)
                policy[i][j] = int(np.random.choice(ACTIONS))  # Convert to Python int
    
    # Convert policy to symbols
    policy_symbols = [[ACTION_SYMBOLS[policy[i][j]] if policy[i][j] is not None else None 
                      for j in range(grid_size)] for i in range(grid_size)]
    
    # Directly evaluate policy and return values with consistent rewards
    V = evaluate_policy_function(grid_size, obstacles, start, end, policy, False)
    
    return jsonify({
        'policy': policy,
        'policySymbols': policy_symbols,
        'pathFound': False,  # Random policies don't guarantee a path
        'values': V.tolist()
    })

@app.route('/generate_path', methods=['POST'])
def generate_path():
    data = request.json
    grid_size = data['gridSize']
    obstacles = data['obstacles']
    start = data['start']
    end = data['end']
    
    # Initialize policy for path
    policy = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Create a breadth-first search to find a path from start to end
    queue = [start]
    visited = {tuple(start): None}  # Store parent for backtracking
    path_found = False
    
    # Possible movements (up, right, down, left)
    moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    
    while queue and not path_found:
        current = queue.pop(0)
        
        if current == end:
            path_found = True
            break
        
        # Try all four directions
        for idx, move in enumerate(moves):
            new_row = current[0] + move[0]
            new_col = current[1] + move[1]
            new_pos = [new_row, new_col]
            
            # Check if the new position is valid
            if (0 <= new_row < grid_size and 
                0 <= new_col < grid_size and 
                new_pos not in obstacles and
                tuple(new_pos) not in visited):
                
                queue.append(new_pos)
                visited[tuple(new_pos)] = (tuple(current), idx)  # Store parent and action
    
    # If path is found, backtrack to get the policy
    start_action = None
    if path_found:
        current = tuple(end)
        while current != tuple(start):
            parent, action = visited[current]
            row, col = parent
            policy[row][col] = action
            if parent == tuple(start):
                start_action = action  # Store the optimal action from start
            current = parent
    
    # For cells not in the path or that can't reach the end,
    # assign random actions that don't lead to obstacles if possible
    for i in range(grid_size):
        for j in range(grid_size):
            if [i, j] not in obstacles and [i, j] != end and policy[i][j] is None:
                valid_actions = []
                for idx, move in enumerate(moves):
                    new_row = i + move[0]
                    new_col = j + move[1]
                    if (0 <= new_row < grid_size and 
                        0 <= new_col < grid_size and 
                        [new_row, new_col] not in obstacles):
                        valid_actions.append(idx)
                
                if valid_actions:
                    policy[i][j] = int(np.random.choice(valid_actions))
                else:
                    policy[i][j] = int(np.random.choice(ACTIONS))
    
    # Convert policy to symbols
    policy_symbols = [[ACTION_SYMBOLS[policy[i][j]] if policy[i][j] is not None else None 
                      for j in range(grid_size)] for i in range(grid_size)]
    
    # Directly evaluate policy and return values
    V = evaluate_policy_function(grid_size, obstacles, start, end, policy, path_found)
    
    return jsonify({
        'policy': policy,
        'policySymbols': policy_symbols,
        'pathFound': path_found,
        'startAction': start_action,
        'values': V.tolist()
    })

# Extract evaluation logic into a separate function to avoid code duplication
def evaluate_policy_function(grid_size, obstacles, start, end, policy, path_found):
    # Initialize value function V(s)
    V = np.zeros((grid_size, grid_size))
    
    # Consistent reward settings
    goal_reward = 1.0    # positive reward for reaching goal
    obstacle_penalty = -1.0   # penalty for hitting obstacle
    fail_reward = -1.0   # negative reward for paths that cannot reach the end
    step_cost = -0.04   # small cost for each step (encourages shorter paths)
    
    # Initialize values
    if not path_found:
        # If no path is found or we're not using path-based evaluation,
        # all cells except goal have fail reward
        V.fill(fail_reward)
        if end:
            V[end[0], end[1]] = goal_reward
    else:
        # Start with 0 for most cells, will be updated by policy evaluation
        V.fill(0.0)
        if end:
            V[end[0], end[1]] = goal_reward
        
        # For obstacles, apply obstacle penalty
        for obs in obstacles:
            V[obs[0], obs[1]] = obstacle_penalty
    
    # Discount factor
    gamma = 0.9
    
    # Convergence parameters
    theta = 1e-6
    max_iterations = 1000
    
    # Policy evaluation
    for _ in range(max_iterations):
        delta = 0
        for i in range(grid_size):
            for j in range(grid_size):
                # Skip end state and obstacles
                if [i, j] == end:
                    V[i, j] = goal_reward
                    continue
                elif [i, j] in obstacles:
                    V[i, j] = obstacle_penalty
                    continue
                
                old_v = V[i, j]
                
                if policy[i][j] is not None:
                    action = policy[i][j]
                    row, col = i, j
                    
                    # Calculate next state based on action
                    if action == UP and row > 0:
                        row -= 1
                    elif action == RIGHT and col < grid_size - 1:
                        col += 1
                    elif action == DOWN and row < grid_size - 1:
                        row += 1
                    elif action == LEFT and col > 0:
                        col -= 1
                    
                    # Check if next state is valid
                    if [row, col] in obstacles:
                        next_reward = obstacle_penalty
                    elif [row, col] == end:
                        next_reward = goal_reward
                    else:
                        next_reward = step_cost
                    
                    # Update value function
                    V[i, j] = next_reward + gamma * V[row, col]
                    
                    # Calculate delta for convergence check
                    delta = max(delta, abs(old_v - V[i, j]))
        
        if delta < theta:
            break
    
    return V

@app.route('/optimize_path', methods=['POST'])
def optimize_path():
    data = request.json
    grid_size = data['gridSize']
    obstacles = data['obstacles']
    start = data['start']
    end = data['end']
    
    # First, find a path using BFS
    path_policy, path_found, _ = find_path_with_bfs(grid_size, obstacles, start, end)
    
    if not path_found:
        # If no path is found, return with failure
        policy_symbols = [[ACTION_SYMBOLS[path_policy[i][j]] if path_policy[i][j] is not None else None 
                          for j in range(grid_size)] for i in range(grid_size)]
        
        V = evaluate_policy_function(grid_size, obstacles, start, end, path_policy, False)
        
        return jsonify({
            'policy': path_policy,
            'policySymbols': policy_symbols,
            'pathFound': False,
            'values': V.tolist()
        })
    
    # If path is found, run value iteration to find optimal policy
    # Initialize value function
    V = np.zeros((grid_size, grid_size))
    V[end[0], end[1]] = 1.0  # Goal state has value 1
    
    # For obstacles, set negative value
    for obs in obstacles:
        V[obs[0], obs[1]] = -1.0  # Match the obstacle_penalty in evaluate_policy_function
    
    # Discount factor
    gamma = 0.9
    
    # Convergence parameters
    theta = 1e-6
    max_iterations = 1000
    
    # Rewards - ensure consistency with evaluate_policy_function
    goal_reward = 1.0
    obstacle_penalty = -1.0  # Match the obstacle_penalty in evaluate_policy_function
    step_cost = -0.04
    
    # Value iteration
    for _ in range(max_iterations):
        delta = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if [i, j] == end or [i, j] in obstacles:
                    continue
                
                old_v = V[i, j]
                
                # Try all actions and pick the best one
                best_value = float('-inf')
                best_action = None
                
                # Check all four directions
                for action in ACTIONS:
                    row, col = i, j
                    
                    if action == UP and row > 0:
                        row -= 1
                    elif action == RIGHT and col < grid_size - 1:
                        col += 1
                    elif action == DOWN and row < grid_size - 1:
                        row += 1
                    elif action == LEFT and col > 0:
                        col -= 1
                    
                    # Calculate value for this action
                    if [row, col] in obstacles:
                        value = obstacle_penalty + gamma * V[i, j]  # Stay in place
                    elif [row, col] == end:
                        value = goal_reward
                    else:
                        value = step_cost + gamma * V[row, col]
                    
                    if value > best_value:
                        best_value = value
                        best_action = action
                
                # Update value function with best action's value
                V[i, j] = best_value
                
                # Calculate delta for convergence check
                delta = max(delta, abs(old_v - V[i, j]))
        
        if delta < theta:
            break
    
    # Extract optimal policy from value function
    optimal_policy = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    
    for i in range(grid_size):
        for j in range(grid_size):
            if [i, j] == end or [i, j] in obstacles:
                continue
            
            # Find action that leads to highest value
            best_value = float('-inf')
            best_action = None
            
            for action in ACTIONS:
                row, col = i, j
                
                if action == UP and row > 0:
                    row -= 1
                elif action == RIGHT and col < grid_size - 1:
                    col += 1
                elif action == DOWN and row < grid_size - 1:
                    row += 1
                elif action == LEFT and col > 0:
                    col -= 1
                
                # Skip invalid moves
                if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
                    continue
                
                # Calculate value for this action
                if [row, col] in obstacles:
                    value = obstacle_penalty + gamma * V[i, j]
                elif [row, col] == end:
                    value = goal_reward
                else:
                    value = step_cost + gamma * V[row, col]
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            optimal_policy[i][j] = best_action
    
    # Ensure start point uses the optimal direction
    # This will be the direction that leads to the highest value
    
    # For cells not in the path, use random valid actions
    for i in range(grid_size):
        for j in range(grid_size):
            if [i, j] not in obstacles and [i, j] != end and optimal_policy[i][j] is None:
                valid_actions = []
                for idx, move in enumerate(moves):
                    new_row = i + move[0]
                    new_col = j + move[1]
                    if (0 <= new_row < grid_size and 
                        0 <= new_col < grid_size and 
                        [new_row, new_col] not in obstacles):
                        valid_actions.append(idx)
                
                if valid_actions:
                    optimal_policy[i][j] = int(np.random.choice(valid_actions))
                else:
                    optimal_policy[i][j] = int(np.random.choice(ACTIONS))
    
    # Convert policy to symbols
    policy_symbols = [[ACTION_SYMBOLS[optimal_policy[i][j]] if optimal_policy[i][j] is not None else None 
                      for j in range(grid_size)] for i in range(grid_size)]
    
    return jsonify({
        'policy': optimal_policy,
        'policySymbols': policy_symbols,
        'pathFound': True,
        'values': V.tolist()
    })

def find_path_with_bfs(grid_size, obstacles, start, end):
    # Initialize policy
    policy = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    
    # BFS algorithm
    queue = [start]
    visited = {tuple(start): None}  # Store parent for backtracking
    path_found = False
    
    # Possible movements (up, right, down, left)
    moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    
    while queue and not path_found:
        current = queue.pop(0)
        
        if current == end:
            path_found = True
            break
        
        # Try all four directions
        for idx, move in enumerate(moves):
            new_row = current[0] + move[0]
            new_col = current[1] + move[1]
            new_pos = [new_row, new_col]
            
            # Check if the new position is valid
            if (0 <= new_row < grid_size and 
                0 <= new_col < grid_size and 
                new_pos not in obstacles and
                tuple(new_pos) not in visited):
                
                queue.append(new_pos)
                visited[tuple(new_pos)] = (tuple(current), idx)  # Store parent and action
    
    # If path is found, backtrack to get the policy
    start_action = None
    if path_found:
        current = tuple(end)
        while current != tuple(start):
            parent, action = visited[current]
            row, col = parent
            policy[row][col] = action
            if parent == tuple(start):
                start_action = action
            current = parent
    
    # For cells not in the path, use random valid actions
    for i in range(grid_size):
        for j in range(grid_size):
            if [i, j] not in obstacles and [i, j] != end and policy[i][j] is None:
                valid_actions = []
                for idx, move in enumerate(moves):
                    new_row = i + move[0]
                    new_col = j + move[1]
                    if (0 <= new_row < grid_size and 
                        0 <= new_col < grid_size and 
                        [new_row, new_col] not in obstacles):
                        valid_actions.append(idx)
                
                if valid_actions:
                    policy[i][j] = int(np.random.choice(valid_actions))
                else:
                    policy[i][j] = int(np.random.choice(ACTIONS))
    
    return policy, path_found, start_action

# Keep the evaluate_policy route for backward compatibility, but it's no longer needed
@app.route('/evaluate_policy', methods=['POST'])
def evaluate_policy():
    data = request.json
    grid_size = data['gridSize']
    obstacles = data['obstacles']
    start = data['start']
    end = data['end']
    policy = data['policy']
    path_found = data.get('pathFound', False)
    
    V = evaluate_policy_function(grid_size, obstacles, start, end, policy, path_found)
    
    return jsonify({
        'values': V.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
