# Grid Map & Policy Evaluation Web Application

This Flask application allows users to create interactive grid maps, set start and end points, place obstacles, and evaluate reinforcement learning policies.

## Features

- **Interactive Grid Map**: Create customizable grid maps (5x5 to 9x9)
- **Set Start/End Points**: Define starting point (green) and end goal (red)
- **Place Obstacles**: Add up to n-2 obstacles (gray)
- **Path Generation**: Attempt to find paths from start to end
- **Random Policy Generation**: Generate policies with preference for paths to the goal
- **Automatic Value Evaluation**: Calculate and display state values immediately

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000/`

## Usage Instructions

1. Set grid size (5-9) and click "Create Grid"
2. Click on a cell to set the start position (green)
3. Click on another cell to set the end position (red)
4. Click on cells to toggle obstacles (gray), maximum n-2 obstacles
5. Click "Generate Random Policy" to create random policies (attempts to find path to end)
6. Click "Generate Path" to find the optimal path from start to end
7. Use "Reset Grid" to clear the grid while maintaining size

## Technical Implementation

- **Backend**: Flask with NumPy for policy evaluation calculations
- **Frontend**: HTML5, CSS3, JavaScript with jQuery
- **Algorithm**: BFS for path finding and policy evaluation for value calculation
