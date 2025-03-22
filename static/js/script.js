$(document).ready(function() {
    // Grid state variables
    let gridSize = 5;
    let grid = [];
    let start = null;
    let end = null;
    let obstacles = [];
    let maxObstacles = gridSize - 2;
    let currentMode = 'setStart'; // Modes: setStart, setEnd, setObstacles
    let policy = null;
    let values = null;
    let pathFound = false;
    
    // Initialize the application
    updateMode();
    
    // Event listeners
    $('#create-grid').click(function() {
        gridSize = parseInt($('#grid-size').val());
        
        // Validate grid size
        if (gridSize < 5 || gridSize > 9) {
            alert('Grid size must be between 5 and 9');
            return;
        }
        
        // Reset grid state
        start = null;
        end = null;
        obstacles = [];
        maxObstacles = gridSize - 2;
        currentMode = 'setStart';
        policy = null;
        values = null;
        pathFound = false;
        
        // Update UI
        $('#max-obstacles').text(maxObstacles);
        $('#obstacle-count').text('0');
        $('#random-policy').prop('disabled', true);
        $('#evaluate-policy').prop('disabled', true);
        
        createGrid();
        updateMode();
    });
    
    $('#reset-grid').click(function() {
        // Reset grid state but maintain size
        start = null;
        end = null;
        obstacles = [];
        currentMode = 'setStart';
        policy = null;
        values = null;
        pathFound = false;
        
        // Update UI
        $('#obstacle-count').text('0');
        $('#random-policy').prop('disabled', true);
        $('#evaluate-policy').prop('disabled', true);
        
        createGrid();
        updateMode();
    });
    
    $('#random-policy').click(function() {
        generateRandomPolicy();
    });
    
    // Remove or disable the evaluate policy button since evaluation is automatic
    $('#evaluate-policy').prop('disabled', true).hide();
    
    // Update the button-group with all three buttons
    $('.button-group').empty().append(
        $('<button>')
            .attr('id', 'random-policy')
            .addClass('btn secondary')
            .text('Random Policy')
            .prop('disabled', true)
            .click(function() {
                generateRandomPolicy();
            }),
            
        $('<button>')
            .attr('id', 'generate-path')
            .addClass('btn secondary')
            .text('Generate Path')
            .prop('disabled', true)
            .click(function() {
                generatePathFromStart();
            }),
            
        $('<button>')
            .attr('id', 'optimize-path')
            .addClass('btn primary')
            .text('Optimal Path')
            .prop('disabled', true)
            .click(function() {
                generateOptimalPath();
            })
    );
    
    // Function to create the grid
    function createGrid() {
        const $grid = $('#grid');
        $grid.empty();
        $grid.css('grid-template-columns', `repeat(${gridSize}, 1fr)`);
        
        grid = [];
        
        for (let i = 0; i < gridSize; i++) {
            const row = [];
            for (let j = 0; j < gridSize; j++) {
                const $cell = $('<div>')
                    .addClass('cell')
                    .attr('data-row', i)
                    .attr('data-col', j)
                    .appendTo($grid);
                
                $cell.click(function() {
                    handleCellClick(i, j);
                });
                
                row.push({
                    element: $cell,
                    type: 'empty'
                });
            }
            grid.push(row);
        }
    }
    
    // Function to handle cell clicks based on current mode
    function handleCellClick(row, col) {
        if (currentMode === 'setStart') {
            // Clear previous start if exists
            if (start !== null) {
                grid[start[0]][start[1]].element.removeClass('start');
                grid[start[0]][start[1]].type = 'empty';
            }
            
            // Set new start
            start = [row, col];
            grid[row][col].element.addClass('start');
            grid[row][col].type = 'start';
            
            // Move to next mode
            currentMode = 'setEnd';
            updateMode();
            
        } else if (currentMode === 'setEnd') {
            // Check if trying to set end on start
            if (row === start[0] && col === start[1]) {
                return;
            }
            
            // Clear previous end if exists
            if (end !== null) {
                grid[end[0]][end[1]].element.removeClass('end');
                grid[end[0]][end[1]].type = 'empty';
            }
            
            // Set new end
            end = [row, col];
            grid[row][col].element.addClass('end');
            grid[row][col].type = 'end';
            
            // Move to next mode
            currentMode = 'setObstacles';
            updateMode();
            
            // Enable all policy buttons
            $('#random-policy').prop('disabled', false);
            $('#generate-path').prop('disabled', false);
            $('#optimize-path').prop('disabled', false);
            
        } else if (currentMode === 'setObstacles') {
            // Check if clicking on start or end
            if ((row === start[0] && col === start[1]) || (row === end[0] && col === end[1])) {
                return;
            }
            
            const cellElement = grid[row][col].element;
            const isObstacle = cellElement.hasClass('obstacle');
            
            if (isObstacle) {
                // Remove obstacle
                cellElement.removeClass('obstacle');
                grid[row][col].type = 'empty';
                obstacles = obstacles.filter(obs => obs[0] !== row || obs[1] !== col);
                $('#obstacle-count').text(obstacles.length);
            } else if (obstacles.length < maxObstacles) {
                // Add obstacle
                cellElement.addClass('obstacle');
                grid[row][col].type = 'obstacle';
                obstacles.push([row, col]);
                $('#obstacle-count').text(obstacles.length);
            }
        }
    }
    
    // Update the current mode display
    function updateMode() {
        let modeText = '';
        let modeClass = '';
        
        switch (currentMode) {
            case 'setStart':
                modeText = 'Set Start';
                modeClass = 'mode-start';
                break;
            case 'setEnd':
                modeText = 'Set End';
                modeClass = 'mode-end';
                break;
            case 'setObstacles':
                modeText = 'Set Obstacles';
                modeClass = 'mode-obstacles';
                break;
        }
        
        const $modeIndicator = $('#current-mode');
        $modeIndicator
            .text(modeText)
            .removeClass('mode-start mode-end mode-obstacles')
            .addClass(modeClass);
    }
    
    // Generate a path from start to end
    function generatePathFromStart() {
        if (start === null || end === null) {
            alert('Please set start and end positions first');
            return;
        }
        
        $.ajax({
            url: '/generate_path',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                gridSize: gridSize,
                obstacles: obstacles,
                start: start,
                end: end
            }),
            success: function(response) {
                policy = response.policy;
                pathFound = response.pathFound;
                values = response.values;  // Get values directly from response
                
                displayPolicy(response.policySymbols);
                displayValues(values);     // Display values immediately
                
                if (pathFound) {
                    showNotification('Path found from start to end!', 'success');
                } else {
                    showNotification('No valid path to end. Random policy applied.', 'warning');
                }
            },
            error: function(error) {
                console.error('Error generating path:', error);
                alert('Failed to generate path');
            }
        });
    }
    
    // Show notification message
    function showNotification(message, type) {
        // Create notification element if it doesn't exist
        if ($('#notification').length === 0) {
            $('<div>')
                .attr('id', 'notification')
                .addClass('notification')
                .appendTo('body');
        }
        
        const $notification = $('#notification');
        $notification.text(message)
            .removeClass('success warning error')
            .addClass(type)
            .fadeIn()
            .delay(3000)
            .fadeOut();
    }
    
    // Generate a random policy
    function generateRandomPolicy() {
        if (start === null || end === null) {
            alert('Please set start and end positions first');
            return;
        }
        
        $.ajax({
            url: '/generate_random_policy',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                gridSize: gridSize,
                obstacles: obstacles,
                start: start,
                end: end
            }),
            success: function(response) {
                policy = response.policy;
                values = response.values;
                pathFound = response.pathFound;  // Get pathFound status
                
                displayPolicy(response.policySymbols);
                displayValues(values);
                
                if (pathFound) {
                    showNotification('Path found with random policy!', 'success');
                } else {
                    showNotification('Random policy generated, no path to end', 'warning');
                }
            },
            error: function(error) {
                console.error('Error generating policy:', error);
                alert('Failed to generate policy');
            }
        });
    }
    
    // Generate optimal path using value iteration
    function generateOptimalPath() {
        if (start === null || end === null) {
            alert('Please set start and end positions first');
            return;
        }
        
        $.ajax({
            url: '/optimize_path',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                gridSize: gridSize,
                obstacles: obstacles,
                start: start,
                end: end
            }),
            success: function(response) {
                policy = response.policy;
                pathFound = response.pathFound;
                values = response.values;
                
                displayPolicy(response.policySymbols);
                displayValues(values);
                
                if (pathFound) {
                    showNotification('Optimal path found!', 'success');
                } else {
                    showNotification('No valid path to end exists.', 'error');
                }
            },
            error: function(error) {
                console.error('Error finding optimal path:', error);
                alert('Failed to find optimal path');
            }
        });
    }
    
    // Display policy on the grid
    function displayPolicy(policySymbols) {
        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                const cell = grid[i][j];
                
                // Clear existing policy arrows
                cell.element.find('.policy-arrow').remove();
                
                if (policySymbols[i][j] !== null) {
                    $('<div>')
                        .addClass('policy-arrow')
                        .text(policySymbols[i][j])
                        .appendTo(cell.element);
                }
            }
        }
    }
    
    // The evaluate policy function is kept for compatibility but no longer needed to be called directly
    function evaluatePolicy() {
        if (policy === null) {
            alert('Please generate a policy first');
            return;
        }
        
        $.ajax({
            url: '/evaluate_policy',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                gridSize: gridSize,
                obstacles: obstacles,
                start: start,
                end: end,
                policy: policy,
                pathFound: pathFound
            }),
            success: function(response) {
                values = response.values;
                displayValues(values);
            },
            error: function(error) {
                console.error('Error evaluating policy:', error);
                alert('Failed to evaluate policy');
            }
        });
    }
    
    // Display state values on the grid
    function displayValues(values) {
        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                const cell = grid[i][j];
                
                // Clear existing value displays
                cell.element.find('.value').remove();
                
                // Format value to 2 decimal places
                const formattedValue = values[i][j].toFixed(2);
                
                $('<div>')
                    .addClass('value')
                    .text(formattedValue)
                    .appendTo(cell.element);
            }
        }
    }
    
    // Initial grid creation
    createGrid();
});
