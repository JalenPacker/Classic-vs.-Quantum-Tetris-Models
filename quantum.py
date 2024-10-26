import pygame
import random
import time
import cirq  # Import Cirq for quantum computing


# Constants for the game
GRID_WIDTH = 10
GRID_HEIGHT = 20
BLOCK_SIZE = 30
SCREEN_WIDTH = GRID_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * BLOCK_SIZE
TIME_LIMIT = 500


# Colors for Tetris pieces (matches original Tetris colors)
COLORS = [
    (0, 0, 0),  # Empty space
    (0, 255, 255),  # Cyan for I
    (0, 0, 255),  # Blue for J
    (255, 165, 0),  # Orange for L
    (255, 255, 0),  # Yellow for O
    (0, 255, 0),  # Green for S
    (128, 0, 128),  # Purple for T
    (255, 0, 0),  # Red for Z
]

# Tetromino shapes
TETROMINOES = [
    [[1, 1, 1, 1]],  # I
    [[2, 0, 0], [2, 2, 2]],  # J
    [[0, 0, 3], [3, 3, 3]],  # L
    [[4, 4], [4, 4]],  # O
    [[0, 5, 5], [5, 5, 0]],  # S
    [[0, 6, 0], [6, 6, 6]],  # T
    [[7, 7, 0], [0, 7, 7]]  # Z
]

# Game class
class Tetris:
    def __init__(self):
        self.start_time = time.time()  # Track start time
        self.elapsed_time = 0  # Track elapsed time
        self.time_limit = TIME_LIMIT  # Set time limit
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = None
        self.piece_x = 0
        self.piece_y = 0
        self.bag = []
        self.fill_bag()
        self.spawn_new_piece()
        self.score = 0
        self.combo_streak = 0
        self.total_combos = 0  # Total combo counter
        self.lines_cleared = 0  # Number of lines cleared
        self.game_over = False
        self.fall_time = 0
        self.fall_speed = 200  # Adjust fall speed
        self.previous_lines_cleared = 0  # Track lines cleared in the previous move

    def fill_bag(self):
        """Fill the bag by shuffling all seven Tetrominoes."""
        self.bag = TETROMINOES[:]  # Copy the list of Tetriminos
        random.shuffle(self.bag)  # Shuffle the bag

    def new_piece(self):
        """Draw a new piece from the bag, refill the bag if empty."""
        if not self.bag:
            self.fill_bag()
        return self.bag.pop()

    def spawn_new_piece(self):
        """Spawn a new piece in the middle of the grid."""
        self.current_piece = self.new_piece()
        self.piece_x = GRID_WIDTH // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0

    def draw_grid(self, screen):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                pygame.draw.rect(screen, COLORS[self.grid[y][x]],
                                 pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)
                pygame.draw.rect(screen, (128, 128, 128),
                                 pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

    def draw_piece(self, screen, piece, x_offset, y_offset):
        for y, row in enumerate(piece):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, COLORS[cell],
                                     pygame.Rect((x + x_offset) * BLOCK_SIZE, (y + y_offset) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

    def rotate_piece(self):
        self.current_piece = [list(row) for row in zip(*self.current_piece[::-1])]

    def move_piece(self, dx, dy):
        self.piece_x += dx
        if self.check_collision(self.current_piece, self.piece_x, self.piece_y):
            self.piece_x -= dx

        self.piece_y += dy
        if self.check_collision(self.current_piece, self.piece_x, self.piece_y):
            self.piece_y -= dy
            return False
        return True

    def hard_drop(self):
        while not self.check_collision(self.current_piece, self.piece_x, self.piece_y + 1):
            self.piece_y += 1
        self.lock_piece()

    def check_collision(self, piece, x_offset, y_offset):
        for y, row in enumerate(piece):
            for x, cell in enumerate(row):
                if cell:
                    if (x + x_offset >= GRID_WIDTH or
                            x + x_offset < 0 or
                            y + y_offset >= GRID_HEIGHT or
                            self.grid[y + y_offset][x + x_offset]):
                        return True
        return False

    def lock_piece(self):
        # Lock the current piece into the grid
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    grid_y = y + self.piece_y
                    grid_x = x + self.piece_x
                    # Ensure we're not trying to access grid cells out of bounds
                    if 0 <= grid_y < GRID_HEIGHT and 0 <= grid_x < GRID_WIDTH:
                        self.grid[grid_y][grid_x] = cell

        # Check for game over: if any block is locked into the top row
        if any(self.grid[0][x] != 0 for x in range(GRID_WIDTH)):
            self.game_over = True

        # Clear any full lines
        self.clear_lines()

        # Spawn a new piece only if the game is not over
        if not self.game_over:
            self.spawn_new_piece()

            # Check if the new piece collides immediately after spawning (another game over condition)
            if self.check_collision(self.current_piece, self.piece_x, self.piece_y):
                self.game_over = True

    def clear_lines(self):
        lines_to_clear = [y for y, row in enumerate(self.grid) if all(row)]
        num_lines_cleared = len(lines_to_clear)

        if num_lines_cleared > 0:
            # Check if this is a combo (requires previous lines cleared to be > 0)
            if self.previous_lines_cleared > 0:
                self.combo_streak += 1  # Increment combo streak
                self.total_combos += 1  # Increment total combos only once
            else:
                self.combo_streak = 1  # Start a new combo streak

            self.lines_cleared += num_lines_cleared  # Update lines cleared
            self.score += self.calculate_score(num_lines_cleared)  # Update score based on cleared lines

        else:
            # Reset combo streak if no lines are cleared
            self.combo_streak = 0

        # Update previous lines cleared for the next move
        self.previous_lines_cleared = num_lines_cleared

        # Clear the lines
        for y in lines_to_clear:
            del self.grid[y]
            self.grid.insert(0, [0] * GRID_WIDTH)

    def calculate_score(self, num_lines):
        """Calculate score based on the number of lines cleared."""
        if num_lines == 1:
            return 40
        elif num_lines == 2:
            return 100
        elif num_lines == 3:
            return 300
        elif num_lines == 4:
            return 1200
        return 0

    def check_time_limit(self):
        """Check if the time limit has been reached."""
        self.elapsed_time = time.time() - self.start_time  # Update elapsed time
        if self.elapsed_time >= self.time_limit:
            self.game_over = True  # End the game if time limit is reached
            print("Time's up! Game over.")

    def find_best_move(self):
        """Find the best move by simulating all possible placements and choosing the highest scoring one."""
        possible_moves = generate_moves(self.current_piece, self.grid)
        best_move = max(possible_moves, key=lambda move: move[0])  # Maximize score
        return best_move  # Return the best score, x position, and rotation

    def update(self, dt):
        # Call the time limit check function
        self.check_time_limit()

        if not self.game_over:
            # Evaluate the best move using AI-based piece placement
            best_score, best_x, best_rotation = self.find_best_move()

            # Apply the best move
            for _ in range(best_rotation):
                self.rotate_piece()

            # Move the piece to the best x position
            while self.piece_x < best_x:
                self.move_piece(1, 0)
            while self.piece_x > best_x:
                self.move_piece(-1, 0)

            # Perform a hard drop after the best move is found
            self.hard_drop()

    def draw_score(self, screen):
        """Draw the score, total combos, lines cleared, and elapsed time on the screen."""
        font = pygame.font.SysFont("Arial", 24)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        combos_text = font.render(f"Total Combos: {self.total_combos}", True, (255, 255, 255))
        lines_text = font.render(f"Lines Cleared: {self.lines_cleared}", True, (255, 255, 255))

        # Calculate elapsed time
        self.elapsed_time = time.time() - self.start_time  # Update elapsed time
        elapsed_time_text = font.render(f"Elapsed Time: {int(self.elapsed_time)}s", True, (255, 255, 255))

        # Blit the texts to the screen
        screen.blit(score_text, (10, 10))
        screen.blit(combos_text, (10, 40))
        screen.blit(lines_text, (10, 70))
        screen.blit(elapsed_time_text, (10, 100))  # Display elapsed time below other stats

def evaluate_grid(grid):
    total_height = 0
    holes = 0
    complete_lines = 0
    bumpiness = 0

    column_heights = [0] * GRID_WIDTH

    for x in range(GRID_WIDTH):
        column_filled = False
        column_height = 0
        for y in range(GRID_HEIGHT):
            if grid[y][x]:
                if not column_filled:
                    column_height = GRID_HEIGHT - y
                    column_heights[x] = column_height
                    column_filled = True
            elif column_filled:
                holes += 1

    for i in range(GRID_WIDTH - 1):
        bumpiness += abs(column_heights[i] - column_heights[i + 1])

    for row in grid:
        if all(row):
            complete_lines += 1

    total_height = sum(column_heights)

    return (-0.51 * total_height) + (0.76 * complete_lines) - (0.36 * holes) - (0.18 * bumpiness)

def generate_moves(piece, grid):
    possible_moves = []
    for rotation in range(4):
        rotated_piece = rotate_piece_n_times(piece, rotation)
        for x_position in range(GRID_WIDTH - len(rotated_piece[0]) + 1):
            simulated_grid = simulate_move(grid, rotated_piece, x_position)
            score = evaluate_grid(simulated_grid)
            possible_moves.append((score, x_position, rotation))
    return possible_moves

def rotate_piece_n_times(piece, n):
    rotated_piece = piece
    for _ in range(n):
        rotated_piece = [list(row) for row in zip(*rotated_piece[::-1])]
    return rotated_piece

def simulate_move(grid, piece, x_position):
    new_grid = [row[:] for row in grid]
    y_position = 0
    while not collision(new_grid, piece, x_position, y_position):
        y_position += 1
    lock_piece_simulated(new_grid, piece, x_position, y_position - 1)
    return new_grid

def collision(grid, piece, x_offset, y_offset):
    for y, row in enumerate(piece):
        for x, cell in enumerate(row):
            if cell:
                if (x + x_offset >= GRID_WIDTH or
                        x + x_offset < 0 or
                        y + y_offset >= GRID_HEIGHT or
                        grid[y + y_offset][x + x_offset]):
                    return True
    return False

def lock_piece_simulated(grid, piece, x_offset, y_offset):
    for y, row in enumerate(piece):
        for x, cell in enumerate(row):
            if cell:
                grid[y + y_offset][x + x_offset] = cell


# Function to quantum-enhance the move selection
def quantum_enhanced_choice(possible_moves):
    # Only consider the top 3 moves to add randomness in choice
    top_moves = sorted(possible_moves, key=lambda x: x[0], reverse=True)[:3]

    # Define qubits and a circuit
    qubits = [cirq.LineQubit(i) for i in range(2)]  # 2 qubits for a choice among 3 options
    circuit = cirq.Circuit()

    # Create a quantum superposition
    circuit.append([cirq.H(qubits[0]), cirq.H(qubits[1])])

    # Measure the qubits
    circuit.append([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')])

    # Run the quantum circuit
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1)
    result_bits = result.measurements['q0'][0][0] * 2 + result.measurements['q1'][0][0]

    # Map quantum result to a move selection (0, 1, 2)
    if result_bits >= len(top_moves):
        result_bits = len(top_moves) - 1  # Handle overflow due to quantum randomness

    selected_move = top_moves[result_bits]
    return selected_move

def choose_best_move(tetris):
    piece = tetris.current_piece
    grid = tetris.grid
    possible_moves = generate_moves(piece, grid)

    # Use the quantum-enhanced decision to select the move
    best_move = quantum_enhanced_choice(possible_moves)
    return best_move

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris with Quantum AI")
    clock = pygame.time.Clock()
    tetris = Tetris()

    while not tetris.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                tetris.game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    tetris.move_piece(-1, 0)
                if event.key == pygame.K_RIGHT:
                    tetris.move_piece(1, 0)
                if event.key == pygame.K_DOWN:
                    tetris.move_piece(0, 1)
                if event.key == pygame.K_UP:
                    tetris.rotate_piece()

        # Update the game state
        dt = clock.tick(30)  # Limit to 30 frames per second
        tetris.update(dt)

        # Draw everything
        screen.fill((0, 0, 0))  # Clear the screen
        tetris.draw_grid(screen)  # Draw the grid
        tetris.draw_piece(screen, tetris.current_piece, tetris.piece_x, tetris.piece_y)  # Draw current piece
        tetris.draw_score(screen)  # Draw the score and elapsed time
        pygame.display.flip()  # Update the display

    pygame.quit()

if __name__ == "__main__":
    main()

pygame.quit()