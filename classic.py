import pygame
import random
import time  # Import time module for timing

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
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = self.new_piece()
        self.piece_x = GRID_WIDTH // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0
        self.combo = 0
        self.score = 0
        self.total_lines_cleared = 0  # Total lines cleared counter
        self.total_combo = 0  # Total combo counter
        self.game_over = False
        self.fall_time = 0
        self.fall_speed = 500  # Milliseconds
        self.time_limit = TIME_LIMIT  # Set time limit
        self.start_time = time.time()  # Track start time
        self.elapsed_time = 0  # Track elapsed time


    def new_piece(self):
        return random.choice(TETROMINOES)

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

    def display_counters(self):
        """Display the counters in the console."""
        print(f"Total Lines Cleared: {self.total_lines_cleared}")
        print(f"Total Combos: {self.total_combo}")
        print(f"Current Score: {self.score}")

    def lock_piece(self):
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[y + self.piece_y][x + self.piece_x] = cell
        self.clear_lines()
        self.current_piece = self.new_piece()
        self.piece_x = GRID_WIDTH // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0
        if self.check_collision(self.current_piece, self.piece_x, self.piece_y):
            self.game_over = True

    def clear_lines(self):
        lines_to_clear = [y for y, row in enumerate(self.grid) if all(row)]
        num_lines_cleared = len(lines_to_clear)  # Number of lines cleared at once

        if num_lines_cleared > 0:
            # Update score based on traditional Tetris scoring
            if num_lines_cleared == 1:
                self.score += 40
            elif num_lines_cleared == 2:
                self.score += 100
            elif num_lines_cleared == 3:
                self.score += 300
            elif num_lines_cleared == 4:
                self.score += 1200

            # Update total lines cleared
            self.total_lines_cleared += num_lines_cleared

            # Increase combo count and add bonus score for streaks
            self.combo += 1
            combo_bonus = 50 * self.combo  # Bonus points increase with combo count
            self.score += combo_bonus

            # Update total combo counter if it's a new combo
            if self.combo == 1:
                self.total_combo += 1

        else:
            # Reset combo counter if no lines were cleared
            self.combo = 0

        # Clear the lines from the grid
        for y in lines_to_clear:
            del self.grid[y]
            self.grid.insert(0, [0] * GRID_WIDTH)

    def check_time_limit(self):
        """Check if the time limit has been reached."""
        self.elapsed_time = time.time() - self.start_time  # Update elapsed time
        if self.elapsed_time >= self.time_limit:
            self.game_over = True  # End the game if time limit is reached
            print("Time's up! Game over.")

    def update(self, dt):
        self.display_counters()
        self.check_time_limit()

        self.fall_time += dt
        if self.fall_time > self.fall_speed:
            self.fall_time = 0
            if not self.move_piece(0, 1):
                self.lock_piece()










# HEURISTIC-BASED AI



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

def choose_best_move(tetris):
    piece = tetris.current_piece
    grid = tetris.grid
    possible_moves = generate_moves(piece, grid)
    best_move = max(possible_moves, key=lambda move: move[0], default=None)
    return best_move  # (score, x_position, rotation)

def apply_move(tetris, best_move):
    if best_move is None:
        return
    score, x_position, rotation = best_move
    for _ in range(rotation):
        tetris.rotate_piece()
    while tetris.piece_x < x_position:
        tetris.move_piece(1, 0)
    while tetris.piece_x > x_position:
        tetris.move_piece(-1, 0)
    tetris.hard_drop()


# Main game loop
def game_loop():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Tetris AI')
    clock = pygame.time.Clock()
    tetris = Tetris()

    start_time = time.time()  # Start timing

    while not tetris.game_over:
        dt = clock.tick(60) # Limit to 60 frames per second
        screen.fill((0, 0, 0))
        tetris.update(dt)
        tetris.draw_grid(screen)
        tetris.draw_piece(screen, tetris.current_piece, tetris.piece_x, tetris.piece_y)

        # AI Move Selection
        best_move = choose_best_move(tetris)
        apply_move(tetris, best_move)

        pygame.display.flip()

        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print(f"Elapsed time: {elapsed_time:.2f} seconds | Score: {tetris.score}")  # Print elapsed time to console

    print("Game Over! Score:", tetris.score)
    pygame.quit()

if __name__ == "__main__":
    pygame.init()
    game_loop()
