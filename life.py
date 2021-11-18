import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Size of the board (the "world" of Game of Life). The board is a square.
BOARD_SIZE = 16

# size of the initial configuration
CONFIG_SIZE = 6

# maximum loops to run Game of Life before giving up
MAX_LOOPS = 30

# States of a board cell.
LIVE = 1
DEAD = 0

# Probability of a cell to be live, when generating a random configuration
LIVE_PROBABILITY = 0.2


def get_neighbors(grid, i, j):
    """ Gets the neighbors of cell (i, j) in the grid.
    An interior cell has 8 neighbors (Moore neighborhood). Cells on the grid borders have 5 neighbors,
    and cells in the corners have 3 neighbors. """
    if i == 0:
        if j == 0:
            neighbors = [grid[i, j + 1],
                         grid[i + 1, j], grid[i + 1, j + 1]]
        elif j == BOARD_SIZE - 1:
            neighbors = [grid[i, j - 1],
                         grid[i + 1, j - 1], grid[i + 1, j]]
        else:
            neighbors = [grid[i, j - 1], grid[i, j + 1],
                         grid[i + 1, j - 1], grid[i + 1, j], grid[i + 1, j + 1]]
    elif i == BOARD_SIZE - 1:
        if j == 0:
            neighbors = [grid[i - 1, j], grid[i - 1, j + 1],
                         grid[i, j + 1]]
        elif j == BOARD_SIZE - 1:
            neighbors = [grid[i - 1, j - 1], grid[i - 1, j],
                         grid[i, j - 1]]
        else:
            neighbors = [grid[i - 1, j - 1], grid[i - 1, j], grid[i - 1, j + 1],
                         grid[i, j - 1], grid[i, j + 1]]
    else:
        if j == 0:
            neighbors = [grid[i - 1, j], grid[i - 1, j + 1],
                         grid[i, j + 1],
                         grid[i + 1, j], grid[i + 1, j + 1]]
        elif j == BOARD_SIZE - 1:
            neighbors = [grid[i - 1, j - 1], grid[i - 1, j],
                         grid[i, j - 1],
                         grid[i + 1, j - 1], grid[i + 1, j]]
        else:
            neighbors = [grid[i - 1, j - 1], grid[i - 1, j], grid[i - 1, j + 1],
                         grid[i, j - 1], grid[i, j + 1],
                         grid[i + 1, j - 1], grid[i + 1, j], grid[i + 1, j + 1]]

    return neighbors


def embed_config_in_grid(config):
    """ Embeds the configuration in the center of an empty grid. """
    x_size, y_size = config.shape
    x = (BOARD_SIZE - x_size) // 2
    y = (BOARD_SIZE - y_size) // 2
    grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    grid[x:x + x_size, y:y + y_size] = config
    return grid


def random_config():
    """ Generates a random configuration and embeds it in the middle of an empty grid. """
    config = np.random.choice([LIVE, DEAD], CONFIG_SIZE * CONFIG_SIZE, p=[LIVE_PROBABILITY, 1-LIVE_PROBABILITY])\
                      .reshape(CONFIG_SIZE, CONFIG_SIZE)
    grid = embed_config_in_grid(config)
    return grid


def random_cell():
    """ Returns coordinates of a random cell within configuration. """
    offset = (BOARD_SIZE - CONFIG_SIZE) // 2
    x = np.random.randint(offset, offset + CONFIG_SIZE)
    y = np.random.randint(offset, offset + CONFIG_SIZE)
    return x, y


class GameOfLife:
    def __init__(self, initial_config=None):
        # the initial configuration
        if initial_config is not None:
            grid = embed_config_in_grid(initial_config)
            self.initial_config = grid
        else:
            self.initial_config = random_config()

        # max. alive cells in any generation until stabilization
        self.max_population = sum(self.initial_config.flatten())

        # generations until stabilization. 0 means it's an oscillator, 1 and above - possible methuselah.
        self.lifespan = -1

        # a configuration is methuselah if its lifespan is greater than 0
        # and stabilizes in a non-empty configuration
        self.is_methuselah = False

        self.checked = False

    def update(self, grid, history):
        """ Updates the grid for the next generation. """
        next_generation = grid.copy()

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                neighbors = get_neighbors(grid, i, j)
                alive_neighbors = sum(neighbors)

                if grid[i, j] == LIVE:
                    if alive_neighbors <= 1 or alive_neighbors >= 4:
                        next_generation[i, j] = DEAD
                if grid[i, j] == DEAD:
                    if alive_neighbors == 3:
                        next_generation[i, j] = LIVE

        if next_generation.tolist() not in history:
            history.append(next_generation.tolist())
            self.max_population = max(self.max_population, sum(next_generation.flatten()))
        else:
            self.lifespan = history.index(next_generation.tolist())
            self.checked = True

        grid[:] = next_generation[:]

    def loop(self):
        """ The main loop of Game of Life. """
        grid = self.initial_config.copy()
        history = [self.initial_config.tolist()]

        i = 0
        while not self.checked and i < MAX_LOOPS:
            self.update(grid, history)
            i += 1

        self.is_methuselah = self.lifespan > 0

    def animate(self, frame_num, img, grid):
        """ Updates the grid for the next generation. """
        next_generation = grid.copy()

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                neighbors = get_neighbors(grid, i, j)
                alive_neighbors = sum(neighbors)

                if grid[i, j] == LIVE:
                    if alive_neighbors <= 1 or alive_neighbors >= 4:
                        next_generation[i, j] = DEAD
                if grid[i, j] == DEAD:
                    if alive_neighbors == 3:
                        next_generation[i, j] = LIVE

        if img is not None:
            img.set_data(next_generation)
        grid[:] = next_generation[:]

    def run_animation(self):
        """ Shows animation of Game of Life based on the initial configuration. """
        grid = self.initial_config.copy()

        update_interval = 200   # delay between frames in milliseconds
        fig, ax = plt.subplots()
        img = ax.imshow(grid, interpolation='nearest')
        ani = animation.FuncAnimation(fig, self.animate, fargs=(img, grid), interval=update_interval, save_count=50)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Game of Life")
    parser.add_argument('--glider', action='store_true', required=False)
    parser.add_argument('--rpentomino', action='store_true', required=False)
    parser.add_argument('--blinker', action='store_true', required=False)
    parser.add_argument('--century', action='store_true', required=False)
    args = parser.parse_args()

    if args.rpentomino:
        init_config = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 1, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0]])
        g = GameOfLife(init_config)
    elif args.glider:
        init_config = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0]])
        g = GameOfLife(init_config)
    elif args.blinker:
        init_config = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]])
        g = GameOfLife(init_config)
    elif args.century:
        init_config = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1],
                                [0, 1, 1, 1, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0]])
        g = GameOfLife(init_config)
    else:
        g = GameOfLife()

    g.run_animation()
