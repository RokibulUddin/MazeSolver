__author__='Rokibul Uddin'

import copy
import re, os
import pygame
from queue import PriorityQueue
from PIL import Image, ImageDraw
import sys
import random

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

DRAW_GRID = True
QUICK = False

class Cell:
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self):
        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False

class Maze:
    def __init__(self, nx, ny, ix=0, iy=0):
        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def cell_at(self, x, y):
        return self.maze_map[x][y]

    def __str__(self):
        maze_rows = ['-' * self.nx*2]
        for y in range(self.ny):
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['E']:
                    maze_row.append(' |')
                else:
                    maze_row.append('  ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return '\n'.join(maze_rows)

    def write_img(self, rows):
        aspect_ratio = self.nx / self.ny
        height = rows * 2
        width = int(height * aspect_ratio)
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = height / self.ny, width / self.nx

        img = Image.new('L', (width, width), color='white')
        draw = ImageDraw.Draw(img)

        for x in range(self.nx):
            for y in range(self.ny):
                if self.cell_at(x,y).walls['S']:
                    x1, y1, x2, y2 = x*scx, (y+1)*scy, (x+1)*scx, (y+1)*scy
                    draw.line([x1,y1,x2,y2], fill='black')
                if self.cell_at(x,y).walls['E']:
                    x1, y1, x2, y2 = (x+1)*scx, y*scy, (x+1)*scx, (y+1)*scy
                    draw.line([x1, y1, x2, y2], fill='black')
        return img


    def find_valid_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [('W', (-1,0)),
                 ('E', (1,0)),
                 ('S', (0,1)),
                 ('N', (0,-1))]
        neighbours = []
        for direction, (dx,dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self):
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)

            if not neighbours:
                # We've reached a dead end: backtrack.
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, next_cell = random.choice(neighbours)
            current_cell.knock_down_wall(next_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1

class FandH:
    def __init__(self, f, h):
        self.f = f
        self.h = h

    def __lt__(self, obj):
        if self.f == obj.f :
            return self.h < obj.h
        return self.f < obj.f

    def __le__(self, obj):
        if self.f == obj.f :
            return self.h <= obj.h
        return self.f <= obj.f

    def __eq__(self, obj):
        return (self.f == obj.f) and (self.h == obj.h)

    def __ne__(self, obj):
        return not self.__eq__(obj)

    def __gt__(self, obj):
        if self.f == obj.f :
            return self.h > obj.h
        return self.f > obj.f

    def __ge__(self, obj):
        if self.f == obj.f :
            return self.h >= obj.h
        return self.f >= obj.f



class Spot:
    def __init__(self, row, col, width, total_rows, barrier=False):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE if not barrier else BLACK
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def is_path(self):
        return self.color == PURPLE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw):
    last = None
    while current in came_from:
        current = came_from[current]
        current.make_path()
        last = current
        if not QUICK:
            draw()
    last.make_start()
    if not QUICK:
        draw()

def reset_grid(grid, draw):
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            item = grid[x][y]
            if item.is_closed() or item.is_open():
                item.reset()
    draw()

def algorithm(draw, grid, start, end):
    global QUICK
    count = 0
    open_set = PriorityQueue()
    open_set.put((FandH(0, 0), count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            #global DRAW_GRID
            #DRAW_GRID = False
            #reset_grid(grid, draw)
            reconstruct_path(came_from, end, draw)
            draw()
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                h_score = h(neighbor.get_pos(), end.get_pos())
                f_score[neighbor] = temp_g_score + h_score
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((FandH(f_score[neighbor], h_score), count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        if not QUICK:
            draw()

        if current != start:
            current.make_closed()

    return False

def grid_cursor(grid):
    x = len(grid)
    y = len(grid[0])
    for i in range(x):
        for j in range(y):
            yield grid[i][j]

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid


def draw_grid(win, rows, width):
    global DRAW_GRID
    if DRAW_GRID:
        gap = width // rows
        for i in range(rows):
            pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
            for j in range(rows):
                pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def copy_grid(grid):
    x = len(grid)
    y = len(grid[0])
    backup = []
    for i in range(x):
        backup.append([])
        for j in range(y):
            s = grid[i][j]
            backup[i].append(Spot(i, j, s.width, s.total_rows))
            backup[i][j].color = copy.copy(s.color)
    return backup


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()

def save(win, grid, rows, width, im):
    cur_num = 0
    regex = re.compile(r'maze_(\d+).png')
    for file in os.listdir("."):
        m = regex.search(file)
        if m:
            num = int(m.group(1))
            if num > cur_num:
                cur_num = num
    cur_num += 1
    im.save("maze_input_" + str(cur_num) + ".png")
    backup = copy_grid(grid)
    for x in range(len(backup)):
        for y in range(len(backup[0])):
            g = backup[x][y]
            if g.is_closed() or g.is_open():
                g.reset()
    global DRAW_GRID
    DRAW_GRID = False
    draw(win, backup, rows, width)
    pygame.image.save(win, "maze_sol_" + str(cur_num) + ".png")
    for x in range(len(backup)):
        for y in range(len(backup[0])):
            g = backup[x][y]
            if g.is_path():
                g.reset()
    draw(win, backup, rows, width)
    pygame.image.save(win, "maze_" + str(cur_num) + ".png")
    DRAW_GRID = True
    draw(win, grid, rows, width)
    return grid

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

def fill_grid_from_img(img, grid, rows):
    for x in range(rows):
        for y in range(rows):
            try:
                if img.getpixel((x, y)) <= 200:
                    (grid[x][y]).make_barrier()
            except:
                pass

def fill_full_image(img):
    grid = make_grid(img.size[0], WIDTH)
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            try:
                if img.getpixel((x, y)) <= 200:
                    (grid[x][y]).make_barrier()
            except:
                pass
    return (grid, img.size[0])

def main(win, width):
    ROWS = 100

    grid = None
    argc = len(sys.argv) - 1
    im = None
    grid = make_grid(ROWS, width)
    if argc >= 1:
        # load from image
        im = Image.open(sys.argv[1]).convert("L")
        if im.size[0] == 100 and im.size[1] == 100:
            ROWS = im.size[0]
        else:
            im.thumbnail((ROWS,ROWS))
        fill_grid_from_img(im, grid, ROWS)
    else:
        # generate maze
        maze = Maze(ROWS, ROWS, 0, 0)
        maze.make_maze()
        im = maze.write_img(ROWS)
        (grid, ROWS) = fill_full_image(im)

    start = None
    end = None

    run = True
    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]: # LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start()

                elif not end and spot != start:
                    end = spot
                    end.make_end()

                elif spot != end and spot != start:
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]: # RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
                if event.key == pygame.K_f:
                    global QUICK
                    QUICK = not QUICK
                if event.key == pygame.K_g:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
                    maze = Maze(ROWS, ROWS, 0, 0)
                    maze.make_maze()
                    im = maze.write_img(ROWS)
                    fill_grid_from_img(im, grid, ROWS)
                if event.key == pygame.K_s:
                    grid = save(win, grid, ROWS, width, im)


    pygame.quit()

main(WIN, WIDTH)
