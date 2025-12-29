"""PLANISUSS by Federica Brasca - mat. 513065

Final exam project for the course: 509477 - COMPUTER PROGRAMMING, ALGORITHMS AND DATA STRUCTURES - MOD. 1 - PROF. FERRARI STEFANO

This script implements a simulation of the fictitious world called "Planisuss", freely inspired by Wa-Tor and Conway's Game of Life.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.widgets as widgets
import pickle
from matplotlib.animation import FuncAnimation

print(f"Current Working Directory: {os.getcwd()}")

def safe_divide(numerator, denominator, default=0.0):
    """Performs division, returning a default value if the denominator is zero."""
    return numerator / denominator if denominator != 0 else default

class Config:
    """Global constants and settings for the simulation."""
    NUMDAYS = 1000
    NUMCELLS = 30
    WATER_PROBABILITY = 0.18
    EMPTY_GROUND_VAL = 0.0
    BASE_GROWING_RATE = 0.5
    Terrains = {
        "PLAIN": {
            "growth_modifier": 1.0,
            "color_empty": mcolors.to_rgb("#523518"),
            "color_full": mcolors.to_rgb("#00FF00")
        },
        "FOREST": {
            "spawn_prob": 0.018,
            "growth_modifier": 2.0,
            "color_empty": mcolors.to_rgb("#8A2BE2"),
            "color_full": mcolors.to_rgb("#FF00FF")
        }}
    MAX_VEGETOB_DENSITY = 100

    class Erbast:
        ID = 1
        MIN_ENERGY = 50
        MAX_ENERGY = 100
        MIN_LIFETIME = 80
        MAX_LIFETIME = 120

    class Carviz:
        ID = 2
        MIN_ENERGY = 50
        MAX_ENERGY = 120
        MIN_LIFETIME = 70
        MAX_LIFETIME = 100

    AGING = 1
    CELL_CAPACITY = 100
    INITIAL_ERBAST = 300
    INITIAL_CARVIZ = 80
    MOVE_ENERGY_COST = 1
    ESCAPE_DISTANCE = 3
    ESCAPE_ENERGY_COST = 3
    ERBAST_MIGRATION_THRESHOLD = 40
    ERBAST_HUNGER_THRESHOLD = 15
    VEGETOB_ENERGY_VALUE = 1

class Animal:
    """Represents a single animal in the simulation."""
    def __init__(self, animal_id: int, energy: float):
        self.id = animal_id
        self.age = 0
        if self.id == Config.Erbast.ID:
            species_config = Config.Erbast
        else:
            species_config = Config.Carviz
        self.min_energy = species_config.MIN_ENERGY
        self.max_energy = species_config.MAX_ENERGY
        self.lifetime = random.randint(species_config.MIN_LIFETIME, species_config.MAX_LIFETIME)
        self.energy = np.clip(energy, 0, self.max_energy)

    def gain_energy(self, amount: float):
        self.energy = min(self.max_energy, self.energy + amount)

    def consume_energy(self, amount: float):
        self.energy = max(0, self.energy - amount)

class Cell:
    """Represents a single unit on the grid."""
    def __init__(self, is_water: bool = False):
        self.is_water = is_water
        self.vegetob_density = 0
        self.animals = []
        self.terrain_type = "PLAIN"

    def set_vegetob(self, density: float):
        self.vegetob_density = np.clip(density, 0, Config.MAX_VEGETOB_DENSITY)

class Grid:
    """Represents the Planisuss world as a 2D grid of cells."""
    def __init__(self, size: int):
        self.size = size
        self.cells = [[Cell() for _ in range(size)] for _ in range(size)]

    def initialize_world(self, num_erbast: int, num_carviz: int):
        for i in range(self.size):
            for j in range(self.size):
                is_boundary = (i == 0 or j == 0 or i == self.size - 1 or j == self.size - 1)
                is_water = is_boundary or (random.random() < Config.WATER_PROBABILITY)
                self.cells[i][j] = Cell(is_water=is_water)
                if not is_water:
                    if random.random() < Config.Terrains['FOREST']['spawn_prob']:
                        self.cells[i][j].terrain_type = "FOREST"
                    else:
                        self.cells[i][j].terrain_type = "PLAIN"
                    self.cells[i][j].set_vegetob(random.randint(0, Config.MAX_VEGETOB_DENSITY))

        ground_coords = [(i, j) for i in range(self.size) for j in range(self.size) if not self.cells[i][j].is_water]
        if not ground_coords: return

        for _ in range(num_erbast):
            i, j = random.choice(ground_coords)
            self.cells[i][j].animals.append(Animal(Config.Erbast.ID, energy=random.randint(Config.Erbast.MIN_ENERGY, Config.Erbast.MAX_ENERGY)))
        
        for _ in range(num_carviz):
            i, j = random.choice(ground_coords)
            self.cells[i][j].animals.append(Animal(Config.Carviz.ID, energy=random.randint(Config.Carviz.MIN_ENERGY, Config.Carviz.MAX_ENERGY)))

    def get_neighbors(self, x: int, y: int):
        neighbors = []
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0: continue  # Skip the cell itself
                ni, nj = x + di, y + dj
                if 0 <= ni < self.size and 0 <= nj < self.size and not self.cells[ni][nj].is_water:
                    neighbors.append((ni, nj))
        return neighbors
    
    def find_distant_land_cells(self, x: int, y: int, distance: int):
        distant_cells = []
        for i in range(self.size):
            for j in range(self.size):
                if max(abs(i - x), abs(j - y)) == distance:
                    if not self.cells[i][j].is_water:
                        distant_cells.append((i, j))
        return distant_cells

    def __getitem__(self, pos):
        x, y = pos
        return self.cells[x][y]


# Simulation phases
def growing_phase(grid: Grid):
    for i in range(grid.size):
        for j in range(grid.size):
            cell = grid[i, j]
            if not cell.is_water:
                modifier = Config.Terrains[cell.terrain_type]['growth_modifier']
                growth_amount = Config.BASE_GROWING_RATE * modifier  # Growth influenced by terrain type
                cell.set_vegetob(cell.vegetob_density + growth_amount)

def aging_and_spawning_phase(grid: Grid):
    for i in range(grid.size):
        for j in range(grid.size):
            cell = grid[i, j]
            survivors, newborns = [], []
            for animal in cell.animals:
                animal.age += 1
                if animal.age % 10 == 0: animal.consume_energy(Config.AGING)
                if animal.energy <= 0: continue  # Animal dies of starvation and does not reproduce
                if animal.age >= animal.lifetime:
                    child_energy1 = (animal.energy / 2) + random.randint(0, 10)
                    child_energy2 = (animal.energy / 2) + random.randint(0, 10)
                    newborns.append(Animal(animal.id, child_energy1))
                    newborns.append(Animal(animal.id, child_energy2))
                    continue  # Parent dies after reproduction
                survivors.append(animal)
            cell.animals = survivors + newborns

def action_phase(grid: Grid):
    animal_actions = []
    for i in range(grid.size):
        for j in range(grid.size):
            if grid[i,j].animals:
                animal_actions.append(((i,j), grid[i,j].animals.copy()))
    for (x, y), animals_in_cell_snapshot in animal_actions:
        if grid[x,y].animals != animals_in_cell_snapshot: continue
        erbast_in_cell = [animal for animal in grid[x,y].animals if animal.id == Config.Erbast.ID]
        carviz_in_cell = [animal for animal in grid[x,y].animals if animal.id == Config.Carviz.ID]
        if erbast_in_cell:
            avg_energy = sum(a.energy for a in erbast_in_cell) / len(erbast_in_cell)
            if len(erbast_in_cell) > 1 and avg_energy >= Config.ERBAST_MIGRATION_THRESHOLD:
                for animal in erbast_in_cell: erbast_move_logic(animal, x, y, grid)
            elif avg_energy < Config.ERBAST_HUNGER_THRESHOLD:
                for animal in erbast_in_cell: erbast_move_logic(animal, x, y, grid)
        if carviz_in_cell:
            for animal in carviz_in_cell: 
                carviz_hunt_logic(animal, x, y, grid)

def erbast_move_logic(animal: Animal, x: int, y: int, grid: Grid):
    neighbors = grid.get_neighbors(x, y)
    if not neighbors: return
    best_pos, max_veg = (x, y), -1
    for ni, nj in neighbors:
        has_carviz = any(a.id == Config.Carviz.ID for a in grid[ni, nj].animals)
        if has_carviz: continue
        if grid[ni, nj].vegetob_density > max_veg:
            max_veg, best_pos = grid[ni, nj].vegetob_density, (ni, nj)
    if best_pos != (x, y):
        nx, ny = best_pos
        if animal in grid[x,y].animals:
             grid[x, y].animals.remove(animal)
             grid[nx, ny].animals.append(animal)
             animal.consume_energy(Config.MOVE_ENERGY_COST)

def carviz_hunt_logic(animal: Animal, x: int, y: int, grid: Grid):
    neighbors = grid.get_neighbors(x, y)
    if not neighbors: return
    prey_found_pos = None
    for ni, nj in neighbors:
        if any(a.id == Config.Erbast.ID for a in grid[ni,nj].animals):
            prey_found_pos = (ni, nj)
            break
    target_pos = prey_found_pos if prey_found_pos else random.choice(neighbors)
    nx, ny = target_pos
    if animal in grid[x, y].animals:
        grid[x, y].animals.remove(animal)
        grid[nx, ny].animals.append(animal)
        animal.consume_energy(Config.MOVE_ENERGY_COST)

def grazing_phase(grid: Grid):
    for i in range(grid.size):
        for j in range(grid.size):
            cell = grid[i, j]
            if cell.is_water or not cell.animals: continue
            erbast_in_cell = [a for a in cell.animals if a.id == Config.Erbast.ID]
            if not erbast_in_cell: continue
            erbast_in_cell.sort(key=lambda a: a.energy)
            for erbast in erbast_in_cell:
                if cell.vegetob_density > 0:
                    erbast.gain_energy(Config.VEGETOB_ENERGY_VALUE)
                    cell.set_vegetob(cell.vegetob_density - 1)
                else:
                    break

def struggle_phase(grid: Grid):
    for i in range(grid.size):
        for j in range(grid.size):
            cell = grid[i, j]
            if len(cell.animals) < 2: continue
            erbast_in_cell = [a for a in cell.animals if a.id == Config.Erbast.ID]
            carviz_in_cell = [a for a in cell.animals if a.id == Config.Carviz.ID]
            if not erbast_in_cell or not carviz_in_cell: continue
            escape_cells = grid.find_distant_land_cells(i, j, distance=Config.ESCAPE_DISTANCE)
            if escape_cells:
                destination_coord = random.choice(escape_cells)
                escape_percentage = random.random()
                num_to_escape = int(len(erbast_in_cell) * escape_percentage)
                erbast_in_cell.sort(key=lambda a: a.energy, reverse=True)
                escaping_animals, remaining_animals = [], []
                for animal in erbast_in_cell:
                    if len(escaping_animals) < num_to_escape and animal.energy > Config.ESCAPE_ENERGY_COST:
                        animal.consume_energy(Config.ESCAPE_ENERGY_COST)
                        escaping_animals.append(animal)
                    else:
                        remaining_animals.append(animal)
                if escaping_animals:
                    grid[destination_coord].animals.extend(escaping_animals)
                    erbast_in_cell = remaining_animals
            if not erbast_in_cell: continue
            erbast_energy = sum(a.energy for a in erbast_in_cell)
            carviz_energy = sum(a.energy for a in carviz_in_cell)
            win_prob = safe_divide(carviz_energy, carviz_energy + erbast_energy)
            if random.random() < win_prob:
                num_to_kill = min(len(carviz_in_cell), len(erbast_in_cell))
                erbast_in_cell.sort(key=lambda a: a.energy)
                killed_animals = erbast_in_cell[:num_to_kill]
                remaining_herd = erbast_in_cell[num_to_kill:]
                energy_gained = sum(a.energy for a in killed_animals)
                energy_per_carviz = safe_divide(energy_gained, len(carviz_in_cell))
                for carviz in carviz_in_cell:
                    carviz.gain_energy(energy_per_carviz)
                cell.animals = carviz_in_cell + remaining_herd

def culling_phase(grid: Grid):  # To handle overcrowding
    for i in range(grid.size):
        for j in range(grid.size):
            cell = grid[i, j]
            if len(cell.animals) > Config.CELL_CAPACITY:
                cell.animals.sort(key=lambda a: a.energy, reverse=True)
                cell.animals = cell.animals[:Config.CELL_CAPACITY]

class Simulator:
    def __init__(self):
        self.day = 0
        self.is_paused = True
        self.grid = None
        self.history = {}
        self.initial_erbast = Config.INITIAL_ERBAST
        self.initial_carviz = Config.INITIAL_CARVIZ
        self.initialize_simulation()

    def initialize_simulation(self):
        self.day = 0
        self.grid = Grid(Config.NUMCELLS)
        self.grid.initialize_world(self.initial_erbast, self.initial_carviz)
        self.history = {'days': [], 'erbast_pop': [], 'carviz_pop': [], 'avg_erbast_energy': [], 'avg_carviz_energy': []}
        self._update_stats()

    def run_one_day(self):
        if self.is_paused or self.day >= Config.NUMDAYS:
            return
        self.day += 1
        growing_phase(self.grid)
        aging_and_spawning_phase(self.grid)
        action_phase(self.grid)
        grazing_phase(self.grid)
        struggle_phase(self.grid)
        culling_phase(self.grid)
        self._update_stats()

    def _update_stats(self):
        stats = self.get_world_stats()
        self.history['days'].append(self.day)
        self.history['erbast_pop'].append(stats['erbast_pop'])
        self.history['carviz_pop'].append(stats['carviz_pop'])
        self.history['avg_erbast_energy'].append(stats['avg_erbast_energy'])
        self.history['avg_carviz_energy'].append(stats['avg_carviz_energy'])

    def get_world_stats(self):
        erbast_pop, carviz_pop = 0, 0
        total_erbast_energy, total_carviz_energy = 0.0, 0.0
        erbast_herds, carviz_prides = 0, 0
        total_vegetob_density, land_cells = 0, 0
        for i in range(self.grid.size):
            for j in range(self.grid.size):
                cell = self.grid[i, j]
                if not cell.is_water:
                    land_cells += 1
                    total_vegetob_density += cell.vegetob_density
                num_erbast_in_cell, num_carviz_in_cell = 0, 0
                for animal in cell.animals:
                    if animal.id == Config.Erbast.ID:
                        erbast_pop += 1
                        total_erbast_energy += animal.energy
                        num_erbast_in_cell += 1
                    else:
                        carviz_pop += 1
                        total_carviz_energy += animal.energy
                        num_carviz_in_cell += 1
                if num_erbast_in_cell > 0: erbast_herds += 1
                if num_carviz_in_cell > 0: carviz_prides += 1
        avg_erbast_energy = safe_divide(total_erbast_energy, erbast_pop)
        avg_carviz_energy = safe_divide(total_carviz_energy, carviz_pop)
        avg_vegetob_density = safe_divide(total_vegetob_density, land_cells)
        return {'erbast_pop': erbast_pop,
                'carviz_pop': carviz_pop,
                'avg_erbast_energy': avg_erbast_energy, 'avg_carviz_energy': avg_carviz_energy,
                'erbast_herds': erbast_herds, 'carviz_prides': carviz_prides,
                'avg_vegetob_density': avg_vegetob_density }

class Visualizer:
    def __init__(self, simulator: Simulator):
        self.simulator = simulator
        self.fig = plt.figure(figsize=(15, 8))
        
        self.fig.canvas.manager.set_window_title('Planisuss Simulation')

        gs = self.fig.add_gridspec(2, 2, width_ratios=[0.7, 0.3])
        self.ax_map = self.fig.add_subplot(gs[:, 0])  # Left side
        self.ax_density = self.fig.add_subplot(gs[0, 1])  # Top-right
        self.ax_energy = self.fig.add_subplot(gs[1, 1])  # Bottom-right
        self._setup_map_plot()
        self._setup_line_plots()
        self._create_widgets()
        self.animation = FuncAnimation(self.fig, self._update, interval=200, blit=False)
        self._update(0)

    def _setup_map_plot(self):
        self.ax_map.set_title("Planisuss World")
        self.ax_map.set_xticks([])
        self.ax_map.set_yticks([])
        self.land_map_image = self.ax_map.imshow(np.zeros((Config.NUMCELLS, Config.NUMCELLS, 3)), origin='lower')
        self.erbast_scatter = self.ax_map.scatter([], [], s=[], c="#2600FF", alpha=0.8, edgecolors='black', linewidth=0.5)
        self.carviz_scatter = self.ax_map.scatter([], [], s=[], c='#FF0000', alpha=0.8, edgecolors='black', linewidth=0.5)
        self.stats_text = self.ax_map.text(0.01, 0.99, '', transform=self.ax_map.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def _setup_line_plots(self):
        self.line_erbast_pop, = self.ax_density.plot([], [], color='green', label='Erbast Pop.')
        self.line_carviz_pop, = self.ax_density.plot([], [], color='red', label='Carviz Pop.')
        self.ax_density.set_title("Population Over Time")
        self.ax_density.set_xlabel("Days")
        self.ax_density.grid(True)
        self.ax_density.legend()
        self.line_erbast_energy, = self.ax_energy.plot([], [], color='green', label='Average Erbast Energy')
        self.line_carviz_energy, = self.ax_energy.plot([], [], color='red', label='Average Carviz Energy')
        self.ax_energy.set_title("Average Energy Over Time")
        self.ax_energy.set_xlabel("Days")
        self.ax_energy.grid(True)
        self.ax_energy.legend()

    def _create_widgets(self):
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.9, hspace=0.4)
        ax_erbast = plt.axes([0.1, 0.12, 0.5, 0.03])
        ax_carviz = plt.axes([0.1, 0.07, 0.5, 0.03])
        ax_pause = plt.axes([0.1, 0.01, 0.1, 0.04])
        ax_reset = plt.axes([0.21, 0.01, 0.1, 0.04])
        ax_save = plt.axes([0.32, 0.01, 0.1, 0.04])
        ax_load = plt.axes([0.43, 0.01, 0.1, 0.04])
        
        ax_erbast.set_facecolor('none')
        ax_carviz.set_facecolor('none')
        
        self.btn_pause = widgets.Button(ax_pause, 'Pause/Resume')
        self.btn_reset = widgets.Button(ax_reset, 'Reset')
        self.btn_save = widgets.Button(ax_save, 'Save')
        self.btn_load = widgets.Button(ax_load, 'Load')

        self.slider_erbast = widgets.Slider(ax=ax_erbast, label='Initial Erbast', valmin=0, valmax=1000, valinit=Config.INITIAL_ERBAST, valstep=10, color='green')
        self.slider_carviz = widgets.Slider(ax=ax_carviz, label='Initial Carviz', valmin=0, valmax=1000, valinit=Config.INITIAL_CARVIZ, valstep=10, color='red')
        
        self.slider_erbast.vline.set_visible(False)
        self.slider_carviz.vline.set_visible(False)
        
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_save.on_clicked(self._on_save)
        self.btn_load.on_clicked(self._on_load)

    def _on_pause(self, event):
        self.simulator.is_paused = not self.simulator.is_paused

    def _on_reset(self, event):
        self.simulator.is_paused = True
        self.simulator.initial_erbast = self.slider_erbast.val
        self.simulator.initial_carviz = self.slider_carviz.val
        self.simulator.initialize_simulation()
        self._update(0)
        plt.draw()
        
    def _on_save(self, event):
        self.simulator.is_paused = True
        save_filename = 'planisuss_save.pkl'
        try:
            with open(save_filename, 'wb') as f:
                pickle.dump(self.simulator, f)
            print(f"Simulation state saved successfully to {save_filename} in the current working directory: {os.getcwd()}.")
            self.fig.suptitle("Simulation saved! (Paused)", fontsize=16)
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error saving simulation: {e}")
        
    def _on_load(self, event):
        self.simulator.is_paused = True
        save_filename = 'planisuss_save.pkl'
        try:
            with open(save_filename, 'rb') as f:
                self.simulator = pickle.load(f)
            print(f"Simulation state loaded successfully from {save_filename}")
            self.simulator.is_paused = True 
            self.slider_erbast.set_val(self.simulator.initial_erbast)
            self.slider_carviz.set_val(self.simulator.initial_carviz)
            self._update(0)
            self.fig.suptitle(f"Simulation loaded! - Day: {self.simulator.day} (Paused)", fontsize=16)
            self.fig.canvas.draw_idle()

        except FileNotFoundError:
            print(f"Save file not found: {save_filename}")
            self.fig.suptitle("LOAD FAILED: Save file not found!", fontsize=16, color='red')
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error loading simulation: {e}")

    def _update_map_plot(self):
        background_image = np.zeros((Config.NUMCELLS, Config.NUMCELLS, 3))
        erbast_coords = {'x': [], 'y': [], 'size': []}
        carviz_coords = {'x': [], 'y': [], 'size': []}
        water_color = mcolors.to_rgb("#417FE4")
        grid = self.simulator.grid
        for i in range(grid.size):
            for j in range(grid.size):
                cell = grid[i, j]
                if cell.is_water:
                    background_image[i, j] = water_color
                else:
                    terrain_props = Config.Terrains[cell.terrain_type]
                    color_empty = np.array(terrain_props['color_empty'])
                    color_full = np.array(terrain_props['color_full'])
                    norm_density = cell.vegetob_density / Config.MAX_VEGETOB_DENSITY
                    final_color = color_empty * (1 - norm_density) + color_full * norm_density
                    background_image[i, j] = final_color
                    if cell.animals:
                        erbast_present = any(a.id == Config.Erbast.ID for a in cell.animals)
                        carviz_present = any(a.id == Config.Carviz.ID for a in cell.animals)
                        if erbast_present and not carviz_present:
                            erbast_coords['x'].append(j)
                            erbast_coords['y'].append(i)
                            erbast_coords['size'].append(sum(1 for a in cell.animals if a.id == Config.Erbast.ID) * 5)
                        elif carviz_present and not erbast_present:
                            carviz_coords['x'].append(j)
                            carviz_coords['y'].append(i)
                            carviz_coords['size'].append(sum(1 for a in cell.animals if a.id == Config.Carviz.ID) * 5)
                        elif erbast_present and carviz_present:
                            erbast_coords['x'].append(j - 0.1)
                            erbast_coords['y'].append(i - 0.1)
                            erbast_coords['size'].append(sum(1 for a in cell.animals if a.id == Config.Erbast.ID) * 5)
                            carviz_coords['x'].append(j + 0.1)
                            carviz_coords['y'].append(i + 0.1)
                            carviz_coords['size'].append(sum(1 for a in cell.animals if a.id == Config.Carviz.ID) * 5)
        self.land_map_image.set_data(background_image)
        self.erbast_scatter.set_offsets(np.c_[erbast_coords['x'], erbast_coords['y']])
        self.erbast_scatter.set_sizes(erbast_coords['size'])
        self.carviz_scatter.set_offsets(np.c_[carviz_coords['x'], carviz_coords['y']])
        self.carviz_scatter.set_sizes(carviz_coords['size'])

    def _update_line_plots(self):
        h = self.simulator.history
        days = h['days']
        if not days: return
        self.line_erbast_pop.set_data(days, h['erbast_pop'])
        self.line_carviz_pop.set_data(days, h['carviz_pop'])
        self.ax_density.relim()
        self.ax_density.autoscale_view()
        self.line_erbast_energy.set_data(days, h['avg_erbast_energy'])
        self.line_carviz_energy.set_data(days, h['avg_carviz_energy'])
        self.ax_energy.relim()
        self.ax_energy.autoscale_view()

    def _update(self, frame):
        if not self.simulator.is_paused:
            self.simulator.run_one_day()
        stats = self.simulator.get_world_stats()
        stats_string = (f"Day: {self.simulator.day}\n"
            f"Erbast: {stats['erbast_pop']}\n"f"Carviz: {stats['carviz_pop']}\n"
            f"Herds: {stats['erbast_herds']}\n"f"Prides: {stats['carviz_prides']}\n"
            f"Vegetob Density: {stats['avg_vegetob_density']:.1f}%")
        self.stats_text.set_text(stats_string)
        self._update_map_plot()
        self._update_line_plots()
        paused_status = "(Paused)" if self.simulator.is_paused else ""
        self.fig.suptitle(f"Planisuss World - Day: {self.simulator.day} {paused_status}", fontsize=16)
        return [self.land_map_image, self.erbast_scatter, self.carviz_scatter, self.stats_text, self.line_erbast_pop, self.line_carviz_pop, self.line_erbast_energy, self.line_carviz_energy]

    def run(self):
        plt.show()

if __name__ == '__main__':
    sim = Simulator()
    viz = Visualizer(sim)
    viz.run()