# Planisuss: Ecosystem Simulation

A dynamic simulation inspired by *Wa-Tor* and *Conway's Game of Life*, developed to model predator-prey interactions in a constrained environment.

This project demonstrates **Object-Oriented Programming (OOP)** principles, managing complex entity states, energy cycles, and spatial interactions within a grid-based topology.

## Simulation Logic
The world is populated by three distinct biological entities interacting in real-time:
* **Vegetob (Resources):** Grows on compatible terrain, providing energy.
* **Erbast (Herbivores):** Move in herds, consume Vegetob, and flee from predators. They have energy levels and lifespan.
* **Carviz (Carnivores):** Apex predators that hunt Erbast. They form prides and must eat to survive.

### Key Mechanics
* **Movement & Pathfinding:** Entities scan their surroundings to make decisions (feed, flee, or reproduce).
* **Energy System:** A thermodynamic-like system where energy is consumed per turn and gained through eating. Zero energy = Death.
* **Evolutionary Pressure:** Includes "Culling Phases" to manage overpopulation and random genetic mutations during reproduction.

## 🛠️ Technologies & Libraries
* **Language:** Python 3.x
* **Paradigm:** Object-Oriented Programming (OOP)
* **Visualization:** `Matplotlib` (Real-time animation of the grid and population statistics - NumPy for grid management).
* **Persistence:** Implemented `Pickle` for Save/Load functionality of the simulation state.