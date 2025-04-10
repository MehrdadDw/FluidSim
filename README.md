# Position-Based Dynamics
Basic Fluid Simulation(PBD) implementation using the SFML library in C++.

<div align="center">
  <img src="sim.gif" alt="Fluid Simulation" width="500">
</div>

## Prerequisites
- SFML library installed.
- A C++ compiler (e.g., g++).
- Make (for building the project).

0. prerequisites:
   ```bash
   # linux
   sudo apt install libsfml-dev
   
   # mac
   brew install sfml
   ```
Build and Run the simulation by:
   ```bash
mkdir build && cd build

cmake ..
cmake --build .

./fluid_sim

   ```


   W, S, A and D to apply force!

Used methods in **Paper: Position Based Fluids
By
Miles Macklin and Matthias Müller**



