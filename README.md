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
g++ -std=c++11 fluid.cpp -o fluid -lsfml-graphics -lsfml-window -lsfml-system

./fluid
   ```


   W, S, A and D to apply force! 

Used methods in **Paper: Position Based Fluids
By
Miles Macklin and Matthias Müller**



