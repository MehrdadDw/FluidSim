# Position-Based Dynamics
Basic Fluid Simulation(PBD) implementation using the SFML library in C++.


<video width="690" height="263" controls>
  <source src="output.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

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



