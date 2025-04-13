```markdown
<div align="center">
  <h1>Position-Based Fluid Simulation</h1>
  <p>A basic fluid simulation using Position-Based Dynamics (PBD) implemented with SFML in C++.</p>
  <img src="sim.gif" alt="Fluid Simulation Demo" width="500">
</div>

---

## ✨ Overview
This project implements a fluid simulation based on **Position-Based Dynamics (PBD)** using the SFML library in C++. It’s a lightweight and interactive demo inspired by the paper *Position Based Fluids* by Miles Macklin and Matthias Müller.

---

## 🛠️ Prerequisites
To run the simulation, ensure you have the following installed:
- **SFML library**
- **A C++ compiler** (e.g., g++)
- **Make** (for building the project)

### Installing SFML
```bash
# On Linux
sudo apt install libsfml-dev

# On macOS
brew install sfml
```

---

## 🚀 Build and Run
Follow these steps to build and run the simulation:

```bash
# Create and navigate to build directory
mkdir build && cd build

# Configure and build
cmake ..
cmake --build .

# Run the simulation
./fluid_sim
```

### Controls
- **W, A, S, D**: Apply forces to the fluid.
- **Spacebar**: Add more fluid particles.

---

## 📚 Reference
This project is based on the methods described in:

**Paper**: *Position Based Fluids*  
**Authors**: Miles Macklin and Matthias Müller  
**Slides**: [View Slides](https://mmacklin.com/pbf_slides.pdf)

---

<div align="center">
  <p>Built with 💧 and C++</p>
</div>
```
