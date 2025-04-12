#include <SFML/Graphics.hpp>
#include <imgui.h>
#include <imgui-SFML.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define PARALLEL_CRITICAL(flag) \
  if (flag) { _Pragma("omp critical"); }

struct Particle {
  sf::Vector2f position;
  sf::Vector2f velocity;
  sf::Color color;
  float lambda;
  sf::Vector2f deltaQ;
};

class FluidSimulation {
public:
  FluidSimulation(float windowWidth, float windowHeight, float radius, bool useParallel = true)
      : windowWidth(windowWidth), windowHeight(windowHeight), radius(radius), useParallel(useParallel) {
    numParticles = 0;
    h = 20.0f;
    h2 = h * h;
    h6 = h2 * h2 * h2;
    h9 = h6 * h2 * h;
    gridSize = h;
    gridRows = static_cast<int>(windowHeight / gridSize) + 1;
    gridCols = static_cast<int>(windowWidth / gridSize) + 1;
    maxParticleInGrid = 100;
    maxNeighbour = 160;

    mass = 1.0f;
    restDensity = 50.0f;
    lambdaEpsilon = 10.0f;
    sCorrK = 2.0f;
    solveIterations = 4;
    energyPreservationOnCollision = 0.95f;
    artificialViscosity = 0.001f;
    particleSizeFactor = 1.0f;

    poly6Coe = 315.0f / (64.0f * M_PI);
    spikyCoe = -45.0f / M_PI;
    sCorrDeltaQ = 0.3f * h;
    collisionIterations = 5;
    minDistThreshold = 1.0f;

    numParticleInGrid.resize(gridCols, std::vector<int>(gridRows, 0));
    tableGrid.resize(gridCols, std::vector<std::vector<int>>(
                                   gridRows, std::vector<int>(maxParticleInGrid, -1)));
    numNeighbour.resize(1000, 0);
    tableNeighbour.resize(1000, std::vector<int>(maxNeighbour, -1));
    initKernels();
  }

  int solveIterations;
  int collisionIterations;
  float energyPreservationOnCollision;
  float artificialViscosity;
  float particleSizeFactor;

  void dropParticles(int count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    sf::Vector2f center(windowWidth / 2.0f, radius * 2.0f);
    float circleRadius = 30.0f;
    int startIndex = numParticles;

    for (int i = 0; i < count; ++i) {
      float angle = (2.0f * M_PI * i) / std::max(1, count);
      float offsetX = circleRadius * std::cos(angle) + dis(gen);
      float offsetY = circleRadius * std::sin(angle) + dis(gen);
      Particle p;
      p.position = center + sf::Vector2f(offsetX, offsetY);
      p.velocity = sf::Vector2f(0.0f, 0.0f);
      p.color = sf::Color(52, 235, 198);
      p.lambda = 0.0f;
      p.deltaQ = sf::Vector2f(0.0f, 0.0f);
      particles.push_back(p);
    }
    numParticles += count;

    if (numParticles > numNeighbour.size()) {
      numNeighbour.resize(numParticles + 100, 0);
      tableNeighbour.resize(numParticles + 100, std::vector<int>(maxNeighbour, -1));
    }
  }

  void reset() {
    particles.clear();
    numParticles = 0;
    numNeighbour.resize(1000, 0);
    tableNeighbour.resize(1000, std::vector<int>(maxNeighbour, -1));
  }

  int getNumParticles() const {
    return numParticles;
  }

  void render(sf::RenderWindow &window, bool useColorFX) {
    window.clear(sf::Color(233, 245, 243));
    sf::CircleShape particleShape(radius * particleSizeFactor);
    particleShape.setOrigin(radius * particleSizeFactor, radius * particleSizeFactor);
    particleShape.setOutlineThickness(1.0f);

    const float minVelSquared = 0.0f;
    const float maxVelSquared = 50000.0f;
    const sf::Color fixedColor(52, 235, 198);

    for (const auto &particle : particles) {
      if (useColorFX) {
        float velSquared = particle.velocity.x * particle.velocity.x +
                           particle.velocity.y * particle.velocity.y;
        float t = (velSquared - minVelSquared) / (maxVelSquared - minVelSquared);
        t = std::max(0.0f, std::min(1.0f, t));

        sf::Uint8 red = static_cast<sf::Uint8>(255 * t);
        sf::Uint8 blue = static_cast<sf::Uint8>(255 * (1.0f - t));
        particleShape.setFillColor(sf::Color(red, 0, blue));
      } else {
        particleShape.setFillColor(fixedColor);
      }

      particleShape.setPosition(particle.position);
      window.draw(particleShape);
    }
  }

  void update(float dt, const sf::Vector2f &inputForce) {
    auto start = std::chrono::high_resolution_clock::now();
    if (numParticles == 0) return;

    const sf::Vector2f gravity(0.0f, 250.0f);
    sf::Vector2f totalForce = gravity + inputForce;

#ifdef _OPENMP
    if (useParallel) {
#pragma omp parallel for
      for (int i = 0; i < numParticles; ++i) {
        particles[i].velocity += totalForce * dt;
        particles[i].position += particles[i].velocity * dt;
      }
    } else
#endif
    {
      for (int i = 0; i < numParticles; ++i) {
        particles[i].velocity += totalForce * dt;
        particles[i].position += particles[i].velocity * dt;
      }
    }
    auto tForces = std::chrono::high_resolution_clock::now();

    neighbourSearch();
    auto tNeighbour = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < solveIterations; ++iter) {
      solveConstraints();
    }
    auto tConstraints = std::chrono::high_resolution_clock::now();

    float minDist = std::numeric_limits<float>::max();
#ifdef _OPENMP
    if (useParallel) {
#pragma omp parallel for reduction(min : minDist)
      for (int p = 0; p < numParticles; ++p) {
        for (int i = 0; i < numNeighbour[p]; ++i) {
          int nbIndex = tableNeighbour[p][i];
          if (nbIndex <= p) continue;
          float distSq = (particles[p].position - particles[nbIndex].position).x *
                             (particles[p].position - particles[nbIndex].position).x +
                         (particles[p].position - particles[nbIndex].position).y *
                             (particles[p].position - particles[nbIndex].position).y;
          float dist = std::sqrt(distSq);
          minDist = std::min(minDist, dist);
        }
      }
    } else
#endif
    {
      for (int p = 0; p < numParticles; ++p) {
        for (int i = 0; i < numNeighbour[p]; ++i) {
          int nbIndex = tableNeighbour[p][i];
          if (nbIndex <= p) continue;
          float distSq = (particles[p].position - particles[nbIndex].position).x *
                             (particles[p].position - particles[nbIndex].position).x +
                         (particles[p].position - particles[nbIndex].position).y *
                             (particles[p].position - particles[nbIndex].position).y;
          float dist = std::sqrt(distSq);
          minDist = std::min(minDist, dist);
        }
      }
    }
    auto tCollisionDetect = std::chrono::high_resolution_clock::now();

    if (minDist < 10.0f - minDistThreshold || minDist > 10.0f + minDistThreshold) {
      for (int iter = 0; iter < collisionIterations; ++iter) {
        resolveCollisions();
      }
    }
    auto tCollisionResolve = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
    if (useParallel) {
#pragma omp parallel for
      for (int i = 0; i < numParticles; ++i) {
        sf::Vector2f oldPos = particles[i].position - particles[i].velocity * dt;
        particles[i].velocity = (particles[i].position - oldPos) / dt;
        boundaryCondition(particles[i]);
      }
    } else
#endif
    {
      for (int i = 0; i < numParticles; ++i) {
        sf::Vector2f oldPos = particles[i].position - particles[i].velocity * dt;
        particles[i].velocity = (particles[i].position - oldPos) / dt;
        boundaryCondition(particles[i]);
      }
    }
    auto tVelocityUpdate = std::chrono::high_resolution_clock::now();

    static int frame = 0;
    if (frame % 60 == 0) {
      auto forcesTime = std::chrono::duration_cast<std::chrono::microseconds>(tForces - start).count();
      auto neighbourTime = std::chrono::duration_cast<std::chrono::microseconds>(tNeighbour - tForces).count();
      auto constraintsTime = std::chrono::duration_cast<std::chrono::microseconds>(tConstraints - tNeighbour).count();
      auto collisionDetectTime = std::chrono::duration_cast<std::chrono::microseconds>(tCollisionDetect - tConstraints).count();
      auto collisionResolveTime = std::chrono::duration_cast<std::chrono::microseconds>(tCollisionResolve - tCollisionDetect).count();
      auto velocityUpdateTime = std::chrono::duration_cast<std::chrono::microseconds>(tVelocityUpdate - tCollisionResolve).count();
      auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(tVelocityUpdate - start).count();

      std::cout << (useParallel ? "[Parallel] " : "[Single] ") << "Frame " << frame << " timings (us):\n";
      std::cout << "  Total=" << totalTime << "\n";
      std::cout << "  ApplyForces=" << forcesTime << "\n";
      std::cout << "  NeighbourSearch=" << neighbourTime << "\n";
      std::cout << "  SolveConstraints=" << constraintsTime << "\n";
      std::cout << "  CollisionDetect=" << collisionDetectTime << "\n";
      std::cout << "  CollisionResolve=" << collisionResolveTime << "\n";
      std::cout << "  VelocityUpdate=" << velocityUpdateTime << "\n";
      std::cout << "  MinDist=" << minDist << " (should be >= " << 2.0f * radius << ")\n";
    }
    frame++;
  }

private:
  int numParticles;
  float windowWidth, windowHeight, radius;
  bool useParallel;
  std::vector<Particle> particles;

  float h, h2, h6, h9;
  float gridSize;
  int gridRows, gridCols;
  int maxParticleInGrid;
  int maxNeighbour;
  float lambdaEpsilon;
  float sCorrK;

  float mass;
  float restDensity;
  float poly6Coe;
  float spikyCoe;
  float sCorrDeltaQ;
  float minDistThreshold;

  std::vector<std::vector<int>> numParticleInGrid;
  std::vector<std::vector<std::vector<int>>> tableGrid;
  std::vector<int> numNeighbour;
  std::vector<std::vector<int>> tableNeighbour;
  std::vector<float> poly6Table;

  void boundaryCondition(Particle &particle) {
    float lowerX = radius * particleSizeFactor;
    float upperX = windowWidth - radius * particleSizeFactor;
    float lowerY = radius * particleSizeFactor;
    float upperY = windowHeight - radius * particleSizeFactor;
    float damping = 0.4f;

    if (particle.position.x <= lowerX) {
      particle.position.x = lowerX;
      particle.velocity.x = -particle.velocity.x * damping;
    } else if (particle.position.x >= upperX) {
      particle.position.x = upperX;
      particle.velocity.x = -particle.velocity.x * damping;
    }

    if (particle.position.y <= lowerY) {
      particle.position.y = lowerY;
      particle.velocity.y = 0;
    } else if (particle.position.y >= upperY) {
      particle.position.y = upperY;
      particle.velocity.y = -particle.velocity.y * damping;
    }
  }

  sf::Vector2i getGrid(sf::Vector2f pos) {
    pos.x = std::max(radius * particleSizeFactor, std::min(windowWidth - radius * particleSizeFactor, pos.x));
    pos.y = std::max(radius * particleSizeFactor, std::min(windowHeight - radius * particleSizeFactor, pos.y));
    int gridX = static_cast<int>(pos.x / gridSize);
    int gridY = static_cast<int>(pos.y / gridSize);
    return sf::Vector2i(gridX, gridY);
  }

  void initKernels() {
    poly6Table.resize(1000);
    for (int i = 0; i < 1000; ++i) {
      float d = i * h / 999.0f;
      float d2 = d * d;
      if (d < h) {
        float rhs = (h2 - d2);
        poly6Table[i] = poly6Coe * rhs * rhs * rhs / h9;
      } else {
        poly6Table[i] = 0.0f;
      }
    }
  }

  float poly6(sf::Vector2f dist, float distSq) {
    float result = 0.0f;
    float d = std::sqrt(distSq);
    if (0.0f < d && d < h) {
      float rhs = (h2 - d * d);
      result = poly6Coe * rhs * rhs * rhs / h9;
    }
    return result;
  }

  float poly6Fast(float dist) {
    int idx = std::min(999, static_cast<int>(dist * 999.0f / h));
    return poly6Table[idx];
  }

  float poly6Scalar(float dist) {
    float result = 0.0f;
    float d = dist;
    if (0.0f < d && d < h) {
      float rhs = (h2 - d * d);
      result = poly6Coe * rhs * rhs * rhs / h9;
    }
    return result;
  }

  sf::Vector2f spiky(sf::Vector2f dist, float distSq) {
    sf::Vector2f result(0.0f, 0.0f);
    float d = std::sqrt(distSq);
    if (d < 0.001f) d = 0.001f;
    if (0.0f < d && d < h) {
      float m = (h - d) * (h - d);
      result = (spikyCoe * m / (h6 * d)) * dist;
    }
    return result;
  }

  float sCorr(sf::Vector2f dist, float distSq) {
    float upper = poly6(dist, distSq);
    float lower = poly6Scalar(sCorrDeltaQ);
    float m = upper / lower;
    return -sCorrK * m * m * m * m;
  }

  void neighbourSearch() {
#ifdef _OPENMP
    if (useParallel) {
#pragma omp parallel for collapse(2)
      for (int i = 0; i < gridCols; ++i) {
        for (int j = 0; j < gridRows; ++j) {
          numParticleInGrid[i][j] = 0;
          for (int k = 0; k < maxParticleInGrid; ++k) {
            tableGrid[i][j][k] = -1;
          }
        }
      }
    } else
#endif
    {
      for (int i = 0; i < gridCols; ++i) {
        for (int j = 0; j < gridRows; ++j) {
          numParticleInGrid[i][j] = 0;
          for (int k = 0; k < maxParticleInGrid; ++k) {
            tableGrid[i][j][k] = -1;
          }
        }
      }
    }

#ifdef _OPENMP
    if (useParallel) {
#pragma omp parallel for
      for (int i = 0; i < numParticles; ++i) {
        numNeighbour[i] = 0;
        for (int j = 0; j < maxNeighbour; ++j) {
          tableNeighbour[i][j] = -1;
        }
      }
    } else
#endif
    {
      for (int i = 0; i < numParticles; ++i) {
        numNeighbour[i] = 0;
        for (int j = 0; j < maxNeighbour; ++j) {
          tableNeighbour[i][j] = -1;
        }
      }
    }

    for (int p = 0; p < numParticles; ++p) {
      sf::Vector2i gridPos = getGrid(particles[p].position);
      int gX = gridPos.x;
      int gY = gridPos.y;
      if (gX < 0 || gX >= gridCols || gY < 0 || gY >= gridRows) {
        std::cout << "Grid out of bounds for particle " << p << ": (" << gX << ", " << gY << ")\n";
        continue;
      }
      PARALLEL_CRITICAL(useParallel) {
        int &count = numParticleInGrid[gX][gY];
        if (count >= maxParticleInGrid) {
          std::cout << "Grid cell (" << gX << ", " << gY << ") overflow at particle " << p << "\n";
        } else {
          tableGrid[gX][gY][count] = p;
          count++;
        }
      }
    }

#ifdef _OPENMP
    if (useParallel) {
#pragma omp parallel for
      for (int p = 0; p < numParticles; ++p) {
        sf::Vector2i gridPos = getGrid(particles[p].position);
        int gX = gridPos.x;
        int gY = gridPos.y;
        if (gX < 0 || gX >= gridCols || gY < 0 || gY >= gridRows) continue;
        for (int offX = -1; offX <= 1; ++offX) {
          for (int offY = -1; offY <= 1; ++offY) {
            int nbX = gX + offX;
            int nbY = gY + offY;
            if (nbX < 0 || nbX >= gridCols || nbY < 0 || nbY >= gridRows) continue;
            for (int i = 0; i < numParticleInGrid[nbX][nbY]; ++i) {
              int nbIndex = tableGrid[nbX][nbY][i];
              if (nbIndex == p) continue;
              float distSq = (particles[p].position - particles[nbIndex].position).x *
                                 (particles[p].position - particles[nbIndex].position).x +
                             (particles[p].position - particles[nbIndex].position).y *
                                 (particles[p].position - particles[nbIndex].position).y;
              if (distSq <= h2) {
                if (numNeighbour[p] >= maxNeighbour) {
                  std::cout << "Neighbor overflow for particle " << p << "\n";
                  break;
                }
                tableNeighbour[p][numNeighbour[p]] = nbIndex;
                numNeighbour[p]++;
              }
            }
          }
        }
      }
    } else
#endif
    {
      for (int p = 0; p < numParticles; ++p) {
        sf::Vector2i gridPos = getGrid(particles[p].position);
        int gX = gridPos.x;
        int gY = gridPos.y;
        if (gX < 0 || gX >= gridCols || gY < 0 || gY >= gridRows) continue;
        for (int offX = -1; offX <= 1; ++offX) {
          for (int offY = -1; offY <= 1; ++offY) {
            int nbX = gX + offX;
            int nbY = gY + offY;
            if (nbX < 0 || nbX >= gridCols || nbY < 0 || nbY >= gridRows) continue;
            for (int i = 0; i < numParticleInGrid[nbX][nbY]; ++i) {
              int nbIndex = tableGrid[nbX][nbY][i];
              if (nbIndex == p) continue;
              float distSq = (particles[p].position - particles[nbIndex].position).x *
                                 (particles[p].position - particles[nbIndex].position).x +
                             (particles[p].position - particles[nbIndex].position).y *
                                 (particles[p].position - particles[nbIndex].position).y;
              if (distSq <= h2) {
                if (numNeighbour[p] >= maxNeighbour) {
                  std::cout << "Neighbor overflow for particle " << p << "\n";
                  break;
                }
                tableNeighbour[p][numNeighbour[p]] = nbIndex;
                numNeighbour[p]++;
              }
            }
          }
        }
      }
    }
  }

  void solveConstraints() {
#ifdef _OPENMP
    if (useParallel) {
#pragma omp parallel for
      for (int p = 0; p < numParticles; ++p) {
        sf::Vector2f pos = particles[p].position;
        float density = 0.0f;
        sf::Vector2f spikySum(0.0f, 0.0f);
        float lowerSum = 0.0f;

        for (int i = 0; i < numNeighbour[p]; ++i) {
          int nbIndex = tableNeighbour[p][i];
          sf::Vector2f nbPos = particles[nbIndex].position;
          float distSq = (pos - nbPos).x * (pos - nbPos).x + (pos - nbPos).y * (pos - nbPos).y;
          density += mass * poly6(pos - nbPos, distSq);
          sf::Vector2f s = spiky(pos - nbPos, distSq) / restDensity;
          spikySum += s;
          lowerSum += s.x * s.x + s.y * s.y;
        }

        float constraint = (density / restDensity) - 1.0f;
        lowerSum += spikySum.x * spikySum.x + spikySum.y * spikySum.y;
        particles[p].lambda = -constraint / (lowerSum + lambdaEpsilon);
      }
    } else
#endif
    {
      for (int p = 0; p < numParticles; ++p) {
        sf::Vector2f pos = particles[p].position;
        float density = 0.0f;
        sf::Vector2f spikySum(0.0f, 0.0f);
        float lowerSum = 0.0f;

        for (int i = 0; i < numNeighbour[p]; ++i) {
          int nbIndex = tableNeighbour[p][i];
          sf::Vector2f nbPos = particles[nbIndex].position;
          float distSq = (pos - nbPos).x * (pos - nbPos).x + (pos - nbPos).y * (pos - nbPos).y;
          density += mass * poly6(pos - nbPos, distSq);
          sf::Vector2f s = spiky(pos - nbPos, distSq) / restDensity;
          spikySum += s;
          lowerSum += s.x * s.x + s.y * s.y;
        }

        float constraint = (density / restDensity) - 1.0f;
        lowerSum += spikySum.x * spikySum.x + spikySum.y * spikySum.y;
        particles[p].lambda = -constraint / (lowerSum + lambdaEpsilon);
      }
    }

#ifdef _OPENMP
    if (useParallel) {
#pragma omp parallel for
      for (int p = 0; p < numParticles; ++p) {
        sf::Vector2f deltaQ(0.0f, 0.0f);
        sf::Vector2f pos = particles[p].position;
        for (int i = 0; i < numNeighbour[p]; ++i) {
          int nbIndex = tableNeighbour[p][i];
          sf::Vector2f nbPos = particles[nbIndex].position;
          float distSq = (pos - nbPos).x * (pos - nbPos).x + (pos - nbPos).y * (pos - nbPos).y;
          float scorr = sCorr(pos - nbPos, distSq);
          float left = particles[p].lambda + particles[nbIndex].lambda + scorr;
          sf::Vector2f right = spiky(pos - nbPos, distSq);
          deltaQ += left * right / restDensity;
        }
        particles[p].deltaQ = deltaQ;
      }
    } else
#endif
    {
      for (int p = 0; p < numParticles; ++p) {
        sf::Vector2f deltaQ(0.0f, 0.0f);
        sf::Vector2f pos = particles[p].position;
        for (int i = 0; i < numNeighbour[p]; ++i) {
          int nbIndex = tableNeighbour[p][i];
          sf::Vector2f nbPos = particles[nbIndex].position;
          float distSq = (pos - nbPos).x * (pos - nbPos).x + (pos - nbPos).y * (pos - nbPos).y;
          float scorr = sCorr(pos - nbPos, distSq);
          float left = particles[p].lambda + particles[nbIndex].lambda + scorr;
          sf::Vector2f right = spiky(pos - nbPos, distSq);
          deltaQ += left * right / restDensity;
        }
        particles[p].deltaQ = deltaQ;
      }
    }

#ifdef _OPENMP
    if (useParallel) {
#pragma omp parallel for
      for (int p = 0; p < numParticles; ++p) {
        particles[p].position += particles[p].deltaQ;
      }
    } else
#endif
    {
      for (int p = 0; p < numParticles; ++p) {
        particles[p].position += particles[p].deltaQ;
      }
    }
  }

  void resolveCollisions() {
#ifdef _OPENMP
    if (useParallel) {
#pragma omp parallel for
      for (int p = 0; p < numParticles; ++p) {
        sf::Vector2f pos = particles[p].position;
        for (int i = 0; i < numNeighbour[p]; ++i) {
          int nbIndex = tableNeighbour[p][i];
          if (nbIndex <= p) continue;
          sf::Vector2f nbPos = particles[nbIndex].position;
          float distSq = (pos - nbPos).x * (pos - nbPos).x + (pos - nbPos).y * (pos - nbPos).y;
          float minDistSq = (2.0f * radius * particleSizeFactor) * (2.0f * radius * particleSizeFactor);
          if (distSq < minDistSq && distSq > 0.001f) {
            float dist = std::sqrt(distSq);
            float minDist = 2.0f * radius * particleSizeFactor;
            float overlap = minDist - dist;
            sf::Vector2f correction = (pos - nbPos) / dist * overlap * 0.5f;
            PARALLEL_CRITICAL(useParallel) {
              particles[p].position += correction;
              particles[nbIndex].position -= correction;
              sf::Vector2f relVel = particles[p].velocity - particles[nbIndex].velocity;
              particles[p].velocity *= (1 - artificialViscosity);
              particles[p].velocity -= relVel * (1.0f - energyPreservationOnCollision) * 0.5f;
              particles[nbIndex].velocity += relVel * (1.0f - energyPreservationOnCollision) * 0.5f;
            }
          }
        }
      }
    } else
#endif
    {
      for (int p = 0; p < numParticles; ++p) {
        sf::Vector2f pos = particles[p].position;
        for (int i = 0; i < numNeighbour[p]; ++i) {
          int nbIndex = tableNeighbour[p][i];
          if (nbIndex <= p) continue;
          sf::Vector2f nbPos = particles[nbIndex].position;
          float distSq = (pos - nbPos).x * (pos - nbPos).x + (pos - nbPos).y * (pos - nbPos).y;
          float minDistSq = (2.0f * radius * particleSizeFactor) * (2.0f * radius * particleSizeFactor);
          if (distSq < minDistSq && distSq > 0.001f) {
            float dist = std::sqrt(distSq);
            float minDist = 2.0f * radius * particleSizeFactor;
            float overlap = minDist - dist;
            sf::Vector2f correction = (pos - nbPos) / dist * overlap * 0.5f;
            particles[p].position += correction;
            particles[nbIndex].position -= correction;
            sf::Vector2f relVel = particles[p].velocity - particles[nbIndex].velocity;
            particles[p].velocity *= (1 - artificialViscosity);
            particles[p].velocity -= relVel * (1.0f - energyPreservationOnCollision) * 0.5f;
            particles[nbIndex].velocity += relVel * (1.0f - energyPreservationOnCollision) * 0.5f;
          }
        }
      }
    }
  }
};

int main() {
  const float windowWidth = 1920.0f/2.0f;  // Increased resolution
  const float windowHeight = 1080.0f/2.0f; // Increased resolution
  const float radius = 6.0f;  // Slightly larger particles for visibility

  sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Fluid Simulation");
  window.setKeyRepeatEnabled(false);

  if (!ImGui::SFML::Init(window)) {
    std::cerr << "Failed to initialize ImGui-SFML" << std::endl;
    return 1;
  }

  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.Colors[ImGuiCol_FrameBg] = ImVec4(0.3f, 0.3f, 0.7f, 0.54f);
  style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.4f, 0.4f, 0.8f, 0.40f);
  style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.5f, 0.5f, 0.9f, 0.67f);
  style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.2f, 0.2f, 1.0f, 1.00f);
  style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.3f, 0.3f, 1.0f, 1.00f);
  style.Colors[ImGuiCol_WindowBg] = ImVec4(0.15f, 0.15f, 0.4f, 0.9f);
  style.Colors[ImGuiCol_TitleBg] = ImVec4(0.15f, 0.15f, 0.5f, 1.0f);
  style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.25f, 0.25f, 0.6f, 1.0f);

  FluidSimulation sim(windowWidth, windowHeight, radius, true);
  sf::Clock clock, deltaClock;
  const float dt = 1.0f / 60.0f;
  sf::Vector2f inputForce(0.0f, 0.0f);
  const float forceMagnitude = 450.0f;
  int numParticlesToDrop = 50;
  bool useColorFX = true;

  // FPS calculation variables
  float fpsElapsedTime = 0.0f;
  int fpsFrameCount = 0;
  float displayedFPS = 60.0f;

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      ImGui::SFML::ProcessEvent(window, event);
      if (event.type == sf::Event::Closed) window.close();
      if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Space) {
        int count = numParticlesToDrop <= 0 ? 1 : numParticlesToDrop;
        sim.dropParticles(count);
      }
    }

    sf::Time deltaTime = deltaClock.restart();

    // Update FPS calculation every 0.5 seconds
    fpsElapsedTime += deltaTime.asSeconds();
    fpsFrameCount++;
    if (fpsElapsedTime >= 0.5f) {
      displayedFPS = fpsElapsedTime > 0.0f ? static_cast<float>(fpsFrameCount) / fpsElapsedTime : 0.0f;
      fpsElapsedTime = 0.0f;
      fpsFrameCount = 0;
    }

    ImGui::SFML::Update(window, deltaTime);

    ImGui::Begin("Simulation Parameters");
    ImGui::PushItemWidth(150.0f);
    ImGui::SliderInt("Solve Iterations", &sim.solveIterations, 1, 10);
    ImGui::SliderInt("Collision Iterations", &sim.collisionIterations, 1, 10);
    ImGui::SliderFloat("Energy Preservation", &sim.energyPreservationOnCollision, 0.0f, 1.0f);
    ImGui::SliderFloat("Artificial Viscosity", &sim.artificialViscosity, 0.0f, 0.01f);
    ImGui::SliderFloat("Particle Size", &sim.particleSizeFactor, 0.6667f, 3.0f);
    ImGui::PopItemWidth();
    ImGui::Checkbox("Color FX", &useColorFX);
    ImGui::Separator();
    ImGui::PushItemWidth(100.0f);
    ImGui::InputInt("Number", &numParticlesToDrop);
    ImGui::SameLine();
    if (ImGui::Button("Drop")) {
      int count = numParticlesToDrop <= 0 ? 1 : numParticlesToDrop;
      sim.dropParticles(count);
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
      sim.reset();
    }
    ImGui::SameLine();
    ImGui::Text("FPS: %.1f Particles: %d", displayedFPS, sim.getNumParticles());
    ImGui::PopItemWidth();
    if (numParticlesToDrop < 0) numParticlesToDrop = 0;
    ImGui::End();

    inputForce = sf::Vector2f(0.0f, 0.0f);
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) inputForce.x = -forceMagnitude;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) inputForce.x = forceMagnitude;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) inputForce.y = -forceMagnitude;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) inputForce.y = forceMagnitude;

    sim.update(dt, inputForce);
    sim.render(window, useColorFX);
    ImGui::SFML::Render(window);
    window.display();

    sf::Time elapsed = clock.restart();
    if (elapsed.asSeconds() < dt) sf::sleep(sf::seconds(dt - elapsed.asSeconds()));
  }

  ImGui::SFML::Shutdown();
  return 0;
}