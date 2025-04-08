#include <SFML/Graphics.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define PARALLEL_FOR(flag, directive)                                          \
  if (flag) {                                                                  \
    directive;                                                                 \
  } else {                                                                     \
  }
#define PARALLEL_CRITICAL(flag)                                                \
  if (flag) {                                                                  \
    _Pragma("omp critical");                                                   \
  }

struct Particle {
  sf::Vector2f position;
  sf::Vector2f velocity;
  sf::Color color;
  float lambda;
  sf::Vector2f deltaQ;
};

class FluidSimulation {
public:
  FluidSimulation(int numParticles, float windowWidth, float windowHeight,
                  float radius, bool useParallel = true)
      : numParticles(numParticles), windowWidth(windowWidth),
        windowHeight(windowHeight), radius(radius), useParallel(useParallel) {
    particles.resize(numParticles);
    initializeParticles();

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
    poly6Coe = 315.0f / (64.0f * M_PI);
    spikyCoe = -45.0f / M_PI;
    sCorrDeltaQ = 0.3f * h;
    collisionIterations = 3;
    minDistThreshold = 0.1f;

    numParticleInGrid.resize(gridCols, std::vector<int>(gridRows, 0));
    tableGrid.resize(gridCols,
                     std::vector<std::vector<int>>(
                         gridRows, std::vector<int>(maxParticleInGrid, -1)));
    numNeighbour.resize(numParticles, 0);
    tableNeighbour.resize(numParticles, std::vector<int>(maxNeighbour, -1));
  }

  void update(float dt, const sf::Vector2f &inputForce) {
    auto start = std::chrono::high_resolution_clock::now();

    // Step 1: Apply forces
    const sf::Vector2f gravity(0.0f, 250.0f);

    sf::Vector2f totalForce = gravity;
    totalForce += inputForce;

    PARALLEL_FOR(useParallel, _Pragma("omp parallel for"))
    for (int i = 0; i < numParticles; ++i) {
      particles[i].velocity += totalForce * dt;
      particles[i].position += particles[i].velocity * dt;
    }
    auto tForces = std::chrono::high_resolution_clock::now();

    // Step 2: Neighbor search
    neighbourSearch();
    auto tNeighbour = std::chrono::high_resolution_clock::now();

    // Step 3: Solve constraints
    for (int iter = 0; iter < solveIterations; ++iter) {
      solveConstraints();
    }
    auto tConstraints = std::chrono::high_resolution_clock::now();

    // Step 4: Collision detection
    float minDist = std::numeric_limits<float>::max();
    PARALLEL_FOR(useParallel,
                 _Pragma("omp parallel for reduction(min : minDist)"))
    for (int p = 0; p < numParticles; ++p) {
      for (int i = 0; i < numNeighbour[p]; ++i) {
        int nbIndex = tableNeighbour[p][i];
        if (nbIndex <= p)
          continue;
        float distSq =
            (particles[p].position - particles[nbIndex].position).x *
                (particles[p].position - particles[nbIndex].position).x +
            (particles[p].position - particles[nbIndex].position).y *
                (particles[p].position - particles[nbIndex].position).y;
        float dist = std::sqrt(distSq);
        minDist = std::min(minDist, dist);
      }
    }
    auto tCollisionDetect = std::chrono::high_resolution_clock::now();

    // Step 5: Resolve collisions
    if (minDist < 10.0f - minDistThreshold ||
        minDist > 10.0f + minDistThreshold) {
      for (int iter = 0; iter < collisionIterations; ++iter) {
        resolveCollisions();
      }
    }
    auto tCollisionResolve = std::chrono::high_resolution_clock::now();

    // Step 6: Update velocities and apply boundary conditions
    PARALLEL_FOR(useParallel, _Pragma("omp parallel for"))
    for (int i = 0; i < numParticles; ++i) {
      sf::Vector2f oldPos = particles[i].position - particles[i].velocity * dt;
      particles[i].velocity = (particles[i].position - oldPos) / dt;
      boundaryCondition(particles[i]);
    }
    auto tVelocityUpdate = std::chrono::high_resolution_clock::now();

    // Log timings every 60 frames
    static int frame = 0;
    if (frame % 60 == 0) {
      auto forcesTime =
          std::chrono::duration_cast<std::chrono::microseconds>(tForces - start)
              .count();
      auto neighbourTime =
          std::chrono::duration_cast<std::chrono::microseconds>(tNeighbour -
                                                                tForces)
              .count();
      auto constraintsTime =
          std::chrono::duration_cast<std::chrono::microseconds>(tConstraints -
                                                                tNeighbour)
              .count();
      auto collisionDetectTime =
          std::chrono::duration_cast<std::chrono::microseconds>(
              tCollisionDetect - tConstraints)
              .count();
      auto collisionResolveTime =
          std::chrono::duration_cast<std::chrono::microseconds>(
              tCollisionResolve - tCollisionDetect)
              .count();
      auto velocityUpdateTime =
          std::chrono::duration_cast<std::chrono::microseconds>(
              tVelocityUpdate - tCollisionResolve)
              .count();
      auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(
                           tVelocityUpdate - start)
                           .count();

      std::cout << (useParallel ? "[Parallel] " : "[Single] ") << "Frame "
                << frame << " timings (us):\n";
      std::cout << "  Total=" << totalTime << "\n";
      std::cout << "  ApplyForces=" << forcesTime << "\n";
      std::cout << "  NeighbourSearch=" << neighbourTime << "\n";
      std::cout << "  SolveConstraints=" << constraintsTime << "\n";
      std::cout << "  CollisionDetect=" << collisionDetectTime << "\n";
      std::cout << "  CollisionResolve=" << collisionResolveTime << "\n";
      std::cout << "  VelocityUpdate=" << velocityUpdateTime << "\n";
      std::cout << "  MinDist=" << minDist << " (should be >= " << 2.0f * radius
                << ")\n";
    }
    frame++;
  }

  void render(sf::RenderWindow &window) {
    window.clear(sf::Color(233, 245, 243));
    sf::CircleShape particleShape(radius);
    particleShape.setOrigin(radius, radius);
    particleShape.setOutlineThickness(1.0f);
    // particleShape.setOutlineColor(sf::Color::Black);
    for (const auto &particle : particles) {
      particleShape.setPosition(particle.position);
      particleShape.setFillColor(particle.color);
      window.draw(particleShape);
    }
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

  float mass;
  float restDensity;
  float poly6Coe;
  float spikyCoe;
  float lambdaEpsilon;
  float sCorrDeltaQ;
  float sCorrK;
  int solveIterations;
  int collisionIterations;
  float minDistThreshold;

  std::vector<std::vector<int>> numParticleInGrid;
  std::vector<std::vector<std::vector<int>>> tableGrid;
  std::vector<int> numNeighbour;
  std::vector<std::vector<int>> tableNeighbour;
  std::vector<float> poly6Table;

  void initializeParticles() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);

    initKernels();

    for (int i = 0; i < numParticles; ++i) {
      float basePosX = 50.0f + 12.0f * (i % 30);
      float posX = basePosX + dis(gen);
      float posY = 50.0f + 12.0f * (i / 20);
      particles[i].position = sf::Vector2f(posX, posY);
      particles[i].velocity = sf::Vector2f(0.0f, 0.0f);
      particles[i].color = sf::Color(52, 235, 198);
      particles[i].lambda = 0.0f;
      particles[i].deltaQ = sf::Vector2f(0.0f, 0.0f);
    }
  }

  void boundaryCondition(Particle &particle) {
    float lowerX = radius;
    float upperX = windowWidth - radius;
    float lowerY = radius;
    float upperY = windowHeight - radius;
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
    pos.x = std::max(radius, std::min(windowWidth - radius, pos.x));
    pos.y = std::max(radius, std::min(windowHeight - radius, pos.y));
    int gridX = static_cast<int>(pos.x / gridSize);
    int gridY = static_cast<int>(pos.y / gridSize);
    return sf::Vector2i(gridX, gridY);
  }

  void neighbourSearch() {
    PARALLEL_FOR(useParallel, _Pragma("omp parallel for"))
    for (int i = 0; i < gridCols; ++i) {
      for (int j = 0; j < gridRows; ++j) {
        numParticleInGrid[i][j] = 0;
        for (int k = 0; k < maxParticleInGrid; ++k) {
          tableGrid[i][j][k] = -1;
        }
      }
    }

    PARALLEL_FOR(useParallel, _Pragma("omp parallel for"))
    for (int i = 0; i < numParticles; ++i) {
      numNeighbour[i] = 0;
      for (int j = 0; j < maxNeighbour; ++j) {
        tableNeighbour[i][j] = -1;
      }
    }

    for (int p = 0; p < numParticles; ++p) {
      sf::Vector2i gridPos = getGrid(particles[p].position);
      int gX = gridPos.x;
      int gY = gridPos.y;
      if (gX < 0 || gX >= gridCols || gY < 0 || gY >= gridRows) {
        std::cout << "Grid out of bounds for particle " << p << ": (" << gX
                  << ", " << gY << ")\n";
        continue;
      }
      PARALLEL_CRITICAL(useParallel) {
        int &count = numParticleInGrid[gX][gY];
        if (count >= maxParticleInGrid) {
          std::cout << "Grid cell (" << gX << ", " << gY
                    << ") overflow at particle " << p << "\n";
        } else {
          tableGrid[gX][gY][count] = p;
          count++;
        }
      }
    }

    PARALLEL_FOR(useParallel, _Pragma("omp parallel for"))
    for (int p = 0; p < numParticles; ++p) {
      sf::Vector2i gridPos = getGrid(particles[p].position);
      int gX = gridPos.x;
      int gY = gridPos.y;
      if (gX < 0 || gX >= gridCols || gY < 0 || gY >= gridRows)
        continue;
      for (int offX = -1; offX <= 1; ++offX) {
        for (int offY = -1; offY <= 1; ++offY) {
          int nbX = gX + offX;
          int nbY = gY + offY;
          if (nbX < 0 || nbX >= gridCols || nbY < 0 || nbY >= gridRows)
            continue;
          for (int i = 0; i < numParticleInGrid[nbX][nbY]; ++i) {
            int nbIndex = tableGrid[nbX][nbY][i];
            if (nbIndex == p)
              continue;
            float distSq =
                (particles[p].position - particles[nbIndex].position).x *
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

  float poly6(sf::Vector2f dist, float distSq) {
    float result = 0.0f;
    float d = std::sqrt(distSq);
    if (0.0f < d && d < h) {
      float rhs = (h2 - d * d);
      result = poly6Coe * rhs * rhs * rhs / h9;
    }
    return result;
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
      poly6Table[i] = 0.0f; // Explicitly set to 0 when d >= h
    }
  }
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
    if (d < 0.001f)
      d = 0.001f;
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

  void solveConstraints() {
    PARALLEL_FOR(useParallel, _Pragma("omp parallel for"))
    for (int p = 0; p < numParticles; ++p) {
      sf::Vector2f pos = particles[p].position;
      float density = 0.0f;
      sf::Vector2f spikySum(0.0f, 0.0f);
      float lowerSum = 0.0f;

      for (int i = 0; i < numNeighbour[p]; ++i) {
        int nbIndex = tableNeighbour[p][i];
        sf::Vector2f nbPos = particles[nbIndex].position;
        float distSq = (pos - nbPos).x * (pos - nbPos).x +
                       (pos - nbPos).y * (pos - nbPos).y;
        density += mass * poly6(pos - nbPos, distSq);
        sf::Vector2f s = spiky(pos - nbPos, distSq) / restDensity;
        spikySum += s;
        lowerSum += s.x * s.x + s.y * s.y;
      }

      float constraint = (density / restDensity) - 1.0f;
      lowerSum += spikySum.x * spikySum.x + spikySum.y * spikySum.y;
      particles[p].lambda = -constraint / (lowerSum + lambdaEpsilon);
    }

    PARALLEL_FOR(useParallel, _Pragma("omp parallel for"))
    for (int p = 0; p < numParticles; ++p) {
      sf::Vector2f deltaQ(0.0f, 0.0f);
      sf::Vector2f pos = particles[p].position;
      for (int i = 0; i < numNeighbour[p]; ++i) {
        int nbIndex = tableNeighbour[p][i];
        sf::Vector2f nbPos = particles[nbIndex].position;
        float distSq = (pos - nbPos).x * (pos - nbPos).x +
                       (pos - nbPos).y * (pos - nbPos).y;
        float scorr = sCorr(pos - nbPos, distSq);
        float left = particles[p].lambda + particles[nbIndex].lambda + scorr;
        sf::Vector2f right = spiky(pos - nbPos, distSq);
        deltaQ += left * right / restDensity;
      }
      particles[p].deltaQ = deltaQ;
    }

    PARALLEL_FOR(useParallel, _Pragma("omp parallel for"))
    for (int p = 0; p < numParticles; ++p) {
      particles[p].position += particles[p].deltaQ;
    }
  }

  void resolveCollisions() {
    const float energyPreservationOnCollision = 0.95f;
    const float artificialViscosity = 0.003f;

    PARALLEL_FOR(useParallel, _Pragma("omp parallel for"))
    for (int p = 0; p < numParticles; ++p) {
      sf::Vector2f pos = particles[p].position;
      for (int i = 0; i < numNeighbour[p]; ++i) {
        int nbIndex = tableNeighbour[p][i];
        if (nbIndex <= p)
          continue;
        sf::Vector2f nbPos = particles[nbIndex].position;
        float distSq = (pos - nbPos).x * (pos - nbPos).x +
                       (pos - nbPos).y * (pos - nbPos).y;
        float minDistSq = (2.0f * radius) * (2.0f * radius);
        if (distSq < minDistSq && distSq > 0.001f) {
          float dist = std::sqrt(distSq);
          float minDist = 2.0f * radius;
          float overlap = minDist - dist;
          sf::Vector2f correction = (pos - nbPos) / dist * overlap * 0.5f;
          PARALLEL_CRITICAL(useParallel) {
            particles[p].position += correction;
            particles[nbIndex].position -= correction;

            sf::Vector2f relVel =
                particles[p].velocity - particles[nbIndex].velocity;

            particles[p].velocity *= (1 - artificialViscosity);
            particles[p].velocity -=
                relVel * (1.0f - energyPreservationOnCollision) * 0.5f;
            particles[nbIndex].velocity +=
                relVel * (1.0f - energyPreservationOnCollision) * 0.5f;
          }
        }
      }
    }
  }
};

int main() {
  const float windowWidth = 500.0f;
  const float windowHeight = 500.0f;
  const int numParticles = 500;
  const float radius = 5.0f;

  sf::Vector2f inputForce(0.0f, 0.0f);
  const float forceMagnitude = 450.0f;

  sf::Clock clock;
  const float dt = 1.0f / 60.0f;

  std::cout << "Running parallel version...\n";
  FluidSimulation simParallel(numParticles, windowWidth, windowHeight, radius,
                              true);
  sf::RenderWindow windowParallel(sf::VideoMode(static_cast<int>(windowWidth),
                                                static_cast<int>(windowHeight)),
                                  "Particle System - Parallel");

  windowParallel.setKeyRepeatEnabled(false);

  while (windowParallel.isOpen()) {
    sf::Event event;
    while (windowParallel.pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        windowParallel.close();
      }
    }

    inputForce = sf::Vector2f(0.0f, 0.0f);
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
      inputForce = sf::Vector2f(-forceMagnitude, 0.0f);
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
      inputForce = sf::Vector2f(forceMagnitude, 0.0f);
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
      inputForce = sf::Vector2f(0.0f, -forceMagnitude);
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
      inputForce = sf::Vector2f(0.0f, forceMagnitude);

    simParallel.update(dt, inputForce);
    simParallel.render(windowParallel);
    windowParallel.display();

    sf::Time elapsed = clock.restart();
    if (elapsed.asSeconds() < dt) {
      sf::sleep(sf::seconds(dt - elapsed.asSeconds()));
    }
  }

  return 0;
}
