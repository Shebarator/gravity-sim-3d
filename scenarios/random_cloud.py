import numpy as np


def create(N: int):
    # Uniform distribution in a sphere of radius 30
    positions = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        # Rejection sampling for uniform sphere distribution
        while True:
            x, y, z = np.random.uniform(-30, 30, 3)
            if x * x + y * y + z * z <= 30 * 30:
                positions[i] = [x, y, z]
                break

    # Small random velocities (cold collapse)
    velocities = np.random.uniform(-0.5, 0.5, size=(N, 3)).astype(np.float32)

    # Random masses: most light, a few heavy
    masses = np.random.pareto(1.5, size=N).astype(np.float32) * 2.0 + 0.1

    return positions, velocities, masses
