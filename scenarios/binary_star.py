"""Binary star scenario: two heavy bodies orbiting each other.

Two central stars at opposite ends of an orbit, with lighter bodies
around them. Creates chaotic dynamics with three-body interactions.
"""

import numpy as np


def create(N: int):
    """Create binary star system with N bodies.

    Returns:
        positions: (N, 3) array.
        velocities: (N, 3) array.
        masses: (N,) array.
    """
    # Two heavy stars
    m1, m2 = 3000.0, 3000.0
    # Total mass
    M = m1 + m2

    # Stars orbit around center of mass at origin
    separation = 15.0
    # Distance from center for each star (mass-weighted)
    r1 = separation * m2 / M
    r2 = separation * m1 / M

    # Stars in xz-plane, opposite sides
    positions = np.zeros((N, 3), dtype=np.float32)
    velocities = np.zeros((N, 3), dtype=np.float32)
    masses = np.zeros(N, dtype=np.float32)

    positions[0] = [-r1, 0, 0]
    positions[1] = [r2, 0, 0]
    masses[0] = m1
    masses[1] = m2

    # Orbital velocity for circular binary: v = sqrt(G * M / separation)
    # Relative velocity: v_rel = sqrt(G * M / separation)
    # Individual: v1 = v_rel * m2 / M, v2 = v_rel * m1 / M
    G = 1.0
    v_rel = np.sqrt(G * M / separation)
    v1 = v_rel * m2 / M
    v2 = v_rel * m1 / M

    velocities[0] = [0, v1, 0]   # star 1 moves +y
    velocities[1] = [0, -v2, 0]  # star 2 moves -y

    # Remaining bodies: random in a shell around the binary
    for i in range(2, N):
        # Random position in shell from 25 to 45
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        r = np.random.uniform(25.0, 45.0)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        positions[i] = [x, y, z]

        # Orbital velocity around total mass M
        r_mag = np.linalg.norm(positions[i])
        v_mag = np.sqrt(G * M / r_mag)
        r_hat = positions[i] / r_mag
        z_axis = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(r_hat, z_axis)) > 0.99:
            z_axis = np.array([1.0, 0.0, 0.0])
        tangent = np.cross(r_hat, z_axis)
        tangent = tangent / np.linalg.norm(tangent)
        velocities[i] = tangent * v_mag

        masses[i] = np.random.uniform(0.5, 3.0)

    return positions, velocities, masses