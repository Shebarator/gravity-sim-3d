"""Solar system scenario: central star + orbiting planets.

Creates one heavy central body and N-1 lighter bodies in circular orbits.
Orbital velocities are computed from Newtonian gravity for stability.
"""

import numpy as np


def create(N: int):
    """Create solar system with N bodies.

    Returns:
        positions: (N, 3) array.
        velocities: (N, 3) array.
        masses: (N,) array.
    """
    # Central star: 90% of total mass
    central_mass = 5000.0

    # Planet masses: random, much lighter
    planet_masses = np.random.uniform(0.1, 5.0, size=N - 1)

    # Positions: central at origin, planets in spherical shells
    positions = np.zeros((N, 3), dtype=np.float32)
    positions[0] = [0.0, 0.0, 0.0]

    # Planets: random directions, distances from 8 to 35
    for i in range(1, N):
        # Random unit vector on sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        r = np.random.uniform(8.0, 35.0)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        positions[i] = [x, y, z]

    # Velocities: orbital velocity for circular orbit
    # v = sqrt(G * M / r), direction perpendicular to radius
    velocities = np.zeros((N, 3), dtype=np.float32)
    G = 1.0

    for i in range(1, N):
        r = np.linalg.norm(positions[i])
        v_mag = np.sqrt(G * central_mass / r)

        # Perpendicular direction: cross product of radius and z-axis
        r_hat = positions[i] / r
        z_axis = np.array([0.0, 0.0, 1.0])
        # Handle case where r_hat is parallel to z-axis
        if abs(np.dot(r_hat, z_axis)) > 0.99:
            z_axis = np.array([1.0, 0.0, 0.0])
        tangent = np.cross(r_hat, z_axis)
        tangent = tangent / np.linalg.norm(tangent)

        velocities[i] = tangent * v_mag

    # Masses: central star + planets
    masses = np.zeros(N, dtype=np.float32)
    masses[0] = central_mass
    masses[1:] = planet_masses

    return positions, velocities, masses