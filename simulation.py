"""3D N-body gravitational simulation using Taichi.

Physics: Newtonian gravity, velocity Verlet integration, softening.
All heavy computation runs in Taichi kernels for GPU parallelism.
"""

import taichi as ti
import numpy as np

@ti.data_oriented
class Simulation:
    """3D N-body gravitational simulation.

    Stores positions, velocities, accelerations, and masses as Taichi fields.
    Uses velocity Verlet integrator which preserves the symplectic structure
    of the equations of motion, keeping energy drift bounded [citation:7].
    """

    def __init__(self, N: int, G: float = 1.0, dt: float = 5e-4,
                 eps: float = 0.05):
        """Initialize simulation with N bodies.

        Args:
            N: Number of bodies.
            G: Gravitational constant (in simulation units).
            dt: Timestep size.
            eps: Softening parameter. Prevents infinite acceleration when
                 two bodies come very close. Added to r² in denominator.
        """
        self.N = N
        self.G = G
        self.dt = dt
        self.eps2 = eps * eps

        # --- Taichi fields ---
        # pos[i] = [x, y, z], vel[i] = [vx, vy, vz], acc[i] = [ax, ay, az]
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self.acc = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self.acc_old = ti.Vector.field(3, dtype=ti.f32, shape=N)
        self.mass = ti.field(dtype=ti.f32, shape=N)

    @ti.kernel
    def compute_acc(self):
        """Compute gravitational acceleration for all bodies.

        For each body i, sum contributions from all other bodies j:
            a[i] = Σⱼ G * m[j] * (pos[j] - pos[i]) / (|r|² + eps²)^(3/2)

        The (r² + eps²)^(3/2) denominator combines the r² from force
        magnitude and the r from vector normalization. Softening prevents
        divergence when bodies get close [citation:7].
        """
        for i in range(self.N):
            # Reset acceleration for body i
            self.acc[i] = ti.Vector([0.0, 0.0, 0.0])

            for j in range(self.N):
                if i != j:
                    r_vec = self.pos[j] - self.pos[i]
                    r2 = r_vec.norm_sqr() + self.eps2
                    r3 = r2 * ti.sqrt(r2)
                    # G * m[j] / r³ * r_vec  →  vector acceleration
                    self.acc[i] += self.G * self.mass[j] * r_vec / r3

    def step(self):
        """Advance simulation by one timestep using velocity Verlet.

        Algorithm:
          1. Save current acceleration: a_old = a
          2. Drift: pos += v * dt + 0.5 * a * dt²
          3. Recompute acceleration: a = compute_acc(pos)
          4. Kick: v += 0.5 * (a_old + a) * dt

        Velocity Verlet is symplectic, meaning it preserves the geometric
        structure of Hamiltonian dynamics. This keeps energy error bounded
        over long simulations, unlike Euler which drifts systematically [citation:7].
        """
        # Step 1: save old acceleration
        self.acc_old.copy_from(self.acc)

        # Step 2: drift (update positions)
        self.drift_kernel()

        # Step 3: recompute acceleration at new positions
        self.compute_acc()

        # Step 4: kick (update velocities)
        self.kick_kernel()

    @ti.kernel
    def drift_kernel(self):
        """Update positions: pos += v * dt + 0.5 * a * dt²."""
        for i in range(self.N):
            self.pos[i] += self.vel[i] * self.dt + \
                           0.5 * self.acc_old[i] * self.dt * self.dt

    @ti.kernel
    def kick_kernel(self):
        """Update velocities: v += 0.5 * (a_old + a_new) * dt."""
        for i in range(self.N):
            self.vel[i] += 0.5 * (self.acc_old[i] + self.acc[i]) * self.dt

    def set_bodies(self, positions, velocities, masses):
        """Set body data from numpy arrays.

        Args:
            positions: (N, 3) array of [x, y, z].
            velocities: (N, 3) array of [vx, vy, vz].
            masses: (N,) array of masses.
        """
        self.pos.from_numpy(positions.astype(np.float32))
        self.vel.from_numpy(velocities.astype(np.float32))
        self.mass.from_numpy(masses.astype(np.float32))
        # Initialize acceleration before first step
        self.compute_acc()

    def get_positions(self):
        """Return current positions as numpy array (N, 3)."""
        return self.pos.to_numpy()

    def get_velocities(self):
        """Return current velocities as numpy array (N, 3)."""
        return self.vel.to_numpy()

    def get_masses(self):
        """Return masses as numpy array (N,)."""
        return self.mass.to_numpy()