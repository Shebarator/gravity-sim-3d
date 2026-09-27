import taichi as ti
import numpy as np


@ti.data_oriented
class Simulation:
    def __init__(self, N: int, G: float = 1.0, dt: float = 5e-4, eps: float = 0.05):
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
            self.pos[i] += (
                self.vel[i] * self.dt + 0.5 * self.acc_old[i] * self.dt * self.dt
            )

    @ti.kernel
    def kick_kernel(self):
        """Update velocities: v += 0.5 * (a_old + a_new) * dt."""
        for i in range(self.N):
            self.vel[i] += 0.5 * (self.acc_old[i] + self.acc[i]) * self.dt

    def set_bodies(self, positions, velocities, masses):
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
