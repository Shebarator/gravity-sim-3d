"""
Usage:
    python main.py                    # default: solar system, 500 bodies
    python main.py solar 800          # solar system with 800 bodies
    python main.py cloud 1000         # random cloud with 1000 bodies
    python main.py binary 600         # binary star with 600 bodies
"""

import sys
import taichi as ti
from simulation import Simulation
from renderer import Renderer


def load_scenario(name, N):
    if name == "solar":
        from scenarios.solar_system import create

        return create(N)
    elif name == "cloud":
        from scenarios.random_cloud import create

        return create(N)
    elif name == "binary":
        from scenarios.binary_star import create

        return create(N)
    else:
        raise ValueError(f"Unknown scenario: {name}")


def main():
    # Parse arguments
    scenario = sys.argv[1] if len(sys.argv) > 1 else "solar"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    # Initialize Taichi. Try GPU first; fall back to CPU.
    # Vulkan backend works on Windows with AMD/Intel GPUs [citation:8].
    try:
        ti.init(arch=ti.gpu, default_fp=ti.f32)
    except Exception:
        print("GPU backend unavailable, falling back to CPU.")
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    print(f"Initializing '{scenario}' with {N} bodies...")

    # Create scenario data
    positions, velocities, masses = load_scenario(scenario, N)

    # Build simulation and renderer
    sim = Simulation(N, G=1.0, dt=5e-4, eps=0.1)
    sim.set_bodies(positions, velocities, masses)

    renderer = Renderer(N)
    renderer.set_colors_from_mass(masses)

    print("Starting simulation. Controls:")
    print("  Right mouse button + drag: rotate camera")
    print("  W/A/S/D: move forward/left/back/right")
    print("  E/Q: move up/down")
    print("  Close window to exit.")

    # Main loop
    while renderer.running():
        # Physics steps per frame (more steps = faster simulation)
        for _ in range(4):
            sim.step()

        renderer.render(sim.pos)

    print("Simulation ended.")


if __name__ == "__main__":
    main()
