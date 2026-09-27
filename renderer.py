import taichi as ti


class Renderer:
    def __init__(self, N: int, width: int = 1200, height: int = 800):
        self.N = N

        # Create window
        self.window = ti.ui.Window(
            "N-Body Gravity Simulation 3D", (width, height), vsync=True
        )
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()

        # Camera setup
        self.camera = ti.ui.Camera()
        self.camera.position(0, 0, 50)  # camera at z=50
        self.camera.lookat(0, 0, 0)  # looking at origin
        self.camera.up(0, 1, 0)  # y-axis is up
        self.camera.fov(60)  # 60 degree field of view
        self.camera.z_near(0.1)
        self.camera.z_far(1000)

        # Particle colors (will be set from masses)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=N)

    def set_colors_from_mass(self, masses):
        import numpy as np

        m = np.array(masses, dtype=np.float32)
        m_norm = (m - m.min()) / (m.max() - m.min() + 1e-9)

        colors = np.zeros((self.N, 3), dtype=np.float32)
        # Blue (light) → cyan → yellow → white (heavy)
        colors[:, 0] = m_norm  # Red channel increases
        colors[:, 1] = m_norm * 0.8 + 0.2  # Green stays high
        colors[:, 2] = 1.0 - m_norm * 0.7  # Blue decreases

        self.colors.from_numpy(colors)

    def render(self, positions):
        # Allow camera control: right mouse button to rotate, WASD to move
        self.camera.track_user_inputs(
            self.window, movement_speed=0.5, hold_key=ti.ui.RMB
        )
        self.scene.set_camera(self.camera)

        # Lighting
        self.scene.ambient_light((0.3, 0.3, 0.3))
        self.scene.point_light(pos=(10, 10, 10), color=(1, 1, 1))
        self.scene.point_light(pos=(-10, -10, -10), color=(0.5, 0.5, 0.8))

        # Draw particles with per-vertex colors
        self.scene.particles(positions, radius=0.15, per_vertex_color=self.colors)

        self.canvas.scene(self.scene)
        self.window.show()

    def running(self):
        return self.window.running
