# 🌌 Gravity Sim 3D

3D N-body gravitational simulation with Taichi. Up to 1000 bodies, velocity Verlet integrator, interactive 3D camera.

## Что внутри

- **3D** — позиции `(x, y, z)`, камера вращается мышью
- **N ≤ 1000** — GPU-параллелизация через Taichi
- **Velocity Verlet** — симпатический интегратор, сохраняет энергию [citation:7]
- **Без границ** — тела могут улетать, камера следует за сценой
- **Softening** — сглаживание сингулярности при сближении [citation:7]

## Установка

```bash
pip install -r requirements.txt