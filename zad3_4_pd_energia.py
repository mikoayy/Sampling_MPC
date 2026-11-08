"""Zadania 3-4: sterownik PD i sterowanie energią + PD."""
from common import make_pendulum, run_episode
from controllers.energy import EnergyPDController
from controllers.pd import PDController

if __name__ == "__main__":
    env = make_pendulum(render=True)
    cases = [
        ("PD, start [0.1, 0.5]", PDController(env, kp=15.0, kd=5.0, goal=0.0), [0.1, 0.5]),
        ("PD, start [3.1, 0.0]", PDController(env, kp=15.0, kd=5.0, goal=0.0), [3.1, 0.0]),
        ("Energia + PD, start [3.1, 0.0]",
         EnergyPDController(env, k_energy=3.0, switch_angle=0.5), [3.1, 0.0]),
    ]
    for name, controller, x0 in cases:
        reward, _ = run_episode(env, controller, 200, init_state=x0)
        print(f"{name:32s} nagroda epizodu: {reward:9.2f}")
    env.close()
