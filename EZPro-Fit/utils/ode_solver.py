import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, Optional, Tuple


class ODEConfig:
    """Configuration for ODE solver"""

    def __init__(
        self,
        method: str = 'euler',
        num_steps: int = 100,
        adaptive: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-7,
    ):
        """
        Args:
            method: 'euler', 'heun', 'rk4', 'dopri45'
            num_steps: Number of integration steps (for non-adaptive methods)
            adaptive: Whether to use adaptive step size
            rtol: Relative tolerance for adaptive methods
            atol: Absolute tolerance for adaptive methods
        """
        self.method = method
        self.num_steps = num_steps
        self.adaptive = adaptive
        self.rtol = rtol
        self.atol = atol


class ODESolver:
    """Base class for ODE solvers"""

    def __init__(self, config: ODEConfig):
        self.config = config

    def solve(
        self,
        x0: torch.Tensor,
        t_span: Tuple[float, float],
        vfield: Callable,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve ODE from t_start to t_end

        Args:
            x0: Initial state [B, L, vocab]
            t_span: (t_start, t_end)
            vfield: Velocity field function v(x, t) -> [B, L, vocab]
            **kwargs: Additional arguments passed to vfield

        Returns:
            x_final: Final state [B, L, vocab]
        """
        raise NotImplementedError


class EulerSolver(ODESolver):
    """
    Euler's method: x_{n+1} = x_n + dt * v(x_n, t_n)

    Error: O(dt) - 1st order
    Stability: Stable for small dt
    Cost: 1 velocity evaluation per step
    """

    def solve(
        self,
        x0: torch.Tensor,
        t_span: Tuple[float, float],
        vfield: Callable,
        **kwargs
    ) -> torch.Tensor:
        device = x0.device
        t_start, t_end = t_span
        num_steps = self.config.num_steps
        dt = (t_end - t_start) / num_steps

        x = x0.clone()

        for i in range(num_steps):
            t = t_start + i * dt
            t_tensor = torch.full((x.shape[0],), t, device=device)

            # v = v_theta(x, t)
            v = vfield(x, t_tensor, **kwargs)

            # x = x + dt * v
            x = x + dt * v

        return x


class HeunSolver(ODESolver):
    """
    Heun's method (RK2): Improved Euler

    x_pred = x_n + dt * v(x_n, t_n)
    x_{n+1} = x_n + (dt/2) * (v(x_n, t_n) + v(x_pred, t_{n+1}))

    Error: O(dt^2) - 2nd order
    Stability: Improved over Euler
    Cost: 2 velocity evaluations per step
    """

    def solve(
        self,
        x0: torch.Tensor,
        t_span: Tuple[float, float],
        vfield: Callable,
        **kwargs
    ) -> torch.Tensor:
        device = x0.device
        t_start, t_end = t_span
        num_steps = self.config.num_steps
        dt = (t_end - t_start) / num_steps

        x = x0.clone()

        for i in range(num_steps):
            t = t_start + i * dt
            t_tensor = torch.full((x.shape[0],), t, device=device)
            t_next_tensor = torch.full((x.shape[0],), t + dt, device=device)

            # v1 = v(x_n, t_n)
            v1 = vfield(x, t_tensor, **kwargs)

            # x_pred = x + dt * v1
            x_pred = x + dt * v1

            # v2 = v(x_pred, t_{n+1})
            v2 = vfield(x_pred, t_next_tensor, **kwargs)

            # x = x + (dt/2) * (v1 + v2)
            x = x + (dt / 2) * (v1 + v2)

        return x


class RK4Solver(ODESolver):
    """
    Runge-Kutta 4th order method

    k1 = v(x_n, t_n)
    k2 = v(x_n + dt*k1/2, t_n + dt/2)
    k3 = v(x_n + dt*k2/2, t_n + dt/2)
    k4 = v(x_n + dt*k3, t_n + dt)
    x_{n+1} = x_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    Error: O(dt^4) - 4th order
    Stability: Excellent
    Cost: 4 velocity evaluations per step
    """

    def solve(
        self,
        x0: torch.Tensor,
        t_span: Tuple[float, float],
        vfield: Callable,
        **kwargs
    ) -> torch.Tensor:
        device = x0.device
        t_start, t_end = t_span
        num_steps = self.config.num_steps
        dt = (t_end - t_start) / num_steps

        x = x0.clone()

        for i in range(num_steps):
            t = t_start + i * dt
            t_tensor = torch.full((x.shape[0],), t, device=device)
            t_mid_tensor = torch.full((x.shape[0],), t + dt / 2, device=device)
            t_next_tensor = torch.full((x.shape[0],), t + dt, device=device)

            # k1
            k1 = vfield(x, t_tensor, **kwargs)
            x_mid1 = x + (dt / 2) * k1

            # k2
            k2 = vfield(x_mid1, t_mid_tensor, **kwargs)
            x_mid2 = x + (dt / 2) * k2

            # k3
            k3 = vfield(x_mid2, t_mid_tensor, **kwargs)
            x_end = x + dt * k3

            # k4
            k4 = vfield(x_end, t_next_tensor, **kwargs)

            # Update
            x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x


class DOPRI45Solver(ODESolver):
    """
    Dormand-Prince RK45 with adaptive step size control

    Error: O(dt^5) - 5th order
    Stability: Excellent, adaptive
    Cost: 6-7 velocity evaluations per step (adaptive)

    Use this for high accuracy requirements
    """

    def solve(
        self,
        x0: torch.Tensor,
        t_span: Tuple[float, float],
        vfield: Callable,
        **kwargs
    ) -> torch.Tensor:
        device = x0.device
        t_start, t_end = t_span

        # Coefficients for Dormand-Prince RK45
        c = torch.tensor([0, 1/5, 3/10, 4/5, 8/9, 1, 1], device=device)
        a = [
            torch.tensor([]),
            torch.tensor([1/5]),
            torch.tensor([3/40, 9/40]),
            torch.tensor([44/45, -56/15, 32/9]),
            torch.tensor([19372/6561, -25360/2187, 64448/6561, -212/729]),
            torch.tensor([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]),
            torch.tensor([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]),
        ]

        x = x0.clone()
        t = t_start
        dt_init = (t_end - t_start) / self.config.num_steps

        while t < t_end - 1e-8:
            dt = min(dt_init, t_end - t)

            # Evaluate at 7 stages
            k = []
            x_stages = []

            for i in range(len(c)):
                t_i = t + c[i] * dt
                x_i = x.clone()

                for j in range(i):
                    x_i = x_i + dt * a[i][j] * k[j]

                t_i_tensor = torch.full((x.shape[0],), t_i, device=device)
                k_i = vfield(x_i, t_i_tensor, **kwargs)
                k.append(k_i)
                x_stages.append(x_i)

            # 5th order solution
            b5 = torch.tensor(
                [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
                device=device
            )
            x_new = x + dt * sum(b5[i] * k[i] for i in range(len(k)))

            # 4th order solution for error estimation
            b4 = torch.tensor(
                [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40],
                device=device
            )
            x_old = x + dt * sum(b4[i] * k[i] for i in range(len(k)))

            # Error estimation
            error = torch.abs(x_new - x_old).max()
            tolerance = self.config.atol + self.config.rtol * torch.abs(x_new).max()

            if error < tolerance or dt < 1e-8:
                x = x_new
                t = t + dt
                dt_init = dt * 1.2  # Increase step size slightly
            else:
                dt_init = dt * 0.5  # Decrease step size

        return x


def get_solver(config: ODEConfig) -> ODESolver:
    """Factory function to get appropriate solver"""
    if config.method == 'euler':
        return EulerSolver(config)
    elif config.method == 'heun':
        return HeunSolver(config)
    elif config.method == 'rk4':
        return RK4Solver(config)
    elif config.method == 'dopri45':
        return DOPRI45Solver(config)
    else:
        raise ValueError(f"Unknown ODE solver: {config.method}")


# Example usage and benchmarking
def benchmark_solvers():
    """Benchmark different ODE solvers"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test function: dx/dt = -x (exponential decay)
    def v_field(x, t):
        return -x

    x0 = torch.ones(10, 100, 20, device=device)
    t_span = (0.0, 1.0)

    configs = [
        ODEConfig(method='euler', num_steps=100),
        ODEConfig(method='heun', num_steps=50),
        ODEConfig(method='rk4', num_steps=25),
    ]

    print("ODE Solver Benchmark")
    print("-" * 60)

    for config in configs:
        solver = get_solver(config)
        x_final = solver.solve(x0, t_span, v_field)
        # Expected: x_final ≈ exp(-1) ≈ 0.368
        error = torch.abs(x_final - np.exp(-1)).mean()
        print(f"{config.method.upper():8s} (steps={config.num_steps:3d}): error = {error:.6f}")
