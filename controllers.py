import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are, solve_continuous_lyapunov
from sim import U_MAX, DT

I_MAX = 5.0


class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
    def reset(self, B):
        self.ie = torch.zeros(B, 3)
    def __call__(self, r, vr, pm, vm, dt):
        e = r - pm; ve = vm - vr                    # velocity error w.r.t. reference velocity
        u_unsat = self.kp * e + self.ki * self.ie - self.kd * ve
        # anti-windup: only integrate when not saturated
        free = (u_unsat.abs() < U_MAX).float()
        self.ie = (self.ie + e * dt * free).clamp(-I_MAX, I_MAX)
        return self.kp * e + self.ki * self.ie - self.kd * ve


class LQG:
    """LQR with integral action + Kalman filter, designed per axis INCLUDING the measured velocity-loop lag."""
    def __init__(self, qe=10.0, qi=1.0, qv=1.0, qa=1e-3, r=0.1, sp=0.03, sv=0.08):
        from lqg_design import design, plant_taus
        self.qe, self.r = qe, r
        K, Ad, Be, L = design(plant_taus(), DT, qe, qi, qv, qa, r, sp, sv)
        t = lambda a: torch.tensor(a, dtype=torch.float32)
        self.K, self.Ad, self.Be, self.L = t(K), t(Ad), t(Be), t(L)
    def reset(self, B):
        self.xh = torch.zeros(B, 3, 3); self.ie = torch.zeros(B, 3); self.up = torch.zeros(B, 3)
    def __call__(self, r, vr, pm, vm, dt):
        pred = torch.einsum('aij,baj->bai', self.Ad, self.xh) + self.Be.unsqueeze(0) * self.up.unsqueeze(-1)
        y = torch.stack([pm, vm], -1)
        self.xh = pred + torch.einsum('aij,baj->bai', self.L, y - pred[..., :2])
        e = r - self.xh[..., 0]
        self.ie = (self.ie + e * dt).clamp(-I_MAX, I_MAX)
        K = self.K
        u = -(K[:, 0] * self.ie + K[:, 1] * e + K[:, 2] * (self.xh[..., 1] - vr) + K[:, 3] * self.xh[..., 2])
        u = u.clamp(-U_MAX, U_MAX); self.up = u
        return u


class SPNN(nn.Module):
    """Sigma-Pi network: linear terms + product (sigma-pi) terms + small hidden layer.
    Inputs per axis: e, v, integral(e). Weights are SHARED across x, y, z."""
    def __init__(self, hidden=16):
        super().__init__()
        self.lin = nn.Linear(3, 1)
        self.prod = nn.Linear(4, 1)          # e*v, e|e|, v|v|, e*ie
        self.hid = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        with torch.no_grad():                # sensible starting point; training changes everything
            self.lin.weight.copy_(torch.tensor([[2.0, -2.5, 0.4]])); self.lin.bias.zero_()
            self.prod.weight.zero_(); self.prod.bias.zero_()
            nn.init.normal_(self.hid[2].weight, std=0.01); self.hid[2].bias.zero_()
    def forward(self, e, v, ie):
        x = torch.cat([e, v, ie], -1)
        P = torch.cat([e * v, e * e.abs(), v * v.abs(), e * ie], -1)
        z = self.lin(x) + self.prod(P) + self.hid(x)
        return U_MAX * torch.tanh(z / U_MAX)


class SPNNController:
    def __init__(self, net):
        self.net = net
    def reset(self, B):
        self.ie = torch.zeros(B, 3)
    def __call__(self, r, vr, pm, vm, dt):
        e = r - pm; ve = vm - vr
        u = self.net(e.unsqueeze(-1), ve.unsqueeze(-1), self.ie.unsqueeze(-1)).squeeze(-1)
        free = (u.abs() < 0.95 * U_MAX).float()          # anti-windup, same as PID
        self.ie = (self.ie + e * dt * free).clamp(-I_MAX, I_MAX)
        return u


class Compensated:
    """Paper design (Jiang, Pourpanah, Hao 2020): base controller (PID or LQG) + ONLINE-ADAPTIVE sigma-pi
    compensator.   u = u_base - u_NN,   u_NN = W^T beta,   beta = kron(X1, X2, X3)
    Adaptation law (Lyapunov-derived):  dW/dt = -Gamma * (e^T P b) * beta - leak * W
    No offline training: W starts at 0 and learns during flight from the tracking error alone."""
    def __init__(self, base, gamma, leak=0.02, e_scale=2.0, v_scale=5.0, basis="sigma_pi"):
        self.base, self.gamma, self.leak, self.es, self.vs, self.basis = base, gamma, leak, e_scale, v_scale, basis
        self.nb = {"sigma_pi": 27, "additive": 7, "linear": 4, "bias": 1}[basis]      # number of basis functions per axis
        if isinstance(base, PID): kp, kd = float(base.kp), float(base.kd)
        else:                     kp, kd = abs(float(base.K[0, 1])), abs(float(base.K[0, 2]))   # LQG effective gains (x axis)
        A = np.array([[0.0, 1.0], [-kp, -kd]])
        P = solve_continuous_lyapunov(A.T, -np.eye(2))                               # A'P + PA = -I
        self.p12, self.p22 = float(P[0, 1]), float(P[1, 1])                          # e'Pb = p12*e + p22*e_dot
    def reset(self, B):
        self.base.reset(B); self.W = torch.zeros(B, 3, self.nb)
    def __call__(self, r, vr, pm, vm, dt):
        ub = self.base(r, vr, pm, vm, dt)
        e, ed = r - pm, vr - vm
        x = torch.stack([e / self.es, vm / self.vs, ub / U_MAX], -1).clamp(-1, 1)   # normalised inputs (B,3 axes,3)
        if self.basis == "sigma_pi":                                                # paper: products of [1, x, x^2] terms
            Bf = torch.stack([torch.ones_like(x), x, x ** 2], -1)
            beta = torch.einsum('bai,baj,bak->baijk', Bf[:, :, 0], Bf[:, :, 1], Bf[:, :, 2]).reshape(e.shape[0], 3, 27)
        else:                                                                       # ablation variants (no product terms)
            one = torch.ones_like(x[..., :1])
            beta = {"bias": one, "linear": torch.cat([one, x], -1), "additive": torch.cat([one, x, x ** 2], -1)}[self.basis]
        u_nn = (self.W * beta).sum(-1)
        s = self.p12 * e + self.p22 * ed
        self.W = (self.W + dt * (-self.gamma * s.unsqueeze(-1) * beta - self.leak * self.W)).clamp(-U_MAX, U_MAX)
        self.u_nn = u_nn                       # stored so scripts can plot what the SPNN is cancelling
        return ub - u_nn
