import torch


def zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                         ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum = group['lr'], group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                assert g.ndim == 2, "Muon only supports 2D matrix params"
                state = self.state[p]
                buf = state.setdefault('momentum_buffer', torch.zeros_like(g))
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                u = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                if group['weight_decay'] != 0:
                    p.mul_(1 - lr * group['weight_decay'])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(u, alpha=-lr * scale)