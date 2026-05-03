import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Caesar

    "Camion équipé d’un système d’artillerie"
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import torch
    from torch import nn
    import numpy as np
    import matplotlib.pyplot as plt

    return nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We attempt to shoot a target at a distance $d$ which is at the same height as our cannon. We control the initial velocity $v > 0$ of the shell as well as the elevation angle $\theta \in \left]0, \pi/2 \right[$ of the barrel. We'd like to reach the target with the lowest possible amount of energy (i.e. lowest initial shell velocity).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analytic Solution
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    \frac{d^2}{dt^2} \begin{bmatrix}x \\ y \end{bmatrix}
    = - \begin{bmatrix} 0 \\ g\end{bmatrix}, \qquad
    \begin{bmatrix}x \\ y \end{bmatrix} (0)
    = \begin{bmatrix} 0 \\ 0\end{bmatrix}, \qquad
    \frac{d}{dt}\begin{bmatrix}x \\ y \end{bmatrix} (0)
    = \begin{bmatrix}x \\ y \end{bmatrix} (0)
    = \begin{bmatrix} v \cos \theta \\  v \sin \theta \end{bmatrix}
    $$

    $$
    \begin{bmatrix}x \\ y \end{bmatrix}(t)
    = - \frac{1}{2} \begin{bmatrix} 0 \\ g \end{bmatrix}t^2 +
    \begin{bmatrix} v \cos \theta \\  v \sin \theta \end{bmatrix} t
    $$

    We get $y(t) = 0$ when $t = 0$ or
    $t = (2 v \sin \theta) / g$ which leads to
    $$x = (2 v^2 \sin \theta \cos \theta)/ g = v^2 \frac{\sin 2\theta}{g}.$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The maximum of the function
    $$
    \theta \in [0, \pi/2] \mapsto \sin 2\theta
    $$
    is $1$ when $\theta = \pi/2$.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    theta = np.linspace(0.0, np.pi / 2, 1000)
    plt.plot(theta, np.sin(2*theta), label=r"$\sin 2\theta$")
    ticks = [0, np.pi / 4, np.pi / 2]
    labels = ["$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"]
    plt.gca().set_xticks(ticks)
    plt.gca().set_xticklabels(labels, fontsize=13)
    plt.grid(True)
    plt.legend()
    mo.center(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hence we can reach a target at a distance $d$ with a minimal initial velocity with
    $$
    \theta = \frac{\pi}{4},
    \quad
    v = \sqrt{g d}.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reinforcement Learning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Model with free parameters $\lambda \in \mathbb{R}^2$.

    $$
    \theta = \frac{\arctan \lambda_1}{2} + \frac{\pi}{4}
    $$

    $$
    v = \exp \lambda_2
    $$
    """)
    return


@app.cell
def _(nn, torch):
    class Shooter(nn.Module):
        def __init__(self):
            super().__init__()
            self.nn = nn.Sequential(
                nn.Linear(1, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 2),
            )

        def forward(self, input):
            input = input.unsqueeze(-1)
            output = self.nn(input)
            theta = torch.arctan(output[..., 0]) / 2 + torch.pi / 4
            v = torch.exp(output[..., 1])
            output = torch.stack([theta, v], dim=-1)
            return output

    return (Shooter,)


@app.cell
def _(Shooter, torch):
    torch.manual_seed(42)
    _shooter = Shooter()
    input = torch.tensor([1.0, 2.0, 3.0])
    with torch.no_grad():
        output = _shooter(input)
    output
    return


@app.cell
def _(torch):
    g = 10.0
    def loss_fn(d, theta_v):
        theta = theta_v[..., 0]
        v = theta_v[..., 1]
        x = v * v * torch.sin(2 * theta) / g
        return (d - x).square().mean()

    return (loss_fn,)


@app.cell
def _(Shooter, loss_fn, torch):
    def train():
        shooter = Shooter()
        distance = torch.rand(1000) * 100.0 + 50.0 # random distance data in the [50, 150] range
        optimizer = torch.optim.Adam(shooter.parameters(), lr=1e-3)
        n_epochs = 1000
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            theta_v = shooter(distance)
            loss = loss_fn(distance, theta_v)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"epoch {epoch}, loss {loss.item():.4f}")
        return shooter

    shooter = train()
    return


if __name__ == "__main__":
    app.run()
