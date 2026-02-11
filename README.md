Code relies on Sympy, re, script.integrate.solve_ivp. This project was meant as a Claude Sonnet 4.6 test, but after thorough inspection and testing it became clear that it provided a useful and effective tool for real-world use.

Allows for any ordinary- or partial differential equation to be converted from string to solution.
Ideal for rapid scientific or Machine Learning dataset construction.

AutoODE allows for the conversion of any order coupled ordinary differential equation via the Parse function, any order dimension agnostic differential equation (Lorenz-96 for example) via Parse_Local, and a function to solve the resulting equations. Also comes with parallelized versions of said functions by adding the prefix Parallel_.

AutoPDE allows for the conversion of any order coupled 1D partial differential equation via the Parse function, and a function to solve the resulting equations. Because it allows for coupled equations, it can also be used for vector-valued partial differential equation. Also comes with parallelized versions of said functions by adding the prefix Parallel_.

ODE example:

    >>> ode = AutoODE()
    >>> ode.Parse(
    ...     ["x1'' + 0.1*x1' + x1 - F(t) = 0"],
    ...     {"F": lambda t: np.sin(t)}
    ... )
    >>> sol = ode.Solve([1.0, 0.0], (0, 20), np.linspace(0, 20, 1000))

PDE example:

    >>> pde = AutoPDE()
    >>> pde.Parse("u1_t - 0.01*u1_xx = 0", {}, (0, 1), 100, bc='dirichlet')
    >>> u0 = np.sin(np.pi * pde.x_grid)
    >>> sol = pde.Solve(u0, (0, 0.5), np.linspace(0, 0.5, 100))


__main__ contains a Lorenz63 dataset generated via the AutoODE class, and the Double Slit Experiment generated via the AutoPDE class.

Result:

<img width="1400" height="500" alt="L63E" src="https://github.com/user-attachments/assets/ae1c71dc-966a-4258-b19f-63a047419664" />

<img width="1700" height="500" alt="DSE" src="https://github.com/user-attachments/assets/d5ba11b6-79fd-46cb-a949-7903ab082ff3" />


