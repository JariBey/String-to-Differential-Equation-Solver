"""
Comprehensive Test Suite for StringToDiffEqSol.py

Tests all functionality in AutoODE and AutoPDE classes including:
- Basic parsing and solving
- Parallel parsing and solving
- Local/periodic systems
- Various boundary conditions
- Edge cases and error handling
"""

import numpy as np
import sys
from pathlib import Path

# Import the module to test
from StringToDiffEqSol import AutoODE, AutoPDE

# pytest will be imported only if available
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create a dummy pytest.raises for the custom runner
    class _DummyPytest:
        @staticmethod
        def raises(exc):
            class _RaisesContext:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {exc} but nothing was raised")
                    return issubclass(exc_type, exc)
            return _RaisesContext()
    pytest = _DummyPytest()


class TestAutoODEBasicParsing:
    """Test basic ODE parsing functionality."""
    
    def test_parse_simple_first_order(self):
        """Test parsing a simple first-order ODE."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse(
            "x1' - x1 = 0",
            {}
        )
        assert callable(ODE)
        assert state_vars == [('x1', 1)]
        assert order == 1
    
    def test_parse_simple_second_order(self):
        """Test parsing a simple second-order ODE."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse(
            "x1'' + x1 = 0",
            {}
        )
        assert callable(ODE)
        assert state_vars == [('x1', 2)]
        assert order == 2
    
    def test_parse_with_forcing_function(self):
        """Test parsing ODE with forcing function."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse(
            "x1'' + 0.1*x1' + x1 - F(t) = 0",
            {"F": lambda t: np.sin(t)}
        )
        assert callable(ODE)
        assert state_vars == [('x1', 2)]
        assert order == 2
    
    def test_parse_coupled_system(self):
        """Test parsing coupled ODE system."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse(
            ["x1' - x2 = 0", "x2' + x1 = 0"],
            {}
        )
        assert callable(ODE)
        assert len(state_vars) == 2
        assert order == 2
    
    def test_parse_lorenz_system(self):
        """Test parsing the Lorenz 63 system."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse(
            [
                "x1' - s(t)*(x2 - x1) = 0",
                "x2' - x1*r(t) + x1*x3 + x2 = 0",
                "x3' - x1*x2 + b(t)*x3 = 0",
            ],
            {
                "s": lambda t: 10.0,
                "r": lambda t: 28.0,
                "b": lambda t: 8/3,
            }
        )
        assert callable(ODE)
        assert len(state_vars) == 3
        assert order == 3


class TestAutoODESolving:
    """Test ODE solving functionality."""
    
    def test_solve_harmonic_oscillator(self):
        """Test solving simple harmonic oscillator."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse("x1'' + x1 = 0", {})
        
        # Initial conditions: x1(0) = 1, x1'(0) = 0
        ic = [1.0, 0.0]
        t_span = (0, 2*np.pi)
        t_eval = np.linspace(0, 2*np.pi, 100)
        
        sol = ode.Solve(ODE, ic, t_span, t_eval)
        
        # Solution should be approximately cos(t)
        assert sol.success
        assert len(sol.t) == 100
        assert np.allclose(sol.y[0, 0], 1.0, atol=1e-6)
        assert np.allclose(sol.y[0, -1], 1.0, atol=1e-3)
    
    def test_solve_exponential_decay(self):
        """Test solving exponential decay."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse("x1' + x1 = 0", {})
        
        ic = [1.0]
        t_span = (0, 5)
        t_eval = np.linspace(0, 5, 100)
        
        sol = ode.Solve(ODE, ic, t_span, t_eval)
        
        # Solution should be exp(-t)
        expected = np.exp(-t_eval)
        assert np.allclose(sol.y[0], expected, atol=1e-3)
    
    def test_solve_with_forcing(self):
        """Test solving ODE with time-dependent forcing."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse(
            "x1' + x1 - F(t) = 0",
            {"F": lambda t: 1.0}
        )
        
        ic = [0.0]
        t_span = (0, 5)
        t_eval = np.linspace(0, 5, 100)
        
        sol = ode.Solve(ODE, ic, t_span, t_eval)
        
        # Solution should approach 1.0
        assert sol.y[0, -1] > 0.99


class TestAutoODEParseLocal:
    """Test local/periodic ODE parsing."""
    
    def test_parse_local_periodic(self):
        """Test parsing local system with periodic boundaries."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse_Local(
            "x_i' - (x_{i+1} - x_{i-1}) = 0",
            {},
            dim=10,
            mod=True
        )
        assert callable(ODE)
        assert len(state_vars) == 10
        assert order == 10
    
    def test_parse_local_fixed(self):
        """Test parsing local system with fixed boundaries."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse_Local(
            "x_i' + x_i = 0",
            {},
            dim=5,
            mod=False
        )
        assert callable(ODE)
        assert len(state_vars) == 5
        assert order == 5
    
    def test_lorenz96(self):
        """Test parsing Lorenz 96 system."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse_Local(
            "x_i' - (x_{i+1} - x_{i-2})*x_{i-1} + x_i - F(t) = 0",
            {"F": lambda t: 8.0},
            dim=40,
            mod=True
        )
        assert callable(ODE)
        assert len(state_vars) == 40
        assert order == 40


class TestAutoODEParallel:
    """Test parallel parsing and solving."""
    
    def test_parallel_parse_basic(self):
        """Test parallel parsing of multiple systems."""
        ode = AutoODE()
        inputs = [
            (["x1' - x1 = 0"], {}),
            (["x1'' + x1 = 0"], {}),
            (["x1' + 2*x1 = 0"], {}),
        ]
        
        odes, state_vars_list, orders = ode.Parallel_Parse(inputs, n_jobs=2, backend="threading")
        
        assert len(odes) == 3
        assert len(state_vars_list) == 3
        assert len(orders) == 3
        assert all(callable(o) for o in odes)
        assert orders[0] == 1
        assert orders[1] == 2
        assert orders[2] == 1
    
    def test_parallel_parse_local(self):
        """Test parallel parsing of local systems."""
        ode = AutoODE()
        inputs = [
            ("x_i' + x_i = 0", {}, 10),
            ("x_i' + x_i = 0", {}, 20),
            ("x_i' + x_i = 0", {}, 30),
        ]
        
        odes, state_vars_list, orders = ode.Parallel_Parse_Local(inputs, n_jobs=2, backend="threading")
        
        assert len(odes) == 3
        assert orders[0] == 10
        assert orders[1] == 20
        assert orders[2] == 30
    
    def test_parallel_solve(self):
        """Test parallel solving of multiple systems."""
        ode = AutoODE()
        
        # Parse the same system multiple times
        inputs = [
            (["x1' - x1 = 0"], {}),
            (["x1' - x1 = 0"], {}),
            (["x1' - x1 = 0"], {}),
        ]
        
        odes, state_vars_list, orders = ode.Parallel_Parse(inputs, n_jobs=2, backend="threading")
        
        # Solve with different initial conditions
        t_span = (0, 2)
        t_eval = np.linspace(0, 2, 50)
        solve_inputs = [
            ([1.0], t_span, t_eval),
            ([2.0], t_span, t_eval),
            ([3.0], t_span, t_eval),
        ]
        
        solutions = ode.Parallel_Solve(odes, solve_inputs, n_jobs=2, backend="threading")
        
        assert len(solutions) == 3
        assert all(sol.success for sol in solutions)
        # Check that solutions have different final values
        assert not np.allclose(solutions[0].y[0, -1], solutions[1].y[0, -1])


class TestAutoPDEBasicParsing:
    """Test basic PDE parsing functionality."""
    
    def test_parse_heat_equation(self):
        """Test parsing heat equation."""
        pde = AutoPDE()
        ODE, field_vars, order = pde.Parse(
            "u1_t - 0.01*u1_xx = 0",
            {},
            (0, 1), 100, bc='periodic'
        )
        assert callable(ODE)
        assert field_vars == ['u1']
        assert order == 100
        assert pde.N == 100
    
    def test_parse_burgers_equation(self):
        """Test parsing Burgers equation."""
        pde = AutoPDE()
        ODE, field_vars, order = pde.Parse(
            "u1_t + u1*u1_x - 0.01*u1_xx = 0",
            {},
            (0, 1), 50, bc='periodic'
        )
        assert callable(ODE)
        assert field_vars == ['u1']
        assert order == 50
    
    def test_parse_coupled_pde(self):
        """Test parsing coupled PDE system."""
        pde = AutoPDE()
        ODE, field_vars, order = pde.Parse(
            ["u1_t - 0.1*u1_xx + u1*u2 = 0",
             "u2_t - 0.2*u2_xx - u1*u2 = 0"],
            {},
            (0, 1), 100, bc='periodic'
        )
        assert callable(ODE)
        assert len(field_vars) == 2
        assert order == 200  # 100 points * 2 fields
    
    def test_parse_dirichlet_bc(self):
        """Test parsing with Dirichlet boundary conditions."""
        pde = AutoPDE()
        ODE, field_vars, order = pde.Parse(
            "u1_t - 0.01*u1_xx = 0",
            {},
            (0, 1), 100, bc='dirichlet'
        )
        assert callable(ODE)
        assert pde.N == 100
    
    def test_parse_neumann_bc(self):
        """Test parsing with Neumann boundary conditions."""
        pde = AutoPDE()
        ODE, field_vars, order = pde.Parse(
            "u1_t - 0.01*u1_xx = 0",
            {},
            (0, 1), 100, bc='neumann'
        )
        assert callable(ODE)
        assert pde.N == 100


class TestAutoPDESolving:
    """Test PDE solving functionality."""
    
    def test_solve_heat_equation(self):
        """Test solving heat equation."""
        pde = AutoPDE()
        ODE, field_vars, order = pde.Parse(
            "u1_t - 0.01*u1_xx = 0",
            {},
            (0, 1), 100, bc='periodic'
        )
        
        # Initial condition: sin wave
        x = pde.x_grid
        u0 = np.sin(2*np.pi*x)
        
        t_span = (0, 0.1)
        t_eval = np.linspace(0, 0.1, 20)
        
        sol = pde.Solve(ODE, u0, t_span, t_eval)
        
        assert sol.success
        # Heat should diffuse, so amplitude should decrease
        u_final = pde.get_field(sol, 'u1')[:, -1]
        assert np.max(np.abs(u_final)) < np.max(np.abs(u0))
    
    def test_solve_wave_propagation(self):
        """Test solving wave equation (as coupled first-order system)."""
        pde = AutoPDE()
        alpha = 0.01
        ODE, field_vars, order = pde.Parse(
            [f"u1_t + {alpha}*u2_xx = 0",
             f"u2_t - {alpha}*u1_xx = 0"],
            {},
            (-10, 10), 200, bc='dirichlet'
        )
        
        x = pde.x_grid
        u1_0 = np.exp(-x**2)
        u2_0 = np.zeros_like(x)
        ic = np.concatenate([u1_0, u2_0])
        
        t_span = (0, 10)
        t_eval = np.linspace(0, 10, 50)
        
        sol = pde.Solve(ODE, ic, t_span, t_eval)
        
        assert sol.success
        assert sol.y.shape == (order, 50)
    
    def test_get_field(self):
        """Test extracting fields from solution."""
        pde = AutoPDE()
        ODE, field_vars, order = pde.Parse(
            ["u1_t - 0.01*u1_xx = 0",
             "u2_t - 0.02*u2_xx = 0"],
            {},
            (0, 1), 50, bc='periodic'
        )
        
        x = pde.x_grid
        u1_0 = np.sin(2*np.pi*x)
        u2_0 = np.cos(2*np.pi*x)
        ic = np.concatenate([u1_0, u2_0])
        
        t_span = (0, 0.1)
        t_eval = np.linspace(0, 0.1, 10)
        
        sol = pde.Solve(ODE, ic, t_span, t_eval)
        
        u1_field = pde.get_field(sol, 'u1')
        u2_field = pde.get_field(sol, 'u2')
        
        assert u1_field.shape == (50, 10)
        assert u2_field.shape == (50, 10)


class TestAutoPDEParallel:
    """Test parallel PDE parsing and solving."""
    
    def test_parallel_parse(self):
        """Test parallel parsing of multiple PDE systems."""
        inputs = [
            ("u1_t - 0.01*u1_xx = 0", {}, (0, 1), 50, 'periodic'),
            ("u1_t - 0.02*u1_xx = 0", {}, (0, 1), 60, 'periodic'),
            ("u1_t - 0.03*u1_xx = 0", {}, (0, 1), 70, 'periodic'),
        ]
        
        odes, field_vars_list, orders = AutoPDE.Parallel_Parse(inputs, n_jobs=2, backend="threading")
        
        assert len(odes) == 3
        assert len(field_vars_list) == 3
        assert len(orders) == 3
        assert orders[0] == 50
        assert orders[1] == 60
        assert orders[2] == 70
    
    def test_parallel_solve(self):
        """Test parallel solving of multiple PDE systems."""
        # Create PDE instances
        pdes = []
        odes = []
        parse_inputs = [
            ("u1_t - 0.01*u1_xx = 0", {}, (0, 1), 50, 'periodic'),
            ("u1_t - 0.01*u1_xx = 0", {}, (0, 1), 50, 'periodic'),
        ]
        
        for inp in parse_inputs:
            pde = AutoPDE()
            ode, fv, ord = pde.Parse(*inp)
            pdes.append(pde)
            odes.append(ode)
        
        # Different initial conditions
        x = pdes[0].x_grid
        u0_1 = np.sin(2*np.pi*x)
        u0_2 = np.sin(4*np.pi*x)
        
        t_span = (0, 0.1)
        t_eval = np.linspace(0, 0.1, 20)
        
        solve_inputs = [
            (u0_1, t_span, t_eval),
            (u0_2, t_span, t_eval),
        ]
        
        solutions = AutoPDE.Parallel_Solve(pdes, odes, solve_inputs, n_jobs=2, backend="threading")
        
        assert len(solutions) == 2
        assert all(sol.success for sol in solutions)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_parse_invalid_equation(self):
        """Test that invalid equation raises error."""
        ode = AutoODE()
        with pytest.raises(ValueError):
            # No derivative to solve for
            ode.Parse("x1 + 1 = 0", {})
    
    def test_solve_without_parse(self):
        """Test that solving without parsing raises error."""
        pde = AutoPDE()
        with pytest.raises(RuntimeError):
            pde.Solve(None, [1.0], (0, 1), [0, 1])
    
    def test_wrong_initial_condition_size(self):
        """Test that wrong IC size raises error."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse("x1' - x1 = 0", {})
        
        with pytest.raises((ValueError, AssertionError)):
            # Wrong size IC
            ode.Solve(ODE, [1.0, 2.0], (0, 1), [0, 1])
    
    def test_high_order_derivatives(self):
        """Test parsing higher-order derivatives."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse(
            "x1'''' + x1 = 0",
            {}
        )
        assert order == 4  # 4th order equation


class TestNumericalAccuracy:
    """Test numerical accuracy of solutions."""
    
    def test_harmonic_oscillator_accuracy(self):
        """Test numerical accuracy for harmonic oscillator."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse("x1'' + x1 = 0", {})
        
        ic = [1.0, 0.0]
        t_span = (0, 10)
        t_eval = np.linspace(0, 10, 1000)
        
        sol = ode.Solve(ODE, ic, t_span, t_eval, method='DOP853', rtol=1e-10, atol=1e-12)
        
        # Analytical solution: cos(t)
        expected = np.cos(t_eval)
        error = np.max(np.abs(sol.y[0] - expected))
        
        assert error < 1e-8
    
    def test_heat_equation_conservation(self):
        """Test that heat equation conserves total heat (for periodic BC)."""
        pde = AutoPDE()
        ODE, field_vars, order = pde.Parse(
            "u1_t - 0.01*u1_xx = 0",
            {},
            (0, 1), 100, bc='periodic'
        )
        
        x = pde.x_grid
        u0 = np.sin(2*np.pi*x) + 2.0  # Constant + wave
        
        t_span = (0, 0.5)
        t_eval = np.linspace(0, 0.5, 50)
        
        sol = pde.Solve(ODE, u0, t_span, t_eval, method='BDF', rtol=1e-8)
        
        # Total heat should be conserved (up to numerical error)
        initial_total = np.sum(u0)
        final_total = np.sum(pde.get_field(sol, 'u1')[:, -1])
        
        assert np.abs(initial_total - final_total) < 1e-6


class TestComplexSystems:
    """Test more complex realistic systems."""
    
    def test_lorenz_attractor(self):
        """Test solving Lorenz attractor."""
        ode = AutoODE()
        ODE, state_vars, order = ode.Parse(
            [
                "x1' - 10*(x2 - x1) = 0",
                "x2' - x1*(28 - x3) + x2 = 0",
                "x3' - x1*x2 + (8/3)*x3 = 0",
            ],
            {}
        )
        
        ic = [1.0, 1.0, 1.0]
        t_span = (0, 25)
        t_eval = np.linspace(0, 25, 2000)
        
        sol = ode.Solve(ODE, ic, t_span, t_eval, method='RK45')
        
        assert sol.success
        # Check that the solution exhibits chaotic behavior (varies significantly)
        variance = np.var(sol.y, axis=1)
        assert all(v > 1.0 for v in variance)
    
    def test_reaction_diffusion(self):
        """Test solving reaction-diffusion system."""
        pde = AutoPDE()
        ODE, field_vars, order = pde.Parse(
            [
                "u1_t - 0.1*u1_xx - u1 + u1*u1*u2 = 0",
                "u2_t - 0.05*u2_xx + u1 - u1*u1*u2 = 0",
            ],
            {},
            (0, 10), 200, bc='neumann'
        )
        
        x = pde.x_grid
        # Small perturbation around steady state
        u1_0 = 1.0 + 0.1*np.random.randn(200)
        u2_0 = 1.0 + 0.1*np.random.randn(200)
        ic = np.concatenate([u1_0, u2_0])
        
        t_span = (0, 1)
        t_eval = np.linspace(0, 1, 50)
        
        sol = pde.Solve(ODE, ic, t_span, t_eval, method='BDF', rtol=1e-6)
        
        assert sol.success


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 70)
    print("Running Comprehensive Test Suite for StringToDiffEqSol")
    print("=" * 70)
    
    # Count tests
    test_classes = [
        TestAutoODEBasicParsing,
        TestAutoODESolving,
        TestAutoODEParseLocal,
        TestAutoODEParallel,
        TestAutoPDEBasicParsing,
        TestAutoPDESolving,
        TestAutoPDEParallel,
        TestEdgeCases,
        TestNumericalAccuracy,
        TestComplexSystems,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\n{class_name}:")
        print("-" * 70)
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                test_instance = test_class()
                test_method = getattr(test_instance, method_name)
                test_method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)}")
                failed_tests += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed_tests}/{total_tests} passed, {failed_tests} failed")
    print("=" * 70)
    
    return passed_tests, failed_tests, total_tests


if __name__ == "__main__":
    # Check if pytest is available
    if PYTEST_AVAILABLE:
        print("Running tests with pytest...\n")
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        print("pytest not available, running with custom test runner...\n")
        passed, failed, total = run_all_tests()
        sys.exit(0 if failed == 0 else 1)