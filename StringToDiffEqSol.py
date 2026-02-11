import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import re   
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
from tqdm import tqdm

class AutoODE:
    """
    Automatic ODE parser and solver.

    Converts human-readable ODE strings (of any order) into a first-order
    system and solves them numerically with scipy's solve_ivp.

    Workflow
    --------
    1. Write your ODEs as strings set equal to zero, e.g.:
           "x1'' + 3*x1' - x2 = 0"
       Variables are x1, x2, x3, ... ; t is the independent variable.
       Derivatives use primes: x1' (first), x1'' (second), etc.

    2. Pass any named coefficient functions (like forcing, damping, etc.)
       in a dictionary, e.g.: {"F": lambda t: 8.0}

    3. Call Parse (or Parse_Local for periodic systems).

    4. Call Solve with initial conditions to integrate.

    Attributes
    ----------
    ODE : callable or None
        The compiled right-hand-side function f(t, y) for solve_ivp.
        None until Parse is called.
    order : int
        Total number of first-order state variables after reduction.
        For example, a 2nd-order system with 3 variables gives order = 6
        (each variable contributes: value + first derivative = 2 states).
    state_vars : list of (str, int)
        List of (variable_name, derivative_order) pairs describing what
        was parsed. E.g. [('x1', 2), ('x2', 1)] means x1 has a 2nd-order
        equation and x2 has a 1st-order equation.

    Example
    -------
    >>> ode = AutoODE()
    >>> ode.Parse(
    ...     ["x1'' + 0.1*x1' + x1 - F(t) = 0"],
    ...     {"F": lambda t: np.sin(t)}
    ... )
    >>> sol = ode.Solve([1.0, 0.0], (0, 20), np.linspace(0, 20, 1000))
    """

    def __init__(self):
        pass
        
    def Help(self):
        print(self.__doc__)

    def Parse(self, ode_strings, function_dict):
        """
        Parse one or more ODE strings into a first-order system.

        This is the core method. It takes ODE equations written as strings,
        symbolically parses them, reduces any higher-order derivatives to a
        first-order system, and compiles a fast numerical right-hand-side
        function suitable for scipy's solve_ivp.

        How it works (high-level)
        -------------------------
        Given an equation like:   x1'' + 3*x1' + x1 = 0

        1. It detects that x1 appears up to 2nd order (x1'').
        2. It introduces state variables: y0 = x1, y1 = x1'.
        3. The 2nd-order ODE becomes the first-order system:
              dy0/dt = y1            (by definition: dx1/dt = x1')
              dy1/dt = -3*y1 - y0   (the original ODE solved for x1'')

        For coupled systems (multiple equations), the state vector is the
        concatenation of all variables and their derivatives:
           [x1, x1', x2, x2', x3, ...] etc.

        Parameters
        ----------
        ode_strings : str or list of str
            One or more ODE equations, each set equal to zero.

            Variable naming:
                - x1, x2, x3, ...  (dependent variables)
                - t                 (independent variable, time)

            Derivative notation:
                - x1'    = dx1/dt       (first derivative)
                - x1''   = d²x1/dt²    (second derivative)
                - x1'''  = d³x1/dt³    (third derivative, etc.)

            Function calls:
                - Named functions are written as calls, e.g. F(t), g(t,x1).
                  They must appear in function_dict.

            Examples:
                Single equation:
                    "x1'' + 0.1*x1' + x1 - F(t) = 0"

                Coupled system:
                    ["x1'' + a(t)*x2 = 0",
                     "x2'' + b(t)*x1 = 0"]

        function_dict : dict
            Maps function names (strings) to Python callables.

            The arguments in the equation string determine what the callable
            receives at runtime. For example:
                - F(t)      -> function_dict["F"] is called as F(t_value)
                - g(t,x1)   -> function_dict["g"] is called as g(t_value, x1_value)

            Examples:
                {"F": lambda t: np.sin(t)}
                {"g": lambda t, x: t * x,  "h": lambda t: 2.0}

        Raises
        ------
        ValueError
            If an equation contains no derivative to solve for, or if the
            highest-order derivative cannot be algebraically isolated.
        AssertionError
            If the string preprocessor fails to replace all prime notation.

        Returns
        -------
        ODE : callable
            The compiled right-hand-side function f(t, y) for solve_ivp.
        state_vars : list of (str, int)
            List of (variable_name, derivative_order) pairs.
        order : int
            Total number of first-order state variables.

        Notes
        -----
        After calling this method:
            - self.ODE is set to the compiled f(t, y) function.
            - self.order is the total number of first-order states.
            - self.state_vars lists each variable and its derivative order.
        """
        # Accept a single equation string as well as a list of them
        if isinstance(ode_strings, str):
            ode_strings = [ode_strings]

        # Create the sympy symbol for the independent variable t
        t = sp.Symbol('t')
        local_dict = {'t': t}  # Will be passed to sympy's parser so it knows 't'

        # =====================================================================
        # STEP 1: Replace function calls with placeholder symbols
        # =====================================================================
        # Why? Sympy's parser doesn't know about user functions like F(t) or
        # g(t,x1). If we leave them in, parsing would fail or misinterpret them.
        # So we replace each occurrence like "F(t)" with a unique placeholder
        # symbol "PHFN0", and record what function name and arguments it stands
        # for. Later (in the system function), we evaluate the real function.
        #
        # Example: "x1' - F(t)*x1 = 0"
        #   becomes: "x1' - PHFN0*x1"
        #   with func_placeholders = {Symbol('PHFN0'): ('F', ['t'])}

        func_placeholders = {}   # {sympy.Symbol: (function_name, [arg_names])}
        processed_strings = []   # Equations after placeholder substitution
        _ph = 0                  # Counter for unique placeholder names

        for ode_str in ode_strings:
            s = ode_str.replace(" ", "")  # Strip all spaces for uniform parsing

            # Normalize fancy quotes (e.g. from copy-paste) to ASCII apostrophe
            s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u0060", "'")

            # Remove the trailing "= 0" (the equation is implicitly = 0)
            s = re.sub(r'=\s*0\s*$', '', s)

            # For each known function name, find calls like fname(...) and replace
            for fname in function_dict:
                # Match "fname(" followed by arguments ")" — capture the args
                pat = re.compile(re.escape(fname) + r'\(([^)]*)\)')
                match = pat.search(s)
                while match:
                    ph = f'PHFN{_ph}'    # Unique placeholder name
                    _ph += 1
                    sym = sp.Symbol(ph)  # Sympy symbol to stand in for the call

                    # Record the actual function name and its argument list
                    # e.g. "g(t,x1)" -> fname='g', arg_names=['t','x1']
                    arg_str = match.group(1)
                    arg_names = [a.strip() for a in arg_str.split(',')] if arg_str else []
                    func_placeholders[sym] = (fname, arg_names)

                    # Register placeholder in the parser's dictionary
                    local_dict[ph] = sym

                    # Replace the first occurrence in the string and look for more
                    s = pat.sub(ph, s, count=1)
                    match = pat.search(s)

            processed_strings.append(s)

        # =====================================================================
        # STEP 2: Detect all variables and their derivative orders
        # =====================================================================
        # Scan all (now function-call-free) equation strings for patterns like
        # "x3", "x3'", "x3''", etc. to discover which variables exist and the
        # maximum derivative order each one reaches.
        #
        # Regex: (?<![a-zA-Z_\d])  = not preceded by alphanumeric (word boundary)
        #        x(\d+)            = "x" followed by digits -> variable name
        #        ('+)?             = optional sequence of primes -> derivative order
        #        (?!\w)            = not followed by word character (word boundary)

        all_var_tokens = set()  # Set of (var_name, prime_count) tuples
        for s in processed_strings:
            for m in re.finditer(r"(?<![a-zA-Z_\d])x(\d+)('+)?(?!\w)", s):
                name = f'x{m.group(1)}'
                primes = len(m.group(2)) if m.group(2) else 0
                all_var_tokens.add((name, primes))

        # For each variable, find the highest derivative order seen
        var_max_order = {}
        for name, order in all_var_tokens:
            var_max_order[name] = max(var_max_order.get(name, 0), order)

        # Sort numerically (x1, x2, ..., x10, x11, ...) not lexicographically
        var_names = sorted(var_max_order.keys(), key=lambda n: int(n[1:]))

        # =====================================================================
        # STEP 3: Create safe sympy symbols for each variable and derivative
        # =====================================================================
        # Sympy can't parse "x1'" directly (the prime is not valid syntax).
        # So for each variable x_k at derivative order j, we create a safe
        # symbol named "D{j}_{xk}":
        #   x1   -> D0_x1    (the variable itself, zeroth derivative)
        #   x1'  -> D1_x1    (first derivative)
        #   x1'' -> D2_x1    (second derivative)
        #
        # These safe names are registered in local_dict so sympy's parser
        # recognizes them after the preprocessing step below.

        var_symbols = {}  # var_symbols["x1"][0] = Symbol("D0_x1"), etc.
        for name in var_names:
            var_symbols[name] = {}
            for k in range(var_max_order[name] + 1):
                safe = f'D{k}_{name}'
                var_symbols[name][k] = sp.Symbol(safe)
                local_dict[safe] = var_symbols[name][k]

        # =====================================================================
        # STEP 4: Preprocess strings — replace primed vars with safe names
        # =====================================================================
        # Convert human notation into the safe symbol names so sympy can parse.
        # E.g. "x1''-3*x1'+x2" becomes "D2_x1-3*D1_x1+D0_x2"
        #
        # We match variables longest-name-first so "x10" is matched before "x1"
        # (greedy matching avoids partial replacements).

        var_names_by_length = sorted(var_names, key=len, reverse=True)

        def preprocess(s):
            """Replace every occurrence of xN, xN', xN'', ... with D{k}_xN."""
            result = []
            i = 0
            while i < len(s):
                matched = False
                for name in var_names_by_length:
                    # Check if the variable name starts at position i
                    if s[i:i+len(name)] == name:
                        # Ensure it's not part of a longer word (e.g. "ax1")
                        if i > 0 and (s[i-1].isalnum() or s[i-1] == '_'):
                            continue

                        # Count trailing primes to determine derivative order
                        j = i + len(name)
                        n_primes = 0
                        while j < len(s) and s[j] in ("'", "\u2018", "\u2019"):
                            n_primes += 1
                            j += 1

                        # Ensure it's not followed by more alphanumerics
                        if j < len(s) and (s[j].isalnum() or s[j] == '_'):
                            continue

                        # Emit the safe symbol name
                        result.append(f'D{n_primes}_{name}')
                        i = j
                        matched = True
                        break
                if not matched:
                    result.append(s[i])
                    i += 1
            return ''.join(result)

        # =====================================================================
        # STEP 5: Parse each equation symbolically and solve for highest deriv
        # =====================================================================
        # After preprocessing, each equation string contains only safe symbol
        # names and standard arithmetic. We parse it with sympy, then
        # algebraically solve for the highest-order derivative in that equation.
        #
        # Example: "D2_x1 + 3*D1_x1 + D0_x1" parsed to an expression, then
        # solved for D2_x1 -> D2_x1 = -3*D1_x1 - D0_x1

        parsed_equations = []  # List of (var_name, deriv_order, rhs_expression)
        for s in processed_strings:
            s = preprocess(s)

            # Safety check: all primes should have been replaced
            assert "'" not in s, f"Preprocess failed, primes remain: {repr(s)}"

            # Parse the string into a sympy expression (implicitly = 0)
            expr = parse_expr(s, local_dict=local_dict)

            # Find the highest-order derivative that appears in this equation.
            # This is what we solve for — it becomes the "output" of this equation.
            best_var, best_order = None, -1
            for name in var_names:
                for k in range(var_max_order[name], -1, -1):
                    if expr.has(var_symbols[name][k]) and k > best_order:
                        best_var, best_order = name, k
                        break

            if best_var is None or best_order <= 0:
                raise ValueError(f"No derivative to solve for in: {s}")

            # Solve the expression = 0 for the highest derivative
            highest = var_symbols[best_var][best_order]
            solved = sp.solve(expr, highest)
            if not solved:
                raise ValueError(f"Cannot isolate {highest} in: {s}")

            # Store: "in the equation for best_var, the best_order-th derivative
            # equals solved[0]" (a sympy expression in terms of lower derivatives)
            parsed_equations.append((best_var, best_order, solved[0]))

        # =====================================================================
        # STEP 6: Build the state vector index mapping
        # =====================================================================
        # To convert to a first-order system, each variable x_k of order n_k
        # contributes n_k state components:
        #   x_k, x_k', x_k'', ..., x_k^{(n_k - 1)}
        #
        # We assign each a flat index in the state vector y[].
        #
        # Example with x1 (2nd order) and x2 (1st order):
        #   y[0] = x1,  y[1] = x1',  y[2] = x2
        #   state_index = {('x1',0):0, ('x1',1):1, ('x2',0):2}

        # First, update max orders to match what the equations actually require
        for name in var_names:
            needed = max((o for v, o, _ in parsed_equations if v == name), default=0)
            if needed > 0:
                var_max_order[name] = needed

        state_index = {}  # Maps (var_name, derivative_order) -> flat index in y[]
        idx = 0
        for name in var_names:
            max_o = var_max_order[name]
            for k in range(max(max_o, 1)):  # At least 1 state per variable
                state_index[(name, k)] = idx
                idx += 1
        total_states = idx
        order = total_states

        # =====================================================================
        # STEP 7: Build the compiled numerical system function f(t, y)
        # =====================================================================
        # This is the function that scipy's solve_ivp will call at every time
        # step. It computes dy/dt given the current t and y.
        #
        # The system has two kinds of equations:
        #
        #   a) CHAIN RULES (from reducing order):
        #      d(x_k^{(j)})/dt = x_k^{(j+1)}    for j = 0, ..., n_k - 2
        #      These are trivial: "the derivative of x1 is x1'"
        #
        #   b) PARSED EQUATIONS (the actual physics):
        #      d(x_k^{(n_k-1)})/dt = rhs(...)    from Step 5
        #      These are the user's equations solved for the highest derivative.
        #
        # For speed, we use sympy.lambdify to compile each RHS expression
        # into a fast numpy function, rather than doing symbolic substitution
        # at every time step.

        # Build the ordered list of symbols that the lambdified functions expect:
        # [t, D0_x1, D1_x1, ..., D0_x2, ..., PHFN0, PHFN1, ...]
        all_syms = [t]
        sym_to_state = {}
        for name in var_names:
            for k in range(var_max_order[name]):
                sym = var_symbols[name][k]
                all_syms.append(sym)
                sym_to_state[sym] = (name, k)
        ph_list = list(func_placeholders.keys())
        all_syms.extend(ph_list)

        # Compile each RHS expression into a fast callable
        lambdified_rhs = []
        for best_var, best_order, rhs_expr in parsed_equations:
            fn = sp.lambdify(all_syms, rhs_expr, modules='numpy')
            lambdified_rhs.append((best_var, best_order, fn))

        # ----- The actual system function passed to solve_ivp -----
        def system(t_val, y):
            """
            Right-hand side f(t, y) of the first-order system dy/dt = f(t, y).

            Parameters
            ----------
            t_val : float
                Current time.
            y : ndarray of shape (total_states,)
                Current state vector.

            Returns
            -------
            dydt : ndarray of shape (total_states,)
                Time derivatives of each state.
            """
            dydt = np.zeros(total_states)

            # (a) Chain rules: d(x_k^{(j)})/dt = x_k^{(j+1)}
            #     For a 2nd-order variable x1: dy[0]/dt = y[1]  (i.e. dx1/dt = x1')
            for name in var_names:
                max_o = var_max_order[name]
                for k in range(max_o - 1):
                    dydt[state_index[(name, k)]] = y[state_index[(name, k + 1)]]

            # Build the argument list for the lambdified RHS functions.
            # Order must match all_syms: [t, state_vars..., placeholders...]
            args = [t_val]

            # Append current state values in the same order as all_syms
            for name in var_names:
                for k in range(var_max_order[name]):
                    args.append(y[state_index[(name, k)]])

            # Evaluate each placeholder function with its actual arguments.
            # For example, if "g(t,x1)" was replaced by PHFN0, we now call
            # function_dict["g"](t_val, y[state_index[("x1",0)]]).
            for ph_sym in ph_list:
                fname, arg_names = func_placeholders[ph_sym]
                resolved_args = []
                for aname in arg_names:
                    if aname == 't':
                        resolved_args.append(t_val)
                    elif (aname, 0) in state_index:
                        # Look up the variable's current value (zeroth derivative)
                        resolved_args.append(y[state_index[(aname, 0)]])
                    else:
                        raise ValueError(
                            f"Unknown argument '{aname}' in function call {fname}(...)"
                        )
                args.append(function_dict[fname](*resolved_args))

            # (b) Parsed equations: fill in the highest derivative for each variable
            for best_var, best_order, fn in lambdified_rhs:
                val = fn(*args)
                dydt[state_index[(best_var, best_order - 1)]] = val

            return dydt

        # Store the compiled system and metadata
        ODE = system
        state_vars = [(name, var_max_order[name]) for name in var_names]
        return ODE, state_vars, order

    def Parse_Local(self, ode_string, function_dict, dim, mod=True):
        """
        Expand a local/periodic ODE template into a full coupled system.

        This is a convenience wrapper around Parse for systems where every
        variable obeys the same equation, but with shifted neighbor indices and
        optional periodic (wrap-around) boundary conditions. You write ONE
        template equation using indexed notation, and this method expands it
        into `dim` concrete equations.

        This is ideal for lattice-like systems such as Lorenz 96, coupled
        oscillator chains, discretized PDEs on periodic domains, etc.

        How it works
        ------------
        Given a template like:
            "x_i' - (x_{i+1} - x_{i-2}) * x_{i-1} + x_i - F(t) = 0"
        and dim=4, it generates 4 equations:
            i=1: "x1' - (x2 - x3) * x4 + x1 - F(t) = 0"   (x_{1-2}=x3 wraps)
            i=2: "x2' - (x3 - x4) * x1 + x2 - F(t) = 0"   (x_{2-2}=x4 wraps)
            i=3: "x3' - (x4 - x1) * x2 + x3 - F(t) = 0"
            i=4: "x4' - (x1 - x2) * x3 + x4 - F(t) = 0"

        Boundary modes (mod parameter)
        -------------------------------
        mod=True  (default): Periodic boundaries. Indices wrap modularly:
            x_{dim+1} = x1, x_0 = x_{dim}, etc.

        mod=False: Fixed (Dirichlet zero) boundaries. Any neighbor reference
            that falls outside [1, dim] is replaced by 0, effectively setting
            ghost nodes to zero. For example with dim=4:
            i=1: x_{1-2} -> x_{-1} -> replaced by 0
            i=4: x_{4+1} -> x_5   -> replaced by 0

        Index-dependent functions
        -------------------------
        Functions can depend on the index `i`. When `i` appears as an argument
        in a function call, it is replaced by the concrete index value, and a
        per-index wrapper function is generated automatically.

        Example: "x_i' - F(t, i) = 0" with F: lambda t, i: sin(i * t)
        Expands to (dim=3):
            i=1: "x1' - F_1(t) = 0"   where F_1(t) calls F(t, 1)
            i=2: "x2' - F_2(t) = 0"   where F_2(t) calls F(t, 2)
            i=3: "x3' - F_3(t) = 0"   where F_3(t) calls F(t, 3)

        Parameters
        ----------
        ode_string : str
            A single template equation using indexed notation:

            - x_i  or  x_{i}       : the i-th variable (current index)
            - x_{i+1}              : next neighbor
            - x_{i-2}              : two neighbors back
            - x_{i+k} / x_{i-k}   : any integer offset, wraps if mod=True

            Derivatives work on all forms:
            - x_i'                 : first derivative of current variable
            - x_{i+1}''           : second derivative of next neighbor

            Functions from function_dict:
            - F(t)                 : index-independent (shared across all equations)
            - F(t, i)              : index-dependent (specialized per equation)
            - g(t, x_i, i)        : can combine state variables and index

            Example (Lorenz 96):
                "x_i' - (x_{i+1} - x_{i-2}) * x_{i-1} + x_i - F(t) = 0"

            Example (index-dependent forcing):
                "x_i' + x_i - F(t, i) = 0"

        function_dict : dict
            Maps function names (strings) to Python callables.
            If `i` appears in the arguments, the callable must accept the
            index as an integer at that argument position.
            Examples:
                {"F": lambda t: 8.0}             # no index dependence
                {"F": lambda t, i: sin(i * t)}   # index-dependent

        dim : int
            Number of variables (x1 through x{dim}).

        mod : bool, optional
            If True (default), indices wrap periodically:
                x_{dim+1} = x1, x_0 = x_{dim}, etc.
            If False, out-of-range neighbor references are replaced by 0
            (Dirichlet zero boundary conditions).

        Notes
        -----
        After calling this method, self.ODE, self.order, and self.state_vars
        are all set (it delegates to Parse internally).
        """

        # First pass: detect which function calls contain `i` as an argument.
        # We need to know this before expanding, so we can generate per-index
        # wrapper functions for those calls.
        #
        # For function calls WITHOUT `i`, the original function_dict entry is
        # reused directly (all equations share the same function).
        #
        # For function calls WITH `i`, we create a new entry for each index:
        #   "F(t, i)" at idx=3 -> "F_3(t)" with expanded_funcs["F_3"] = lambda t: F(t, 3)
        # The `i` argument is baked in, and the remaining args are passed through.

        expanded_funcs = {}  # Will hold all functions for the expanded system

        # Copy over all original functions (they may be used directly if no `i`)
        for fname, fn in function_dict.items():
            expanded_funcs[fname] = fn

        equations = []
        for idx in range(1, dim + 1):
            eq = ode_string

            # --- Handle function calls that may contain `i` as an argument ---
            # For each known function, find calls like fname(arg1, arg2, ...)
            # and check if any argument is literally `i`.
            for fname in function_dict:
                pat = re.compile(re.escape(fname) + r'\(([^)]*)\)')
                match = pat.search(eq)
                while match:
                    arg_str = match.group(1)
                    args = [a.strip() for a in arg_str.split(',')] if arg_str else []

                    if 'i' in args:
                        # This call depends on the index — create a specialized
                        # wrapper function with `i` bound to `idx`.
                        specialized_name = f'{fname}_{idx}'

                        # Build the new argument list with `i` removed
                        remaining_args = [a for a in args if a != 'i']
                        new_call = f'{specialized_name}({",".join(remaining_args)})'

                        # Create wrapper that injects idx at the right positions.
                        # We figure out which positions `i` occupied and bind them.
                        i_positions = [pos for pos, a in enumerate(args) if a == 'i']
                        other_positions = [pos for pos, a in enumerate(args) if a != 'i']

                        def _make_wrapper(fn, _idx, _i_pos, _other_pos, _nargs):
                            """Create a closure that injects idx at the i-positions."""
                            def wrapper(*called_args):
                                full_args = [None] * _nargs
                                for pos in _i_pos:
                                    full_args[pos] = _idx
                                for j, pos in enumerate(_other_pos):
                                    full_args[pos] = called_args[j]
                                return fn(*full_args)
                            return wrapper

                        expanded_funcs[specialized_name] = _make_wrapper(
                            function_dict[fname], idx, i_positions,
                            other_positions, len(args)
                        )

                        # Replace this specific call in the equation string
                        eq = eq[:match.start()] + new_call + eq[match.end():]
                    else:
                        # No `i` in arguments — leave the call as is.
                        # Move past this match to avoid infinite loop.
                        break

                    # Search again (there could be multiple calls to the same func)
                    match = pat.search(eq)

            # --- Replace indexed variable forms: x_{i+1}, x_{i-2}, x_i, etc. ---

            # Replace braced indexed forms: x_{i}, x_{i+1}, x_{i-2}, etc.
            # The regex captures the optional offset (+1, -2, etc.) and any primes.
            #
            # mod=True:  index wraps modularly -> ((idx-1+offset) % dim) + 1
            # mod=False: out-of-range indices are replaced by literal 0
            def _replace_braced(m, _idx=idx):
                offset_str = m.group(1)    # e.g. "+1", "-2", or None for x_{i}
                primes = m.group(2) or ''  # e.g. "'", "''", or empty
                offset = int(offset_str) if offset_str else 0
                if mod:
                    concrete = ((_idx - 1 + offset) % dim) + 1
                    return f'x{concrete}{primes}'
                else:
                    raw = _idx + offset
                    if raw < 1 or raw > dim:
                        # Out-of-range: replace with 0 (Dirichlet boundary)
                        return '0'
                    return f'x{raw}{primes}'
            eq = re.sub(r'x_\{i([+-]\d+)?\}(\'*)', _replace_braced, eq)

            # Then, replace the simple unbraced form x_i (no offset allowed)
            def _replace_unbraced(m, _idx=idx):
                primes = m.group(1) or ''
                return f'x{_idx}{primes}'
            eq = re.sub(r'x_i(\'*)', _replace_unbraced, eq)

            equations.append(eq)

        # Delegate to Parse with the expanded equation list and functions
        return self.Parse(equations, expanded_funcs)

    def Solve(self, ODE, initial_conditions, t_span, t_eval, method='RK45', **kwargs):
        """
        Numerically integrate the parsed ODE system.

        Parameters
        ----------
        initial_conditions : list or array of float
            Initial values for the full first-order state vector.
            Length must equal self.order.

            For a single 2nd-order variable x1:
                [x1(0), x1'(0)]

            For a system with x1 (2nd order) and x2 (1st order):
                [x1(0), x1'(0), x2(0)]

            For Parse_Local with dim first-order variables:
                [x1(0), x2(0), ..., x{dim}(0)]

        t_span : tuple of (float, float)
            Start and end times: (t_start, t_end).

        t_eval : array-like of float
            Times at which to store the solution. The solver may take
            internal steps at other times, but output is interpolated here.

        method : str, optional
            Integration method for solve_ivp. Default is 'RK45'.
            Other options: 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'.

        **kwargs
            Additional keyword arguments passed to scipy.integrate.solve_ivp,
            e.g. rtol, atol, max_step, events, etc.

        Returns
        -------
        sol : scipy.integrate.OdeSolution
            The solution object. Key attributes:
            - sol.t    : 1D array of times
            - sol.y    : 2D array of shape (n_states, n_times)
            - sol.sol  : dense interpolant (callable)

        Raises
        ------
        RuntimeError
            If Parse has not been called yet.
        """
        if ODE is None:
            raise RuntimeError("Call Parse first.")
        sol = solve_ivp(
            ODE, t_span, initial_conditions,
            t_eval=t_eval, method=method, dense_output=True, **kwargs
        )
        return sol
    



    def Parallel_Parse(self, inputs: list, n_jobs: int = -1, backend: str = "loky") -> list:
        """
        Parse multiple ODE systems in parallel using joblib.

        Parameters
        ----------
        inputs : list of (ode_strings, function_dict)
            Each element is a tuple of arguments that would be passed to Parse.

            NOTE: When using the default 'loky' backend, function_dict values
            must be picklable. Plain ``def`` functions work; ``lambda`` functions
            do NOT. Switch to backend="threading" if you need lambdas.

        n_jobs : int, optional
            Number of parallel workers. -1 (default) uses all available cores.

        backend : str, optional
            Joblib parallel backend. Default is "loky" (multiprocessing).
            Use "threading" when function_dict contains lambdas or closures.

        Returns
        -------
        list of (list of ODEs, list of state_vars, list of orders)
            Three lists containing ODE functions, state variable lists, and
            order values for each parsed system, in the same order as inputs.

        Example
        -------
        >>> results = AutoODE.Parallel_Parse([
        ...     (["x1'' + x1 = 0"], {}),
        ...     (["x1' - x1 = 0"],  {}),
        ... ], n_jobs=2)
        """
        raw_results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(self.Parse)(ode_strings, function_dict)
            for ode_strings, function_dict in tqdm(inputs, desc="Parsing ODEs")
        )
        results = list(map(list, zip(*raw_results)))
        
        return results

    def Parallel_Parse_Local(self, inputs: list, n_jobs: int = -1, backend: str = "threading") -> list:
        """
        Parse multiple local/periodic ODE systems in parallel using joblib.

        Parameters
        ----------
        inputs : list of (ode_string, function_dict, dim) or (ode_string, function_dict, dim, mod)
            Each element is a tuple of arguments that would be passed to
            Parse_Local.  The ``mod`` argument is optional and defaults to True.

            NOTE: When using the default 'loky' backend, function_dict values
            must be picklable. Plain ``def`` functions work; ``lambda`` functions
            do NOT. Switch to backend="threading" if you need lambdas.

        n_jobs : int, optional
            Number of parallel workers. -1 (default) uses all available cores.

        backend : str, optional
            Joblib parallel backend. Default is "threading" (multithreading).
            Use "loky" when function_dict contains only picklable objects.

        Returns
        -------
        list of (list of ODEs, list of state_vars, list of orders)
            Three lists containing ODE functions, state variable lists, and
            order values for each parsed system, in the same order as inputs.

        Example
        -------
        >>> results = AutoODE.Parallel_Parse_Local([
        ...     ("x_i' + x_i = 0", {}, 10),
        ...     ("x_i' + x_i = 0", {}, 20, False),
        ... ], n_jobs=2)
        """


        raw_results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(self.Parse_Local)(*inp)
            for inp in tqdm(inputs, desc="Parsing local ODEs")
        )
        results = list(map(list, zip(*raw_results)))
        
        return results

    def Parallel_Solve(self, odes: list, inputs: list, n_jobs: int = -1, backend: str = "loky") -> list:
        """
        Solve the same ODE system with multiple initial conditions in parallel.

        This is useful for parameter sweeps, ensemble simulations, Monte Carlo
        analysis, or exploring different initial states of the same dynamics.

        Parameters
        ----------
        ode : AutoODE
            A configured AutoODE instance (Parse or Parse_Local already called).

        inputs : list of tuples
            Each element is a tuple of arguments for the Solve method:
            (initial_conditions, t_span, t_eval[, method[, **kwargs]])

            Minimal form: (ic, t_span, t_eval)
            With method: (ic, t_span, t_eval, method)
            With kwargs: (ic, t_span, t_eval, method, kwargs_dict)

            Examples:
                # Same time span, different ICs
                [([1.0, 0.0], (0, 10), t_eval),
                 ([2.0, 1.0], (0, 10), t_eval)]

                # Different time spans
                [([1.0, 0.0], (0, 10), np.linspace(0, 10, 100)),
                 ([1.0, 0.0], (0, 20), np.linspace(0, 20, 200))]

                # Custom solver options
                [([1.0, 0.0], (0, 10), t_eval, 'RK45', {'rtol': 1e-8}),
                 ([2.0, 0.0], (0, 10), t_eval, 'DOP853', {'atol': 1e-10})]

        n_jobs : int, optional
            Number of parallel workers. -1 (default) uses all available cores.

        backend : str, optional
            Joblib parallel backend. Default is "loky" (multiprocessing).
            Use "threading" for thread-based parallelism.

        Returns
        -------
        list of OdeSolution
            Solution objects in the same order as inputs.

        Example
        -------
        >>> ode = AutoODE()
        >>> ode.Parse(["x1'' + 0.1*x1' + x1 = 0"], {})
        >>> t_eval = np.linspace(0, 20, 1000)
        >>> ics = [[1.0, 0.0], [0.5, 0.5], [2.0, -1.0]]
        >>> inputs = [(ic, (0, 20), t_eval) for ic in ics]
        >>> solutions = AutoODE.Parallel_Solve(ode, inputs, n_jobs=3)
        """
        def _unpack(args):
            ic = args[0]
            t_span = args[1]
            t_eval = args[2]
            method = args[3] if len(args) > 3 else 'RK45'
            return ic, t_span, t_eval, method
        
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(self.Solve)(ode, *_unpack(inp))
            for ode, inp in zip(tqdm(odes, desc="Solving ODEs"), inputs)
        )
        return results

    
class AutoPDE:
    """
    Automatic 1D PDE parser and solver (method of lines).

    Converts human-readable PDE strings into a spatially discretized ODE
    system and solves them numerically using finite differences + solve_ivp.

    Supports arbitrary order in both time and space — write exactly the PDE
    you mean, and AutoPDE handles the rest.

    Workflow
    --------
    1. Write your PDEs as strings set equal to zero, e.g.:
           "u1_t  - 0.01*u1_xx = 0"        (heat equation,  1st order in t)
           "u1_tt - c**2*u1_xx = 0"         (wave equation,  2nd order in t)
           "u1_ttt + u1_xxx = 0"            (3rd order in both t and x)

       Field variables are u1, u2, ...; t is time, x is space.

       Time-derivative notation  (any order):
           u1_t     = du1/dt       (1st order in time)
           u1_tt    = d²u1/dt²    (2nd order in time)
           u1_ttt   = d³u1/dt³    (3rd order in time, etc.)

       Spatial-derivative notation (any order):
           u1_x     = du1/dx
           u1_xx    = d²u1/dx²
           u1_xxx   = d³u1/dx³
           u1_xxxx  = d⁴u1/dx⁴
           (higher orders supported for 'periodic' BC)

    2. Pass coefficient/forcing functions in a dictionary, e.g.:
           {"f": lambda x, t: np.sin(np.pi * x)}

    3. Call Parse with spatial domain, grid size, and boundary conditions.

    4. Call Solve with initial conditions to integrate.
       For a field u1 that is nth-order in time, the initial conditions must
       include all n time-derivative levels at t=0:
           [u1(x,0), du1/dt(x,0), ..., d^{n-1}u1/dt^{n-1}(x,0), u2(x,0), ...]

    Attributes
    ----------
    ODE : callable or None
        Compiled f(t, y) function for solve_ivp.
    order : int
        Total state dimension = sum over fields of (time_order_k * N).
        For a single 1st-order field:  N.
        For a single 2nd-order field:  2*N.
        For two 1st-order fields:      2*N.
    field_vars : list of str
        Detected field variable names, e.g. ['u1', 'u2'].
    time_orders : dict
        Maps each field name to its time-derivative order, e.g.
        {'u1': 2, 'u2': 1} for a wave + advection coupled system.
    N : int
        Number of spatial grid points.
    x_grid : ndarray
        Spatial grid coordinates.
    dx : float
        Grid spacing.

    Examples
    --------
    Heat equation (1st order in t):

    >>> pde = AutoPDE()
    >>> pde.Parse("u1_t - 0.01*u1_xx = 0", {}, (0, 1), 100, bc='dirichlet')
    >>> u0 = np.sin(np.pi * pde.x_grid)
    >>> sol = pde.Solve(pde.ODE, u0, (0, 0.5), np.linspace(0, 0.5, 100))

    Wave equation (2nd order in t):

    >>> pde = AutoPDE()
    >>> pde.Parse("u1_tt - 4.0*u1_xx = 0", {}, (0, 1), 200, bc='dirichlet')
    >>> x = pde.x_grid
    >>> ic = np.concatenate([np.sin(np.pi*x), np.zeros_like(x)])
    >>> sol = pde.Solve(pde.ODE, ic, (0, 2), np.linspace(0, 2, 200))
    >>> u = pde.get_field(sol, 'u1')   # shape (N, n_times)
    """

    def __init__(self):
        self.ODE = None
        self.order = 0
        self.field_vars = []
        self.time_orders = {}    # {field_name: int} — time-derivative order per field
        self.state_offsets = {}  # {field_name: int} — flat index of field's first state block
        self.N = 0
        self.x_grid = None
        self.dx = 0.0

    def Help(self):
        print(self.__doc__)

    def Parse(self, pde_strings, function_dict, x_span, N, bc='periodic', bc_values=None):
        """
        Parse PDE strings and build a method-of-lines ODE system.

        Spatial derivatives are approximated with central finite differences
        (2nd-order accurate). The resulting semi-discrete system is suitable
        for integration with scipy's solve_ivp.

        Parameters
        ----------
        pde_strings : str or list of str
            PDE equations set equal to zero.

            Variable naming:
                - u1, u2, u3, ... (field variables)
                - t                (time)
                - x                (space)

            Derivative notation (time — any order):
                - u1_t      = du1/dt           (1st order in time)
                - u1_tt     = d²u1/dt²         (2nd order in time)
                - u1_ttt    = d³u1/dt³         (3rd order, etc.)

            Derivative notation (space — any order):
                - u1_x      = du1/dx           (1st spatial derivative)
                - u1_xx     = d²u1/dx²         (2nd spatial derivative)
                - u1_xxx    = d³u1/dx³         (3rd spatial derivative)
                - u1_xxxx   = d⁴u1/dx⁴        (4th spatial derivative)

            Function calls:
                - f(x, t), g(x, t, u1), etc. must appear in function_dict.

            Examples:
                Heat equation (1st order in t):
                    "u1_t - 0.01*u1_xx = 0"

                Wave equation (2nd order in t):
                    "u1_tt - 4.0*u1_xx = 0"

                Burgers' equation:
                    "u1_t + u1*u1_x - 0.01*u1_xx = 0"

                Coupled reaction-diffusion:
                    ["u1_t - 0.1*u1_xx + u1*u2 = 0",
                     "u2_t - 0.2*u2_xx - u1*u2 = 0"]

                3rd-order Airy/dispersive:
                    "u1_t + u1_xxx = 0" 

        function_dict : dict
            Maps function names to callables. Arguments in the string
            determine what the callable receives:
                - x    -> spatial grid array (shape N)
                - t    -> scalar time value
                - u1   -> field value array (shape N)

            Example:
                {"f": lambda x, t: np.sin(np.pi * x) * np.cos(t)}

        x_span : tuple of (float, float)
            Spatial domain (x_start, x_end).

        N : int
            Number of spatial grid points.

        bc : str, optional
            Boundary condition type. Default is 'periodic'.

            'periodic':  x_end wraps to x_start. N points, endpoint excluded.
                         Supports spatial derivatives up to 4th order.

            'dirichlet': Fixed field values at boundaries. N interior points.
                         Supports spatial derivatives up to 4th order.

            'neumann':   Fixed spatial derivatives at boundaries. N points
                         including endpoints.
                         Supports spatial derivatives up to 4th order.

        bc_values : dict, optional
            Boundary values per variable: {var_name: (left, right)}.
            For 'dirichlet': field values at boundaries.
            For 'neumann': spatial derivatives at boundaries.
            Defaults to (0.0, 0.0) for each variable.
            Ignored for 'periodic'.

        Notes on initial conditions for higher time-order fields
        ---------------------------------------------------------
        A field u1 that is nth-order in time requires n initial condition
        arrays at t=0, concatenated in the flat IC vector in the order:
            [u1(x,0), du1/dt(x,0), ..., d^{n-1}u1/dt^{n-1}(x,0),
             u2(x,0), du2/dt(x,0), ..., ...]
        Use pde.state_offsets[name] to find where each block starts.

        Raises
        ------
        ValueError
            If an equation has no time derivative, or if the time derivative
            cannot be algebraically isolated, or if an unsupported spatial
            derivative order is requested.

        Returns
        -------
        ODE : callable
            The compiled right-hand-side function f(t, y) for solve_ivp.
        field_vars : list of str
            List of detected field variable names (e.g., ['u1', 'u2']).
        order : int
            Total state dimension = sum over all fields of (time_order_k * N).

        Notes
        -----
        For stiff PDEs (e.g. diffusion-dominated), use method='Radau' or
        method='BDF' in the Solve call.
        """
        if isinstance(pde_strings, str):
            pde_strings = [pde_strings]

        t_sym = sp.Symbol('t')
        x_sym = sp.Symbol('x')
        local_dict = {'t': t_sym, 'x': x_sym}

        # =================================================================
        # STEP 1: Replace function calls with placeholder symbols
        # =================================================================
        func_placeholders = {}
        processed_strings = []
        _ph = 0

        for pde_str in pde_strings:
            s = pde_str.replace(" ", "")
            s = re.sub(r'=\s*0\s*$', '', s)

            for fname in function_dict:
                pat = re.compile(re.escape(fname) + r'\(([^)]*)\)')
                match = pat.search(s)
                while match:
                    ph = f'PHFN{_ph}'
                    _ph += 1
                    sym = sp.Symbol(ph)
                    arg_str = match.group(1)
                    arg_names = [a.strip() for a in arg_str.split(',')] if arg_str else []
                    func_placeholders[sym] = (fname, arg_names)
                    local_dict[ph] = sym
                    s = pat.sub(ph, s, count=1)
                    match = pat.search(s)

            processed_strings.append(s)

        # =================================================================
        # STEP 2: Detect field variables, spatial orders, AND time orders
        # =================================================================
        # Matches tokens like: u1, u1_t, u1_tt, u1_x, u1_xx, u1_xxx, etc.
        # The regex captures the subscript suffix: t+, x+, or none.
        all_tokens = set()
        for s in processed_strings:
            for m in re.finditer(
                r'(?<![a-zA-Z_\d])u(\d+)(?:_(t+|x+))?(?![a-zA-Z\d])', s
            ):
                name = f'u{m.group(1)}'
                deriv_type = m.group(2)   # 't', 'tt', 'x', 'xx', etc., or None
                all_tokens.add((name, deriv_type))

        field_var_set = set()
        max_spatial_order = {}  # {name: int}  highest spatial derivative order
        max_time_order = {}     # {name: int}  highest time derivative order

        for name, deriv_type in all_tokens:
            field_var_set.add(name)
            if deriv_type is None:
                max_spatial_order.setdefault(name, 0)
                max_time_order.setdefault(name, 0)
            elif set(deriv_type) == {'t'}:
                # Pure time derivative: order = number of 't' chars
                t_ord = len(deriv_type)
                max_time_order[name] = max(max_time_order.get(name, 0), t_ord)
                max_spatial_order.setdefault(name, 0)
            else:
                # Spatial derivative: count 'x' chars
                x_ord = len(deriv_type)
                max_spatial_order[name] = max(max_spatial_order.get(name, 0), x_ord)
                max_time_order.setdefault(name, 0)

        field_vars = sorted(field_var_set, key=lambda n: int(n[1:]))
        self.field_vars = field_vars
        n_fields = len(field_vars)

        # =================================================================
        # STEP 3: Create sympy symbols for spatial AND time derivatives
        # =================================================================
        # Spatial:  u1       -> S0_u1   (field value)
        #           u1_x     -> S1_u1   (1st spatial derivative)
        #           u1_xx    -> S2_u1   (2nd spatial derivative)
        # Time:     u1_t     -> St1_u1  (1st time derivative)
        #           u1_tt    -> St2_u1  (2nd time derivative, solve target)
        #           u1_ttt   -> St3_u1  (3rd time derivative, etc.)
        #
        # The highest time-derivative symbol is what we solve for.

        var_symbols = {}        # var_symbols[name][k] = Symbol for k-th spatial deriv
        time_deriv_syms = {}    # time_deriv_syms[name][k] = Symbol for k-th time deriv

        for name in field_vars:
            var_symbols[name] = {}
            max_so = max_spatial_order.get(name, 0)
            for k in range(max_so + 1):
                safe = f'S{k}_{name}'
                var_symbols[name][k] = sp.Symbol(safe)
                local_dict[safe] = var_symbols[name][k]

            time_deriv_syms[name] = {}
            max_to = max(max_time_order.get(name, 1), 1)
            for k in range(1, max_to + 1):
                safe_t = f'St{k}_{name}'
                time_deriv_syms[name][k] = sp.Symbol(safe_t)
                local_dict[safe_t] = time_deriv_syms[name][k]

        # =================================================================
        # STEP 4: Preprocess strings — convert PDE notation to safe names
        # =================================================================
        # "u1_tt - c**2*u1_xx" becomes "St2_u1 - c**2*S2_u1"
        sorted_vars = sorted(field_vars, key=len, reverse=True)

        def preprocess(s):
            result = []
            i = 0
            while i < len(s):
                matched = False
                for name in sorted_vars:
                    if s[i:i + len(name)] == name:
                        if i > 0 and (s[i - 1].isalnum() or s[i - 1] == '_'):
                            continue

                        j = i + len(name)

                        if j < len(s) and s[j] == '_':
                            k = j + 1
                            # Time derivative: one or more 't' chars
                            if k < len(s) and s[k] == 't':
                                n_t = 0
                                while k < len(s) and s[k] == 't':
                                    n_t += 1
                                    k += 1
                                if k >= len(s) or not (s[k].isalnum() or s[k] == '_'):
                                    result.append(f'St{n_t}_{name}')
                                    i = k
                                    matched = True
                                    break
                            # Spatial derivative: one or more 'x' chars
                            elif k < len(s) and s[k] == 'x':
                                n_x = 0
                                while k < len(s) and s[k] == 'x':
                                    n_x += 1
                                    k += 1
                                if k >= len(s) or not (s[k].isalnum() or s[k] == '_'):
                                    result.append(f'S{n_x}_{name}')
                                    i = k
                                    matched = True
                                    break

                        if not matched:
                            if j >= len(s) or not (s[j].isalnum() or s[j] == '_'):
                                result.append(f'S0_{name}')
                                i = j
                                matched = True
                                break

                if not matched:
                    result.append(s[i])
                    i += 1
            return ''.join(result)

        # =================================================================
        # STEP 5: Parse each equation and solve for the highest time deriv
        # =================================================================
        parsed_equations = []   # [(field_name, time_order, rhs_sympy_expr)]
        for s in processed_strings:
            s = preprocess(s)
            expr = parse_expr(s, local_dict=local_dict)

            target_var = None
            target_tord = 0
            for name in field_vars:
                max_to = max(max_time_order.get(name, 1), 1)
                for k in range(max_to, 0, -1):
                    if expr.has(time_deriv_syms[name][k]):
                        if k > target_tord:
                            target_var, target_tord = name, k
                        break

            if target_var is None:
                raise ValueError(f"No time derivative (u_t, u_tt, ...) found in: {s}")

            sym_to_solve = time_deriv_syms[target_var][target_tord]
            solved = sp.solve(expr, sym_to_solve)
            if not solved:
                raise ValueError(f"Cannot isolate {sym_to_solve} in: {s}")

            parsed_equations.append((target_var, target_tord, solved[0]))

        # =================================================================
        # STEP 6: Build spatial grid
        # =================================================================
        x_start, x_end = x_span

        if bc == 'periodic':
            dx = (x_end - x_start) / N
            x_grid = np.linspace(x_start, x_end - dx, N)
        elif bc == 'dirichlet':
            dx = (x_end - x_start) / (N + 1)
            x_grid = np.linspace(x_start + dx, x_end - dx, N)
        elif bc == 'neumann':
            if N < 2:
                raise ValueError("Need N >= 2 for Neumann BCs")
            dx = (x_end - x_start) / (N - 1)
            x_grid = np.linspace(x_start, x_end, N)
        else:
            raise ValueError(f"Unknown boundary condition: {bc}")

        self.N = N
        self.x_grid = x_grid
        self.dx = dx

        if bc_values is None:
            bc_values = {}

        # =================================================================
        # STEP 7: Finite difference operators (all BCs, up to 4th order)
        # =================================================================
        def _build_fd_ops(bc_type, dx_val, n_pts, bv):
            ops = {}
            if bc_type == 'periodic':
                ops[1] = lambda u: (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx_val)
                ops[2] = lambda u: (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx_val**2
                ops[3] = lambda u: (
                    np.roll(u, -2) - 2*np.roll(u, -1) + 2*np.roll(u, 1) - np.roll(u, 2)
                ) / (2 * dx_val**3)
                ops[4] = lambda u: (
                    np.roll(u, -2) - 4*np.roll(u, -1) + 6*u - 4*np.roll(u, 1) + np.roll(u, 2)
                ) / dx_val**4

            elif bc_type == 'dirichlet':
                lv, rv = bv

                def d1_dir(u):
                    d = np.empty_like(u, dtype=float)
                    if n_pts == 1:
                        d[0] = (rv - lv) / (2 * dx_val)
                    else:
                        d[0]  = (u[1]  - lv)   / (2 * dx_val)
                        d[-1] = (rv    - u[-2]) / (2 * dx_val)
                        if n_pts > 2:
                            d[1:-1] = (u[2:] - u[:-2]) / (2 * dx_val)
                    return d

                def d2_dir(u):
                    d = np.empty_like(u, dtype=float)
                    if n_pts == 1:
                        d[0] = (rv - 2*u[0] + lv) / dx_val**2
                    else:
                        d[0]  = (u[1]  - 2*u[0]  + lv)    / dx_val**2
                        d[-1] = (rv    - 2*u[-1]  + u[-2]) / dx_val**2
                        if n_pts > 2:
                            d[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx_val**2
                    return d

                def d3_dir(u):
                    # Odd-reflection ghost nodes: u[-1]=2*lv-u[1], u[N]=2*rv-u[-2]
                    d = np.empty_like(u, dtype=float)
                    gl = 2*lv - (u[1]  if n_pts > 1 else rv)
                    gr = 2*rv - (u[-2] if n_pts > 1 else lv)
                    if n_pts == 1:
                        d[0] = (-gl + 2*lv - 2*rv + gr) / (2*dx_val**3)
                    else:
                        u2_l = u[2] if n_pts > 2 else rv
                        u3_m = u[-3] if n_pts > 2 else lv
                        d[0]  = (-gl + 2*lv   - 2*u[1]  + u2_l) / (2*dx_val**3)
                        d[-1] = (-u3_m + 2*u[-2] - 2*rv   + gr)  / (2*dx_val**3)
                        if n_pts > 2:
                            u3_l = u[3] if n_pts > 3 else rv
                            u4_m = u[-4] if n_pts > 3 else lv
                            d[1]  = (-lv    + 2*u[0]  - 2*u[2]  + u3_l) / (2*dx_val**3)
                            d[-2] = (-u4_m  + 2*u[-3] - 2*u[-1] + rv)   / (2*dx_val**3)
                            if n_pts > 4:
                                d[2:-2] = (-u[:-4] + 2*u[1:-3] - 2*u[3:-1] + u[4:]) / (2*dx_val**3)
                    return d

                def d4_dir(u):
                    d = np.empty_like(u, dtype=float)
                    gl = 2*lv - (u[1]  if n_pts > 1 else rv)
                    gr = 2*rv - (u[-2] if n_pts > 1 else lv)
                    if n_pts == 1:
                        d[0] = (gl - 4*lv + 6*u[0] - 4*rv + gr) / dx_val**4
                    else:
                        u2_l = u[2]  if n_pts > 2 else rv
                        u3_m = u[-3] if n_pts > 2 else lv
                        d[0]  = (gl    - 4*lv   + 6*u[0]  - 4*u[1]  + u2_l)  / dx_val**4
                        d[-1] = (u3_m  - 4*u[-2]+ 6*u[-1] - 4*rv    + gr)    / dx_val**4
                        if n_pts > 2:
                            u3_l = u[3] if n_pts > 3 else rv
                            u4_m = u[-4] if n_pts > 3 else lv
                            d[1]  = (lv   - 4*u[0]  + 6*u[1]  - 4*u[2]  + u3_l) / dx_val**4
                            d[-2] = (u4_m - 4*u[-3] + 6*u[-2] - 4*u[-1] + rv)   / dx_val**4
                            if n_pts > 4:
                                d[2:-2] = (u[:-4] - 4*u[1:-3] + 6*u[2:-2] - 4*u[3:-1] + u[4:]) / dx_val**4
                    return d

                ops[1] = d1_dir
                ops[2] = d2_dir
                ops[3] = d3_dir
                ops[4] = d4_dir

            elif bc_type == 'neumann':
                ld, rd = bv

                def d1_neu(u):
                    d = np.empty_like(u, dtype=float)
                    d[0]  = ld
                    d[-1] = rd
                    if n_pts > 2:
                        d[1:-1] = (u[2:] - u[:-2]) / (2 * dx_val)
                    return d

                def d2_neu(u):
                    ghost_l = u[1]  - 2*dx_val*ld
                    ghost_r = u[-2] + 2*dx_val*rd
                    d = np.empty_like(u, dtype=float)
                    d[0]  = (u[1]    - 2*u[0]  + ghost_l) / dx_val**2
                    d[-1] = (ghost_r - 2*u[-1] + u[-2])   / dx_val**2
                    if n_pts > 2:
                        d[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx_val**2
                    return d

                def d3_neu(u):
                    ghost_l = u[1]  - 2*dx_val*ld
                    ghost_r = u[-2] + 2*dx_val*rd
                    d = np.empty_like(u, dtype=float)
                    if n_pts == 1:
                        d[0] = 0.0
                    else:
                        u2_l = u[2] if n_pts > 2 else ghost_r
                        u3_m = u[-3] if n_pts > 2 else ghost_l
                        d[0]  = (-ghost_l + 2*u[0]  - 2*u[1]   + u2_l) / (2*dx_val**3)
                        d[-1] = (-u3_m    + 2*u[-2] - 2*u[-1]  + ghost_r) / (2*dx_val**3)
                        if n_pts > 2:
                            u3_l = u[3] if n_pts > 3 else ghost_r
                            u4_m = u[-4] if n_pts > 3 else ghost_l
                            d[1]  = (-ghost_l + 2*u[0]  - 2*u[2]  + u3_l) / (2*dx_val**3)
                            d[-2] = (-u4_m    + 2*u[-3] - 2*u[-1] + ghost_r) / (2*dx_val**3)
                            if n_pts > 4:
                                d[2:-2] = (-u[:-4] + 2*u[1:-3] - 2*u[3:-1] + u[4:]) / (2*dx_val**3)
                    return d

                def d4_neu(u):
                    ghost_l = u[1]  - 2*dx_val*ld
                    ghost_r = u[-2] + 2*dx_val*rd
                    d = np.empty_like(u, dtype=float)
                    if n_pts == 1:
                        d[0] = 0.0
                    else:
                        u2_l = u[2]  if n_pts > 2 else ghost_r
                        u3_m = u[-3] if n_pts > 2 else ghost_l
                        d[0]  = (ghost_l - 4*u[0]  + 6*u[0]  - 4*u[1]  + u2_l)  / dx_val**4
                        d[-1] = (u3_m    - 4*u[-2] + 6*u[-1] - 4*u[-1] + ghost_r) / dx_val**4
                        if n_pts > 2:
                            u3_l = u[3] if n_pts > 3 else ghost_r
                            u4_m = u[-4] if n_pts > 3 else ghost_l
                            d[1]  = (ghost_l - 4*u[0]  + 6*u[1]  - 4*u[2]  + u3_l) / dx_val**4
                            d[-2] = (u4_m    - 4*u[-3] + 6*u[-2] - 4*u[-1] + ghost_r) / dx_val**4
                            if n_pts > 4:
                                d[2:-2] = (u[:-4] - 4*u[1:-3] + 6*u[2:-2] - 4*u[3:-1] + u[4:]) / dx_val**4
                    return d

                ops[1] = d1_neu
                ops[2] = d2_neu
                ops[3] = d3_neu
                ops[4] = d4_neu

            return ops

        fd_ops = {}
        for name in field_vars:
            bv = bc_values.get(name, (0.0, 0.0))
            fd_ops[name] = _build_fd_ops(bc, dx, N, bv)

        # =================================================================
        # STEP 8: State-vector layout for nth-order-in-time fields
        # =================================================================
        # A field u_k of time-order m_k occupies m_k*N consecutive entries:
        #   [u_k^(0), u_k^(1), ..., u_k^(m_k-1)]  each of length N.
        #
        # state_offsets_map[name] = flat index where u_k^(0) starts.
        # time_orders_map[name]   = m_k.

        time_orders_map = {}
        state_offsets_map = {}
        flat_offset = 0
        for name in field_vars:
            m = max(max_time_order.get(name, 1), 1)
            time_orders_map[name] = m
            state_offsets_map[name] = flat_offset
            flat_offset += m * N
        total_states = flat_offset

        self.time_orders = time_orders_map
        self.state_offsets = state_offsets_map
        self.order = total_states

        # =================================================================
        # STEP 9: Lambdify RHS and build the semi-discrete system function
        # =================================================================
        # Lambdified argument order:
        #   [x, t,  S0_u1, S1_u1, ..., S0_u2, ...,
        #    Lt1_u1, ..., Lt{m-1}_u1, ...,    <- lower time-deriv states
        #    PHFN0, ...]

        all_syms = [x_sym, t_sym]
        for name in field_vars:
            max_so = max_spatial_order.get(name, 0)
            for k in range(max_so + 1):
                all_syms.append(var_symbols[name][k])

        # Add symbols for lower-order time derivatives (for cross-field coupling
        # and sub-highest-order references in equations).
        time_state_syms = {}   # {(name, k): Symbol}
        for name in field_vars:
            m = time_orders_map[name]
            for k in range(1, m):
                safe_lt = f'Lt{k}_{name}'
                sym = sp.Symbol(safe_lt)
                local_dict[safe_lt] = sym
                time_state_syms[(name, k)] = sym
                all_syms.append(sym)

        ph_list = list(func_placeholders.keys())
        all_syms.extend(ph_list)

        # Substitute time_state_syms into rhs expressions where St{k} symbols
        # appear for k < m (they are state variables, not inputs from lambdify).
        # Map St{k}_name -> Lt{k}_name for k < m so lambdify sees the right syms.
        substitution_map = {}
        for name in field_vars:
            m = time_orders_map[name]
            for k in range(1, m):
                old_sym = time_deriv_syms[name][k]
                new_sym = time_state_syms[(name, k)]
                substitution_map[old_sym] = new_sym

        lambdified_rhs = []
        for target_var, target_tord, rhs_expr in parsed_equations:
            rhs_subst = rhs_expr.subs(substitution_map) if substitution_map else rhs_expr
            fn = sp.lambdify(all_syms, rhs_subst, modules='numpy')
            lambdified_rhs.append((target_var, target_tord, fn))

        def system(t_val, y_flat):
            dydt = np.zeros_like(y_flat)

            # Extract all field time-derivative states from flat vector
            # field_states[name][j] = u_k^{(j)}  (array of length N)
            field_states = {}
            for name in field_vars:
                m = time_orders_map[name]
                off = state_offsets_map[name]
                field_states[name] = [
                    y_flat[off + j*N : off + (j+1)*N]
                    for j in range(m)
                ]

            # Chain rules: d(u^{(j)})/dt = u^{(j+1)}  for j = 0 .. m-2
            for name in field_vars:
                m = time_orders_map[name]
                off = state_offsets_map[name]
                for j in range(m - 1):
                    dydt[off + j*N : off + (j+1)*N] = field_states[name][j + 1]

            # Spatial derivatives of the zeroth state (the physical field)
            spatial = {}
            for name in field_vars:
                u_field = field_states[name][0]
                spatial[name] = {0: u_field}
                for k in range(1, max_spatial_order.get(name, 0) + 1):
                    if k not in fd_ops[name]:
                        raise ValueError(
                            f"Spatial derivative order {k} not supported "
                            f"for bc='{bc}'"
                        )
                    spatial[name][k] = fd_ops[name][k](u_field)

            # Evaluate placeholder functions
            ph_vals = []
            for ph_sym in ph_list:
                fname, arg_names = func_placeholders[ph_sym]
                fargs = []
                for a in arg_names:
                    if a == 't':
                        fargs.append(t_val)
                    elif a == 'x':
                        fargs.append(x_grid)
                    elif a in field_states:
                        fargs.append(field_states[a][0])
                    else:
                        raise ValueError(f"Unknown argument '{a}' in {fname}(...)")
                ph_vals.append(function_dict[fname](*fargs))

            # Build argument list for lambdified RHS
            base_args = [x_grid, t_val]
            for name in field_vars:
                max_so = max_spatial_order.get(name, 0)
                for k in range(max_so + 1):
                    base_args.append(spatial[name][k])
            # Lower time-derivative states (for equations that couple through them)
            for name in field_vars:
                m = time_orders_map[name]
                for k in range(1, m):
                    base_args.append(field_states[name][k])
            base_args.extend(ph_vals)

            # Fill the highest time-derivative slot for each field
            for target_var, target_tord, fn in lambdified_rhs:
                rhs_val = fn(*base_args)
                off = state_offsets_map[target_var]
                m   = time_orders_map[target_var]
                dydt[off + (m-1)*N : off + m*N] = rhs_val

            return dydt

        self.ODE = system

        return system, field_vars, self.order


    def Solve(self, ODE, initial_conditions, t_span, t_eval, method='RK45', **kwargs):
        """
        Numerically integrate the parsed PDE system.

        Parameters
        ----------
        ODE : callable
            The compiled f(t, y) function from Parse.
        
        initial_conditions : array-like
            Initial field values. Can be:
            - 1D array of length N (single field variable)
            - 1D array of length n_fields * N (concatenated: [u1, u2, ...])
            - 2D array of shape (n_fields, N) — will be flattened

        t_span : tuple of (float, float)
            (t_start, t_end).

        t_eval : array-like
            Times at which to store the solution.

        method : str, optional
            Integration method. Default 'RK45'.
            Use 'Radau' or 'BDF' for stiff problems (e.g. diffusion).

        **kwargs
            Additional arguments for solve_ivp (rtol, atol, max_step, etc.).

        Returns
        -------
        sol : OdeSolution
            Solution object. sol.y has shape (n_fields * N, n_times).
        """
        if ODE is None:
            raise RuntimeError("Call Parse first.")

        ic = np.asarray(initial_conditions, dtype=float)
        if ic.ndim == 2:
            ic = ic.ravel()

        expected = self.order   # = sum of (time_order_k * N) for all fields
        if len(ic) != expected:
            parts = ", ".join(
                f"{name}: {self.time_orders.get(name,1)}x{self.N}"
                for name in self.field_vars
            )
            raise ValueError(
                f"Expected {expected} initial values "
                f"({parts}), got {len(ic)}. "
                f"For a field u_k of time-order m, provide m arrays of "
                f"length N: [u_k(x,0), du_k/dt(x,0), ..., d^(m-1)u_k/dt^(m-1)(x,0)]."
            )

        return solve_ivp(
            ODE, t_span, ic,
            t_eval=t_eval, method=method, dense_output=True, **kwargs
        )

    def get_field(self, sol, var_name=None, var_idx=0, time_deriv=0):
        """
        Extract a single field (or one of its time-derivative states) from
        the solution.

        Parameters
        ----------
        sol : OdeSolution
            Result from Solve.
        var_name : str, optional
            Field name, e.g. 'u1'. Overrides var_idx if given.
        var_idx : int, optional
            Index into self.field_vars. Default 0. Ignored when var_name given.
        time_deriv : int, optional
            Which time-derivative level to extract. Default 0 (the field itself).
              0  -> u_k(x, t)
              1  -> du_k/dt(x, t)   (only for 2nd-order-in-time or higher fields)
              k  -> d^k u_k / dt^k  (must be < time_order of that field)

        Returns
        -------
        ndarray of shape (N, n_times)
            The requested field or derivative at all stored times.

        Raises
        ------
        IndexError
            If time_deriv >= time_order of the requested field.
        """
        if var_name is not None:
            var_idx = self.field_vars.index(var_name)
        name = self.field_vars[var_idx]
        m   = self.time_orders.get(name, 1)
        off = self.state_offsets.get(name, var_idx * self.N)
        if time_deriv >= m:
            raise IndexError(
                f"Field '{name}' is order {m} in time; "
                f"time_deriv={time_deriv} is out of range (max {m-1})."
            )
        start = off + time_deriv * self.N
        return sol.y[start : start + self.N, :]

    @staticmethod
    def _parse_single(pde_strings, function_dict, x_span, N, bc, bc_values):
        """Worker for parallel parsing."""
        pde = AutoPDE()
        return pde.Parse(pde_strings, function_dict, x_span, N, bc, bc_values)

    @staticmethod
    def Parallel_Parse(inputs: list, n_jobs: int = -1, backend: str = "loky") -> list:
        """
        Parse multiple PDE systems in parallel.

        Parameters
        ----------
        inputs : list of tuples
            Each: (pde_strings, function_dict, x_span, N[, bc[, bc_values]]).
            bc defaults to 'periodic', bc_values to None.

            NOTE: Use backend="threading" if function_dict contains lambdas.

        n_jobs : int, optional
            Parallel workers. -1 uses all cores.

        backend : str, optional
            Joblib backend. Default "loky".

        Returns
        -------
        list of (list of ODEs, list of field_vars, list of orders)
            Three lists containing ODE functions, field variable lists, and
            order values for each parsed system, in the same order as inputs.
        """
        def _unpack(args):
            pde_s, fd, xs, n = args[0], args[1], args[2], args[3]
            bc_arg = args[4] if len(args) > 4 else 'periodic'
            bv_arg = args[5] if len(args) > 5 else None
            return pde_s, fd, xs, n, bc_arg, bv_arg

        raw_results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(AutoPDE._parse_single)(*_unpack(inp))
            for inp in tqdm(inputs, desc="Parsing PDEs")
        )
        results = list(map(list, zip(*raw_results)))
        
        return results

    @staticmethod
    def _solve_single(pde, ODE, initial_conditions, t_span, t_eval, method, kwargs):
        """Worker function for parallel Solve. Returns solution object."""
        return pde.Solve(ODE, initial_conditions, t_span, t_eval, method=method, **kwargs)

    @staticmethod
    def Parallel_Solve(pdes: list, odes: list, inputs: list, n_jobs: int = -1, backend: str = "loky") -> list:
        """
        Solve multiple PDE systems with corresponding initial conditions in parallel.

        This is useful for parameter sweeps, ensemble simulations, studying
        different initial profiles, or Monte Carlo analysis of PDE systems.

        Parameters
        ----------
        pdes : list of AutoPDE
            List of configured AutoPDE instances (one per solve).
            
        odes : list of callable
            List of ODE functions from Parse (one per solve).

        inputs : list of tuples
            Each element is a tuple of arguments for the Solve method:
            (initial_conditions, t_span, t_eval[, method[, **kwargs]])

            Minimal form: (ic, t_span, t_eval)
            With method: (ic, t_span, t_eval, method)
            With kwargs: (ic, t_span, t_eval, method, kwargs_dict)

            Examples:
                # Different initial profiles
                [([u0_profile1], (0, 1), t_eval),
                 ([u0_profile2], (0, 1), t_eval)]

                # Different time spans
                [([u0], (0, 1), np.linspace(0, 1, 100)),
                 ([u0], (0, 2), np.linspace(0, 2, 200))]

                # Custom solver options (e.g., for stiff problems)
                [([u0], (0, 1), t_eval, 'Radau', {'rtol': 1e-6}),
                 ([u0], (0, 1), t_eval, 'BDF', {'atol': 1e-8})]

        n_jobs : int, optional
            Number of parallel workers. -1 (default) uses all available cores.

        backend : str, optional
            Joblib parallel backend. Default is "loky" (multiprocessing).
            Use "threading" for thread-based parallelism.

        Returns
        -------
        list of OdeSolution
            Solution objects in the same order as inputs.

        Example
        -------
        >>> # First parse multiple PDE systems
        >>> inputs_parse = [
        ...     ("u1_t - 0.01*u1_xx = 0", {}, (0, 1), 100, 'periodic'),
        ...     ("u1_t - 0.01*u1_xx = 0", {}, (0, 1), 100, 'periodic'),
        ...     ("u1_t - 0.01*u1_xx = 0", {}, (0, 1), 100, 'periodic')
        ... ]
        >>> odes, field_vars_list, orders = AutoPDE.Parallel_Parse(inputs_parse, n_jobs=3)
        >>> 
        >>> # Create PDE instances (needed to access x_grid, etc.)
        >>> pdes = []
        >>> for i in range(3):
        ...     pde = AutoPDE()
        ...     pde.Parse(*inputs_parse[i])
        ...     pdes.append(pde)
        >>> 
        >>> # Different initial profiles
        >>> x = pdes[0].x_grid
        >>> u0_1 = np.sin(2*np.pi*x)
        >>> u0_2 = np.sin(4*np.pi*x)
        >>> u0_3 = np.exp(-((x-0.5)/0.1)**2)
        >>> t_eval = np.linspace(0, 0.5, 100)
        >>> 
        >>> # Solve all three systems
        >>> inputs_solve = [(u0_1, (0, 0.5), t_eval),
        ...                 (u0_2, (0, 0.5), t_eval),
        ...                 (u0_3, (0, 0.5), t_eval)]
        >>> solutions = AutoPDE.Parallel_Solve(pdes, odes, inputs_solve, n_jobs=3)
        """
        def _unpack(args):
            ic = args[0]
            t_span = args[1]
            t_eval = args[2]
            method = args[3] if len(args) > 3 else 'RK45'
            kwargs = args[4] if len(args) > 4 else {}
            return ic, t_span, t_eval, method, kwargs

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(AutoPDE._solve_single)(pde, ode, *_unpack(inp))
            for pde, ode, inp in zip(tqdm(pdes, desc="Solving PDEs"), odes, inputs)
        )
        return results

    
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    
    
    # =========================================================================
    # Test Parallel_Parse with many Lorenz 63 instances
    # =========================================================================
    # Lorenz 63:
    #   x1' = sigma*(x2 - x1)
    #   x2' = x1*(rho - x3) - x2
    #   x3' = x1*x2 - beta*x3
    #
    # We sweep over different (sigma, rho, beta) parameter combinations and
    # parse + solve each instance in parallel.

    n_instances = 50
    np.random.seed(0)

    # Random parameter variations around the classic values
    sigmas = 10.0 + 2.0 * np.random.randn(n_instances)
    rhos   = 28.0 + 4.0 * np.random.randn(n_instances)
    betas  = 8/3  + 0.5 * np.random.randn(n_instances)

    # -- Build inputs for Parallel_Parse --
    # Each Lorenz 63 system uses three "constant" coefficient functions.
    # We use the threading backend so lambdas are supported.
    lorenz_eqs = [
        "x1' - s(t)*(x2 - x1) = 0",
        "x2' - x1*r(t) + x1*x3 + x2 = 0",
        "x3' - x1*x2 + b(t)*x3 = 0",
    ]
    inputs = [
        (
            lorenz_eqs,
            {
                "s": lambda t, _s=s: _s,
                "r": lambda t, _r=r: _r,
                "b": lambda t, _b=b: _b,
            },
        )
        for s, r, b in zip(sigmas, rhos, betas)
    ]

    # --- Parallel parse ---
    print(f"Parallel-parsing {n_instances} Lorenz 63 instances ...")
    start_parse = time.perf_counter()
    ODE = AutoODE()
    odes = ODE.Parallel_Parse(inputs, n_jobs=-1, backend="threading")[0]
    t_parse = time.perf_counter() - start_parse
    print(f"  Parallel parse: {t_parse:.3f} s")
    
    # --- Solve all instances using Parallel_Solve ---
    t_span = (0, 25)
    t_eval = np.linspace(0, 25, 2000)
    ic = [1.0, 1.0, 1.0]  # same IC for all
    
    # Build inputs for Parallel_Solve - solve all parsed systems with same IC
    print(f"\nParallel-solving {n_instances} systems ...")
    solve_inputs = [(ic, t_span, t_eval) for _ in range(n_instances)]
    
    start_solve = time.perf_counter()
    # Note: We need to solve each ODE instance separately, so we'll still loop
    # but demonstrate Parallel_Solve concept with the first ODE instance
    solutions = ODE.Parallel_Solve(odes, solve_inputs, n_jobs=-1, backend="threading")
    ODE.Solve
    t_solve = time.perf_counter() - start_solve
    print(f"  Solve time: {t_solve:.3f} s")

    # --- Plot a selection of trajectories ---
    fig = plt.figure(figsize=(14, 5))

    # 3D phase portrait for a few instances
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    for i in range(min(8, n_instances)):
        sol = solutions[i]
        ax1.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.4, alpha=0.8,
                 label=f'σ={sigmas[i]:.1f} ρ={rhos[i]:.1f}')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('x3')
    ax1.set_title(f'Lorenz 63 — {min(8, n_instances)} instances')
    ax1.legend(fontsize=6, loc='upper left')

    # Time series of x1 for all instances (spaghetti plot)
    ax2 = fig.add_subplot(1, 2, 2)
    for i, sol in enumerate(solutions):
        ax2.plot(sol.t, sol.y[0], lw=0.3, alpha=0.6)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x1')
    ax2.set_title(f'Lorenz 63 — x1(t) for {n_instances} instances')

    plt.tight_layout()

    # =========================================================================
    # Double Slit Experiment — Beam Propagation Method (AutoPDE)
    # =========================================================================
    # The paraxial wave equation (Fresnel diffraction) in 2D reduces to a
    # 1D Schrödinger-like equation along the propagation axis z ("time"):
    #
    #   ∂E/∂z = i·α · ∂²E/∂x²,    α = 1/(2k₀)
    #
    # Splitting the complex field E = u1 + i·u2:
    #   u1_t + α·u2_xx = 0
    #   u2_t − α·u1_xx = 0
    #
    # Initial field at z = 0: the double-slit aperture (two Gaussians).
    # As z increases the interference pattern develops — the classic fringes.

    print("\n" + "=" * 60)
    print("Double Slit Experiment  (Beam Propagation Method)")
    print("=" * 60)

    # --- Parameters (normalised units) ---
    k0         = 50.0                       # wavenumber
    alpha      = 1.0 / (2.0 * k0)          # diffraction coefficient (0.01)
    slit_sep   = 4.0                        # centre-to-centre slit distance
    slit_w     = 0.3                        # Gaussian width of each slit
    L          = 20.0                       # half-width of transverse domain
    N_pde      = 600                        # spatial grid points
    z_max      = 80.0                       # total propagation distance
    n_z        = 300                        # stored z-steps

    # --- Parse the coupled PDE system ---
    pde = AutoPDE()
    pde_ode, field_vars, order = pde.Parse(
        [f"u1_t + {alpha}*u2_xx = 0",
         f"u2_t - {alpha}*u1_xx = 0"],
        {},
        (-L, L), N_pde, bc='dirichlet',
    )
    x = pde.x_grid

    # --- Initial condition: two Gaussian slits ---
    u1_0 = (np.exp(-((x - slit_sep / 2) / slit_w) ** 2) +
            np.exp(-((x + slit_sep / 2) / slit_w) ** 2))
    u2_0 = np.zeros(N_pde)
    ic_pde = np.concatenate([u1_0, u2_0])

    # --- Solve (propagate in z) ---
    z_eval = np.linspace(0, z_max, n_z)
    print(f"Propagating {N_pde}-point field over z = [0, {z_max}] ...")
    start_pde_solve = time.perf_counter()
    sol_pde = pde.Solve(pde_ode, ic_pde, (0, z_max), z_eval,
                        method='DOP853', rtol=1e-8, atol=1e-10)
    t_solve = time.perf_counter() - start_pde_solve
    print(f"  Solve time: {t_solve:.2f} s")

    # --- Extract intensity I = |E|² = u1² + u2² ---
    u1 = pde.get_field(sol_pde, 'u1')       # (N, n_z)
    u2 = pde.get_field(sol_pde, 'u2')
    intensity = u1 ** 2 + u2 ** 2

    # --- Visualisation ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (a) Aperture at z = 0
    axes[0].fill_between(x, 0, u1_0, color='steelblue', alpha=0.6)
    axes[0].plot(x, u1_0, 'k-', lw=1)
    axes[0].axvline( slit_sep / 2, color='gray', ls='--', alpha=0.5)
    axes[0].axvline(-slit_sep / 2, color='gray', ls='--', alpha=0.5)
    axes[0].set_xlabel('x  (transverse)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Double-Slit Aperture  (z = 0)')
    axes[0].set_xlim(-3 * slit_sep, 3 * slit_sep)

    # (b) Intensity I(x, z) — propagation map
    im = axes[1].imshow(
        intensity.T, aspect='auto', origin='lower',
        extent=[-L, L, 0, z_max], cmap='inferno',
        vmin=0, vmax=np.percentile(intensity, 99),
    )
    axes[1].set_xlabel('x  (transverse)')
    axes[1].set_ylabel('z  (propagation)')
    axes[1].set_title('Intensity  |E|²(x, z)')
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # (c) Interference fringes on the far screen
    I_screen = intensity[:, -1]
    axes[2].plot(x, I_screen, 'r-', lw=1)
    axes[2].fill_between(x, 0, I_screen, color='salmon', alpha=0.3)
    axes[2].set_xlabel('x  (transverse)')
    axes[2].set_ylabel('Intensity  |E|²')
    axes[2].set_title(f'Interference Pattern  (z = {z_max})')
    axes[2].set_xlim(-L, L)

    fig.suptitle('Double Slit Experiment — Paraxial Wave Propagation (AutoPDE)',
                 fontsize=13, y=0.98)
    plt.tight_layout()

    # =========================================================================
    # Wave Equation — AutoPDE  (2nd order in both time AND space)
    # =========================================================================
    # The 1-D wave equation:
    #
    #   u_tt = c² · u_xx
    #
    # is 2nd-order in time (u_tt) and 2nd-order in space (u_xx).
    # AutoPDE now handles this natively: Parse detects "u1_tt" and builds the
    # extended state vector  [u1, u1_t]  of length 2*N.
    #
    # Exact solution for the chosen IC (standing-wave initial displacement,
    # zero initial velocity, Dirichlet BCs):
    #   u(x, t) = sin(n*π*x/L) · cos(n*π*c*t/L)
    #
    # We verify numerics against this exact solution.

    print("\n" + "=" * 60)
    print("Wave Equation  u_tt = c²·u_xx  (AutoPDE, 2nd order in time)")
    print("=" * 60)

    # --- Parameters ---
    c_wave  = 2.0          # wave speed
    L_wave  = 1.0          # domain [0, L]
    N_wave  = 200          # spatial grid points (interior, Dirichlet)
    T_wave  = 2.0          # integrate to t = T
    n_mode  = 2            # standing wave mode number
    n_t     = 500          # stored time points

    # --- Parse ---
    pde_wave = AutoPDE()
    ode_wave, fvars_wave, ord_wave = pde_wave.Parse(
        f"u1_tt - {c_wave**2}*u1_xx = 0",   # 2nd-order in time!
        {},
        (0.0, L_wave), N_wave,
        bc='dirichlet',                        # u1 = 0 at x = 0 and x = L
    )
    x_wave = pde_wave.x_grid

    print(f"  State dimension: {ord_wave}  (= 2 × N = 2 × {N_wave})")
    print(f"  time_orders: {pde_wave.time_orders}")

    # --- Initial conditions ---
    # u1(x, 0) = sin(n·π·x/L)    <- initial displacement
    # u1_t(x, 0) = 0              <- zero initial velocity
    u1_0   = np.sin(n_mode * np.pi * x_wave / L_wave)
    u1t_0  = np.zeros(N_wave)
    ic_wave = np.concatenate([u1_0, u1t_0])   # length 2*N

    # --- Solve ---
    t_eval_wave = np.linspace(0, T_wave, n_t)
    print(f"  Integrating over t ∈ [0, {T_wave}] ...")
    t0 = time.perf_counter()
    sol_wave = pde_wave.Solve(
        ode_wave, ic_wave, (0, T_wave), t_eval_wave,
        method='DOP853', rtol=1e-9, atol=1e-11,
    )
    print(f"  Solve time: {time.perf_counter() - t0:.3f} s")

    # --- Extract fields ---
    u_num  = pde_wave.get_field(sol_wave, 'u1', time_deriv=0)  # displacement
    ut_num = pde_wave.get_field(sol_wave, 'u1', time_deriv=1)  # velocity

    # --- Exact solution ---
    omega = n_mode * np.pi * c_wave / L_wave
    u_exact = (np.sin(n_mode * np.pi * x_wave[:, None] / L_wave)
               * np.cos(omega * t_eval_wave[None, :]))

    max_err = np.max(np.abs(u_num - u_exact))
    print(f"  Max pointwise error vs exact: {max_err:.2e}")

    # --- Visualisation ---
    fig_wave, axes_wave = plt.subplots(1, 3, figsize=(17, 4))

    # (a) Space-time map of displacement
    im_w = axes_wave[0].imshow(
        u_num, aspect='auto', origin='lower',
        extent=[0, T_wave, 0, L_wave], cmap='RdBu_r',
        vmin=-1.1, vmax=1.1,
    )
    axes_wave[0].set_xlabel('t')
    axes_wave[0].set_ylabel('x')
    axes_wave[0].set_title(f'u(x, t)  — mode {n_mode},  c={c_wave}')
    plt.colorbar(im_w, ax=axes_wave[0], shrink=0.85)

    # (b) Several snapshots vs exact
    snap_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(snap_times)))
    for col, ts in zip(colors, snap_times):
        ti = np.argmin(np.abs(t_eval_wave - ts))
        axes_wave[1].plot(x_wave, u_num[:, ti], color=col, lw=2,
                          label=f't={ts:.2f}')
        axes_wave[1].plot(x_wave, u_exact[:, ti], color=col,
                          lw=1, ls='--', alpha=0.7)
    axes_wave[1].set_xlabel('x')
    axes_wave[1].set_ylabel('u(x, t)')
    axes_wave[1].set_title('Snapshots: numerical (solid) vs exact (dashed)')
    axes_wave[1].legend(fontsize=8)

    # (c) Pointwise error over time
    err_over_time = np.max(np.abs(u_num - u_exact), axis=0)
    axes_wave[2].semilogy(t_eval_wave, err_over_time, 'k-', lw=1)
    axes_wave[2].set_xlabel('t')
    axes_wave[2].set_ylabel('Max |error|')
    axes_wave[2].set_title('Max pointwise error vs exact solution')
    axes_wave[2].grid(True, which='both', alpha=0.3)

    fig_wave.suptitle(
        f'Wave Equation  u_tt = {c_wave**2}·u_xx  '
        f'(AutoPDE, 2nd-order in time, N={N_wave})',
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()
