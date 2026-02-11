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
        self.ODE = None        # Will hold the compiled f(t, y) function
        self.order = 0         # Total number of first-order state variables
        self.state_vars = []   # [(var_name, max_derivative_order), ...]
        
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
        self.order = total_states

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
        self.ODE = system
        self.state_vars = [(name, var_max_order[name]) for name in var_names]

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
        import re

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
        self.Parse(equations, expanded_funcs)

    def Solve(self, initial_conditions, t_span, t_eval, method='RK45', **kwargs):
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
        if self.ODE is None:
            raise RuntimeError("Call Parse first.")
        sol = solve_ivp(
            self.ODE, t_span, initial_conditions,
            t_eval=t_eval, method=method, dense_output=True, **kwargs
        )
        return sol
    
    @staticmethod
    def _parse_single(ode_strings, function_dict):
        """Worker function for parallel Parse. Returns a configured AutoODE."""
        ode = AutoODE()
        ode.Parse(ode_strings, function_dict)
        return ode

    @staticmethod
    def _parse_local_single(ode_string, function_dict, dim, mod):
        """Worker function for parallel Parse_Local. Returns a configured AutoODE."""
        ode = AutoODE()
        ode.Parse_Local(ode_string, function_dict, dim, mod)
        return ode

    @staticmethod
    def Parallel_Parse(inputs, n_jobs=-1, backend="loky"):
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
        list of AutoODE
            One configured AutoODE instance per input, in the same order.

        Example
        -------
        >>> results = AutoODE.Parallel_Parse([
        ...     (["x1'' + x1 = 0"], {}),
        ...     (["x1' - x1 = 0"],  {}),
        ... ], n_jobs=2)
        """
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(AutoODE._parse_single)(ode_strings, function_dict)
            for ode_strings, function_dict in tqdm(inputs, desc="Parsing ODEs")
        )
        return results

    @staticmethod
    def Parallel_Parse_Local(inputs, n_jobs=-1, backend="threading"):
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
        list of AutoODE
            One configured AutoODE instance per input, in the same order.

        Example
        -------
        >>> results = AutoODE.Parallel_Parse_Local([
        ...     ("x_i' + x_i = 0", {}, 10),
        ...     ("x_i' + x_i = 0", {}, 20, False),
        ... ], n_jobs=2)
        """
        def _unpack(args):
            if len(args) == 3:
                return args[0], args[1], args[2], True
            return args[0], args[1], args[2], args[3]

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(AutoODE._parse_local_single)(*_unpack(inp))
            for inp in tqdm(inputs, desc="Parsing local ODEs")
        )
        return results

    
class AutoPDE:
    """
    Automatic 1D PDE parser and solver (method of lines).

    Converts human-readable PDE strings into a spatially discretized ODE
    system and solves them numerically using finite differences + solve_ivp.

    Workflow
    --------
    1. Write your PDEs as strings set equal to zero, e.g.:
           "u1_t - 0.01*u1_xx = 0"
       Field variables are u1, u2, ...; t is time, x is space.
       Time derivative: u1_t
       Spatial derivatives: u1_x, u1_xx, u1_xxx, u1_xxxx

    2. Pass coefficient/forcing functions in a dictionary, e.g.:
           {"f": lambda x, t: np.sin(np.pi * x)}

    3. Call Parse with spatial domain, grid size, and boundary conditions.

    4. Call Solve with initial conditions to integrate.

    Attributes
    ----------
    ODE : callable or None
        Compiled f(t, y) function for solve_ivp.
    order : int
        Total state dimension (N * n_fields).
    field_vars : list of str
        Detected field variable names, e.g. ['u1', 'u2'].
    N : int
        Number of spatial grid points.
    x_grid : ndarray
        Spatial grid coordinates.
    dx : float
        Grid spacing.

    Example
    -------
    >>> pde = AutoPDE()
    >>> pde.Parse("u1_t - 0.01*u1_xx = 0", {}, (0, 1), 100, bc='dirichlet')
    >>> u0 = np.sin(np.pi * pde.x_grid)
    >>> sol = pde.Solve(u0, (0, 0.5), np.linspace(0, 0.5, 100))
    """

    def __init__(self):
        self.ODE = None
        self.order = 0
        self.field_vars = []
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

            Derivative notation:
                - u1_t      = du1/dt       (time derivative)
                - u1_x      = du1/dx       (first spatial derivative)
                - u1_xx     = d2u1/dx2     (second spatial derivative)
                - u1_xxx    etc.           (third and higher)

            Function calls:
                - f(x, t), g(x, t, u1), etc. must appear in function_dict.

            Examples:
                Heat equation:
                    "u1_t - 0.01*u1_xx = 0"

                Burgers' equation:
                    "u1_t + u1*u1_x - 0.01*u1_xx = 0"

                Coupled reaction-diffusion:
                    ["u1_t - 0.1*u1_xx + u1*u2 = 0",
                     "u2_t - 0.2*u2_xx - u1*u2 = 0"]

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
                         Supports spatial derivatives up to 2nd order.

            'neumann':   Fixed spatial derivatives at boundaries. N points
                         including endpoints.
                         Supports spatial derivatives up to 2nd order.

        bc_values : dict, optional
            Boundary values per variable: {var_name: (left, right)}.
            For 'dirichlet': field values at boundaries.
            For 'neumann': spatial derivatives at boundaries.
            Defaults to (0.0, 0.0) for each variable.
            Ignored for 'periodic'.

        Raises
        ------
        ValueError
            If an equation has no time derivative, or if the time derivative
            cannot be algebraically isolated, or if an unsupported spatial
            derivative order is requested.

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
        # STEP 2: Detect field variables and their derivative orders
        # =================================================================
        # Matches: u1, u1_t, u1_x, u1_xx, u1_xxx, etc.
        all_tokens = set()
        for s in processed_strings:
            for m in re.finditer(
                r'(?<![a-zA-Z_\d])u(\d+)(?:_(t|x+))?(?![a-zA-Z\d])', s
            ):
                name = f'u{m.group(1)}'
                deriv_type = m.group(2)
                all_tokens.add((name, deriv_type))

        field_var_set = set()
        max_spatial_order = {}

        for name, deriv_type in all_tokens:
            field_var_set.add(name)
            if deriv_type is not None and deriv_type != 't':
                order = len(deriv_type)
                max_spatial_order[name] = max(
                    max_spatial_order.get(name, 0), order
                )
            else:
                max_spatial_order.setdefault(name, 0)

        field_vars = sorted(field_var_set, key=lambda n: int(n[1:]))
        self.field_vars = field_vars
        n_fields = len(field_vars)

        # =================================================================
        # STEP 3: Create sympy symbols for each field and derivative
        # =================================================================
        # u1       -> S0_u1   (field value)
        # u1_x     -> S1_u1   (first spatial derivative)
        # u1_xx    -> S2_u1   (second spatial derivative)
        # u1_t     -> St_u1   (time derivative — what we solve for)
        var_symbols = {}
        time_deriv_syms = {}

        for name in field_vars:
            var_symbols[name] = {}
            max_o = max_spatial_order.get(name, 0)
            for k in range(max_o + 1):
                safe = f'S{k}_{name}'
                var_symbols[name][k] = sp.Symbol(safe)
                local_dict[safe] = var_symbols[name][k]

            safe_t = f'St_{name}'
            time_deriv_syms[name] = sp.Symbol(safe_t)
            local_dict[safe_t] = time_deriv_syms[name]

        # =================================================================
        # STEP 4: Preprocess strings — convert PDE notation to safe names
        # =================================================================
        # "u1_xx - u1" becomes "S2_u1 - S0_u1"
        sorted_vars = sorted(field_vars, key=len, reverse=True)

        def preprocess(s):
            """Replace u1_xx etc. with safe sympy names like S2_u1."""
            result = []
            i = 0
            while i < len(s):
                matched = False
                for name in sorted_vars:
                    if s[i:i + len(name)] == name:
                        # Word boundary check
                        if i > 0 and (s[i - 1].isalnum() or s[i - 1] == '_'):
                            continue

                        j = i + len(name)

                        # Check for derivative subscript (_t or _x+)
                        if j < len(s) and s[j] == '_':
                            k = j + 1
                            # Time derivative
                            if k < len(s) and s[k] == 't':
                                end = k + 1
                                if end >= len(s) or not (
                                    s[end].isalnum() or s[end] == '_'
                                ):
                                    result.append(f'St_{name}')
                                    i = end
                                    matched = True
                                    break
                            # Spatial derivative
                            elif k < len(s) and s[k] == 'x':
                                n_x = 0
                                while k < len(s) and s[k] == 'x':
                                    n_x += 1
                                    k += 1
                                if k >= len(s) or not (
                                    s[k].isalnum() or s[k] == '_'
                                ):
                                    result.append(f'S{n_x}_{name}')
                                    i = k
                                    matched = True
                                    break

                        # Variable without subscript
                        if j >= len(s) or not (
                            s[j].isalnum() or s[j] == '_'
                        ):
                            result.append(f'S0_{name}')
                            i = j
                            matched = True
                            break

                if not matched:
                    result.append(s[i])
                    i += 1
            return ''.join(result)

        # =================================================================
        # STEP 5: Parse each equation and solve for the time derivative
        # =================================================================
        parsed_equations = []
        for s in processed_strings:
            s = preprocess(s)
            expr = parse_expr(s, local_dict=local_dict)

            target_var = None
            for name in field_vars:
                if expr.has(time_deriv_syms[name]):
                    target_var = name
                    break

            if target_var is None:
                raise ValueError(f"No time derivative (u_t) found in: {s}")

            solved = sp.solve(expr, time_deriv_syms[target_var])
            if not solved:
                raise ValueError(
                    f"Cannot isolate {time_deriv_syms[target_var]} in: {s}"
                )

            parsed_equations.append((target_var, solved[0]))

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
        self.order = N * n_fields

        if bc_values is None:
            bc_values = {}

        # =================================================================
        # STEP 7: Build finite difference operators
        # =================================================================
        # Central finite differences for spatial derivatives.
        # Periodic: np.roll handles wrap-around.
        # Dirichlet/Neumann: ghost nodes at boundaries.

        def _build_fd_ops(bc_type, dx_val, n_pts, bv):
            """Build spatial derivative operators for one field variable."""
            ops = {}
            if bc_type == 'periodic':
                ops[1] = lambda u: (
                    np.roll(u, -1) - np.roll(u, 1)
                ) / (2 * dx_val)
                ops[2] = lambda u: (
                    np.roll(u, -1) - 2 * u + np.roll(u, 1)
                ) / dx_val**2
                ops[3] = lambda u: (
                    np.roll(u, -2) - 2 * np.roll(u, -1)
                    + 2 * np.roll(u, 1) - np.roll(u, 2)
                ) / (2 * dx_val**3)
                ops[4] = lambda u: (
                    np.roll(u, -2) - 4 * np.roll(u, -1) + 6 * u
                    - 4 * np.roll(u, 1) + np.roll(u, 2)
                ) / dx_val**4

            elif bc_type == 'dirichlet':
                lv, rv = bv

                def d1_dir(u):
                    d = np.empty_like(u)
                    if n_pts == 1:
                        d[0] = (rv - lv) / (2 * dx_val)
                    else:
                        d[0] = (u[1] - lv) / (2 * dx_val)
                        d[-1] = (rv - u[-2]) / (2 * dx_val)
                        if n_pts > 2:
                            d[1:-1] = (u[2:] - u[:-2]) / (2 * dx_val)
                    return d

                def d2_dir(u):
                    d = np.empty_like(u)
                    if n_pts == 1:
                        d[0] = (rv - 2 * u[0] + lv) / dx_val**2
                    else:
                        d[0] = (u[1] - 2 * u[0] + lv) / dx_val**2
                        d[-1] = (rv - 2 * u[-1] + u[-2]) / dx_val**2
                        if n_pts > 2:
                            d[1:-1] = (
                                u[2:] - 2 * u[1:-1] + u[:-2]
                            ) / dx_val**2
                    return d

                ops[1] = d1_dir
                ops[2] = d2_dir

            elif bc_type == 'neumann':
                ld, rd = bv

                def d1_neu(u):
                    d = np.empty_like(u)
                    d[0] = ld
                    d[-1] = rd
                    if n_pts > 2:
                        d[1:-1] = (u[2:] - u[:-2]) / (2 * dx_val)
                    return d

                def d2_neu(u):
                    ghost_l = u[1] - 2 * dx_val * ld
                    ghost_r = u[-2] + 2 * dx_val * rd
                    d = np.empty_like(u)
                    d[0] = (u[1] - 2 * u[0] + ghost_l) / dx_val**2
                    d[-1] = (ghost_r - 2 * u[-1] + u[-2]) / dx_val**2
                    if n_pts > 2:
                        d[1:-1] = (
                            u[2:] - 2 * u[1:-1] + u[:-2]
                        ) / dx_val**2
                    return d

                ops[1] = d1_neu
                ops[2] = d2_neu

            return ops

        fd_ops = {}
        for name in field_vars:
            bv = bc_values.get(name, (0.0, 0.0))
            fd_ops[name] = _build_fd_ops(bc, dx, N, bv)

        # =================================================================
        # STEP 8: Lambdify RHS and build the system function
        # =================================================================
        # Argument order: [x, t, S0_u1, S1_u1, ..., S0_u2, ..., PHFN0, ...]
        all_syms = [x_sym, t_sym]
        for name in field_vars:
            max_o = max_spatial_order.get(name, 0)
            for k in range(max_o + 1):
                all_syms.append(var_symbols[name][k])
        ph_list = list(func_placeholders.keys())
        all_syms.extend(ph_list)

        lambdified_rhs = []
        for target_var, rhs_expr in parsed_equations:
            fn = sp.lambdify(all_syms, rhs_expr, modules='numpy')
            lambdified_rhs.append((target_var, fn))

        field_idx = {name: i for i, name in enumerate(field_vars)}

        def system(t_val, y_flat):
            """RHS f(t, y) for the semi-discrete PDE system."""
            dydt = np.zeros_like(y_flat)

            # Extract field arrays from flat state vector
            fields = {}
            for name in field_vars:
                fi = field_idx[name]
                fields[name] = y_flat[fi * N:(fi + 1) * N]

            # Compute spatial derivatives via finite differences
            spatial = {}
            for name in field_vars:
                spatial[name] = {0: fields[name]}
                for k in range(1, max_spatial_order.get(name, 0) + 1):
                    if k not in fd_ops[name]:
                        raise ValueError(
                            f"Spatial derivative order {k} not supported "
                            f"for bc='{bc}'"
                        )
                    spatial[name][k] = fd_ops[name][k](fields[name])

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
                    elif a in fields:
                        fargs.append(fields[a])
                    else:
                        raise ValueError(
                            f"Unknown argument '{a}' in {fname}(...)"
                        )
                ph_vals.append(function_dict[fname](*fargs))

            # Evaluate each RHS equation
            for target_var, fn in lambdified_rhs:
                args = [x_grid, t_val]
                for name in field_vars:
                    for k in range(max_spatial_order.get(name, 0) + 1):
                        args.append(spatial[name][k])
                args.extend(ph_vals)

                fi = field_idx[target_var]
                dydt[fi * N:(fi + 1) * N] = fn(*args)

            return dydt

        self.ODE = system

    def Solve(self, initial_conditions, t_span, t_eval, method='RK45', **kwargs):
        """
        Numerically integrate the parsed PDE system.

        Parameters
        ----------
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
        if self.ODE is None:
            raise RuntimeError("Call Parse first.")

        ic = np.asarray(initial_conditions, dtype=float)
        if ic.ndim == 2:
            ic = ic.ravel()

        if len(ic) != self.order:
            raise ValueError(
                f"Expected {self.order} initial values "
                f"({len(self.field_vars)} fields x {self.N} points), "
                f"got {len(ic)}"
            )

        return solve_ivp(
            self.ODE, t_span, ic,
            t_eval=t_eval, method=method, dense_output=True, **kwargs
        )

    def get_field(self, sol, var_name=None, var_idx=0):
        """
        Extract a single field from the solution.

        Parameters
        ----------
        sol : OdeSolution
            Result from Solve.
        var_name : str, optional
            Field name, e.g. 'u1'. Overrides var_idx if given.
        var_idx : int, optional
            Index into self.field_vars. Default 0.

        Returns
        -------
        ndarray of shape (N, n_times)
        """
        if var_name is not None:
            var_idx = self.field_vars.index(var_name)
        return sol.y[var_idx * self.N:(var_idx + 1) * self.N, :]

    @staticmethod
    def _parse_single(pde_strings, function_dict, x_span, N, bc, bc_values):
        """Worker for parallel parsing."""
        pde = AutoPDE()
        pde.Parse(pde_strings, function_dict, x_span, N, bc, bc_values)
        return pde

    @staticmethod
    def Parallel_Parse(inputs, n_jobs=-1, backend="loky"):
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
        list of AutoPDE
        """
        def _unpack(args):
            pde_s, fd, xs, n = args[0], args[1], args[2], args[3]
            bc_arg = args[4] if len(args) > 4 else 'periodic'
            bv_arg = args[5] if len(args) > 5 else None
            return pde_s, fd, xs, n, bc_arg, bv_arg

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(AutoPDE._parse_single)(*_unpack(inp))
            for inp in tqdm(inputs, desc="Parsing PDEs")
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
    t0 = time.perf_counter()
    odes = AutoODE.Parallel_Parse(inputs, n_jobs=-1, backend="threading")
    t_parse = time.perf_counter() - t0
    print(f"  Parallel parse: {t_parse:.3f} s")

    # --- Sequential parse (for comparison) ---
    print(f"Sequential-parsing {n_instances} Lorenz 63 instances ...")
    t0 = time.perf_counter()
    odes_seq = []
    for ode_strs, fdict in tqdm(inputs, desc="Sequential parse"):
        o = AutoODE()
        o.Parse(ode_strs, fdict)
        odes_seq.append(o)
    t_seq = time.perf_counter() - t0
    print(f"  Sequential parse: {t_seq:.3f} s")
    print(f"  Speedup: {t_seq / t_parse:.2f}x")

    # --- Solve all instances ---
    t_span = (0, 25)
    t_eval = np.linspace(0, 25, 2000)
    ic = [1.0, 1.0, 1.0]  # same IC for all

    print(f"\nSolving {n_instances} systems ...")
    solutions = []
    for ode in tqdm(odes, desc="Solving"):
        solutions.append(ode.Solve(ic, t_span, t_eval))

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
    pde.Parse(
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
    t0 = time.perf_counter()
    sol_pde = pde.Solve(ic_pde, (0, z_max), z_eval,
                        method='DOP853', rtol=1e-8, atol=1e-10)
    t_solve = time.perf_counter() - t0
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
                 fontsize=13, y=0.98)  # <-- Increased y to avoid cutoff
    plt.tight_layout()
    plt.show()
