from typing import NamedTuple, Union
import jax
from jax import numpy as jnp

class PeacemanRachfordState(NamedTuple):
    """Results from PeacemanRachford splitting.
    Parameters:
        n_steps: integer the number of iterations of the update.
        z: array containing the prev argument value found during the search.
        objective: array containing prev lowest 2 norm of the objective function
    """
    n_step: Union[int, jnp.ndarray]
    z: jnp.ndarray
    u: jnp.ndarray
    objective: jnp.ndarray

    alpha: Union[float, jnp.ndarray]
    Winv: jnp.ndarray
    bias: jnp.ndarray

def build_peacemanrachford_update(alpha, linear_module, nonlin_module):

    def init_fn(z0, u0, x):
        init_objective = 1.
        W_inv = linear_module.calW_inv(1 + alpha, -alpha)

        return PeacemanRachfordState(
            n_step = 0,
            z = z0,
            u = u0,
            objective=init_objective,
            alpha=alpha,
            Winv=W_inv,
            bias = linear_module.bias(x),
        )

    def update_fn(state: PeacemanRachfordState):
        u_12 = 2*state.z - state.u
        z_12 = linear_module.inverse(state.Winv, u_12 + alpha*state.bias)
        u = 2 * z_12 - u_12
        zn = nonlin_module(u)

        # fn = nonlin_module(linear_module.W(zn) + state.bias)
        new_objective = jnp.linalg.norm(zn - state.z)/(jnp.linalg.norm(zn) + 1e-6)
        
        state = state._replace(
            n_step=state.n_step + 1,
            z = zn,
            u = u,
            objective=new_objective,
            
            # trace=trace,
        )

        return state

    return init_fn, update_fn

class SolverState(NamedTuple):
    solve_step: Union[int, jnp.ndarray] 
    cur_step: PeacemanRachfordState
    min_step: PeacemanRachfordState
    trace: list

def solve(dyn_state: PeacemanRachfordState, update_fn, max_iter, tol):
    trace = jnp.zeros(max_iter)
    trace = trace.at[0].set(dyn_state.objective)
    state = SolverState(
        solve_step=0,
        cur_step=dyn_state,
        min_step=dyn_state,
        trace=trace
    )
    
    def cond_fun(state: SolverState):
        return (jnp.logical_not(state.min_step.objective < tol) &
                (state.solve_step < max_iter))
    
    def body_fn(state: SolverState):
        new_step: PeacemanRachfordState = update_fn(state.cur_step)
        min_found = new_step.objective < state.min_step.objective
        trace = state.trace.at[state.solve_step + 1].set(new_step.objective)
        state = state._replace(
            solve_step=state.solve_step + 1,
            cur_step=new_step,
            min_step=jax.lax.cond(min_found,lambda: new_step,lambda: state.min_step),
            trace=trace
        )
        return state
    
    state = jax.lax.while_loop(cond_fun, body_fn, state)
    return state

