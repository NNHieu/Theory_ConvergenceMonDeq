from audioop import bias
from functools import partial
import jax
from jax import numpy as jnp, random as jrandom
from jax import custom_vjp, vjp
import equinox as eqx

from mondeq.splittings import PeacemanRachfordState, solve

class MonLinear(eqx.Module):
    p_A: jnp.array
    p_B: jnp.array
    p_U: jnp.array
    m: eqx.static_field
    out_size: eqx.static_field

    def __init__(self, in_size, out_size, m=0.1, *, key) -> None:
        key_A, key_B, key_U = jax.random.split(key, 3)
        self.p_A = jrandom.normal(key_A, (out_size, out_size)) / jnp.sqrt(out_size)
        self.p_B = jrandom.normal(key_A, (out_size, out_size)) / jnp.sqrt(out_size)
        self.p_U = jrandom.normal(key_A, (in_size, out_size)) / jnp.sqrt(out_size)
        self.m = m
        self.out_size = out_size

    def W(self, z):
        return (1 - self.m) * z - z@self.p_A@self.p_A.T + z@self.p_B - z@self.p_B.T
    
    def W_trans(self, z):
        return (1 - self.m) * z - z@self.p_A@self.p_A.T - z@self.p_B + z@self.p_B.T
    
    def bias(self, z):
        return z @ self.p_U
    
    def calW_trans(self):
        return (1 - self.m)*jnp.eye(*self.p_A.shape) - self.p_A@self.p_A.T + self.p_B - self.p_B.T

    def calW_inv(self, alpha, beta):
        W_trans = (1 - self.m)*jnp.eye(*self.p_A.shape) - self.p_A@self.p_A.T + self.p_B - self.p_B.T
        Winv_trans = jnp.linalg.inv(alpha*jnp.eye(*W_trans.shape) + beta*W_trans)
        return Winv_trans
    
    def inverse(self, Winv_trans, z):
        return z @ Winv_trans
    
    def __call__(self, z, x):
        return self.W(z) + x @ self.p_U

class Relu(eqx.Module):

    def __call__(self, x):
        return jax.nn.relu(x)
    
    def derivative(self, x):
        return jax.lax.select(x > 0, jax.lax.full_like(x, 1.), jax.lax.full_like(x, 0.))


def _deq(dyn, max_iter, tol, nonlin_mdl, lin_mdl, Z0, X):
    bs = X.shape[0]
    U0 = jnp.zeros_like(Z0)

    dyn_init, dyn_update = dyn(lin_mdl, nonlin_mdl)
    dyn_state: PeacemanRachfordState = dyn_init(Z0, U0, X)
    solver_state = jax.lax.stop_gradient(solve(dyn_state, dyn_update, max_iter, tol))
    return solver_state

@partial(custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def deq(dyn, max_iter, tol, nonlin_mdl, lin_mdl, Z0, X):
    solver_state = _deq(dyn, max_iter, tol, nonlin_mdl, lin_mdl, Z0, X)
    return solver_state.min_step.z

def deq_forward(dyn, max_iter, tol, nonlin_mdl, lin_mdl, Z0, X):
    solver_state = _deq(dyn, max_iter, tol, nonlin_mdl, lin_mdl, Z0, X)
    return solver_state.min_step.z, (solver_state.min_step, lin_mdl, X)

def _fp_bwd(dyn, max_iter, tol, nonlin_mdl, res, g):
    dyn_state: PeacemanRachfordState = res[0]
    lin_mdl = res[1]
    x = res[2]

    z_star = dyn_state.z
    j = nonlin_mdl.derivative(z_star)
    I = j == 0
    d = (1 - j) / j
    v = j * g

    z0 = jnp.zeros_like(z_star)
    u0 = jnp.zeros_like(z_star)

    def alter_nonlin_mdl_bwd(u):
        zn = (u + dyn_state.alpha*(1 + d)*v) / (1 + dyn_state.alpha*d)
        zn = jax.lax.select(I, v, zn)
        return zn

    dyn_init, dyn_update = dyn(lin_mdl, alter_nonlin_mdl_bwd)
    bwd_dyn_state: PeacemanRachfordState = dyn_init(z0, u0, x)
    bwd_dyn_state = bwd_dyn_state._replace(
        Winv=bwd_dyn_state.Winv.T,
        bias=jnp.zeros_like(bwd_dyn_state.bias)
    )
    solver_state = solve(bwd_dyn_state, dyn_update, max_iter, tol)
    
    dg = lin_mdl.W_trans(solver_state.min_step.z)
    dg = g + dg
    return dg

def deq_backward(dyn, max_iter, tol, nonlin_mdl, res, g):
    dyn_state: PeacemanRachfordState = res[0]
    lin_mdl = res[1]
    x = res[2]
    z_star = dyn_state.z
    dg = _fp_bwd(dyn, max_iter, tol, nonlin_mdl, res, g)

    # Problem: nonlin_mdl
    _, vjp_lin_fn = vjp(lambda lin_mdl, x: nonlin_mdl(lin_mdl(z_star, x)), lin_mdl, x)

    dlin, dx = vjp_lin_fn(dg)

    return (dlin, None ,dx)

deq.defvjp(deq_forward, deq_backward)

def fcmon(dyn, max_iter, tol, u, nonlin_module, lin_module, Z0, X):
    Z_star = deq(dyn, max_iter, tol, nonlin_module, lin_module, Z0, X)
    return Z_star@u