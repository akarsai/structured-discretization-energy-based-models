#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements the Cahn--Hilliard equation example
# using piecewise linear finite elements in space
#
#

import jax
import jax.numpy as jnp

from helpers.energy_based_model import EnergyBasedModel_LinearJR
from main.space_discretization import AnsatzSpace

# sparse computations
from jax.scipy.sparse.linalg import bicgstab
from helpers.other import sparse_blockmatrix

# for noise initial condition
from helpers.noise import fractal_noise_on_points

# to compute an equilibrium for the manufactured solution
from main.time_discretization import projection_method

# other
from helpers.other import dprint
from timeit import default_timer as timer

class CahnHilliard(EnergyBasedModel_LinearJR):
    
    def __init__(
            self,
            mesh_settings: dict = None,
            sigma: float = 1.0,
            eps: float = 0.15,
            W: callable = None,
            visualize_initial_condition: bool = False,
            ):
        
        # parameters
        self.sigma = sigma
        self.eps = eps
        
        # chemical potential
        if W is None:
            W = lambda x: 1/4 * (x**2 - 1)**2
            # theta = 0.3
            # theta_c = 0.8
            # W = lambda x: theta/2 * ( (1+x) * jnp.log((1+x)/2) + (1-x) * jnp.log((1-x)/2) ) + theta_c/2 * (1-x**2)
        self.W = W # assumes W is already vmapped
        self.Wprime = jax.vmap(jax.grad(lambda x: W(x[0])), in_axes=0,) # W(x[0]) forces W to be interpreted as a scalar function
        
        # define ansatz space for space discretization
        if mesh_settings is None:
            mesh_settings = {
                'Lx': 1.0,
                'Ly': 1.0,
                'nx': 10, # todo: increase for publication
                'ny': 10, # todo: increase for publication
                }
            
        self.space = AnsatzSpace(
            mesh_settings=mesh_settings,
            inner_product='H1',
            )
        
        # set up J
        M = self.space.l2_mass_matrix
        J_matrix = sparse_blockmatrix(0*M, M, -1*M, 0*M)
        
        # set up R
        K = self.space.l2_stiffness_matrix
        R_matrix = sparse_blockmatrix(0*K, 0*K, 0*K, self.sigma * K)
        
        super().__init__(J_matrix=J_matrix, R_matrix=R_matrix)
        
        # set pde flag
        self.was_pde = True
        
        # set dimensions
        self.dims = (self.space.dim, 0, self.space.dim) # 0 for z2 dim
        
        ### initial condition
        # set initial condition for v
        self.v_init = lambda xy: fractal_noise_on_points(xy, x_range=(0.0,self.space.mesh.Lx), y_range=(0.0,self.space.mesh.Ly), seed=0,)
        # evaluate v_init on mapped triangle quadrature nodes, compute projection to get coeffs of v_init
        self.v_init_quad = self.v_init(self.space.mapped_quad_nodes) # shape (self.space.mesh.num_triangles, self.space.num_quad_nodes)
        v0 = self.space.get_projection_coeffs(self.v_init_quad, inner_product='L2')
        if visualize_initial_condition: self.space.visualize_coefficient_vector(v0, title='Cahn--Hilliard model: initial condition for $v$')
        # compute consistent initial condition - implicitly assumes u2(t_0) = nabla v(t_0) \dot n = 0
        # the initial condition w0 is determined via
        # eps * stiffness_matrix @ v0 + 1/eps * [int_Omega W'(v0) phi_i dx]_i=1,...,dim = mass_matrix @ w0
        Wprime_v0 = self.Wprime(self.v_init_quad) # shape (self.space.mesh.num_triangles, self.space.num_quad_nodes)
        Wprime_v0_gbf = jnp.einsum('tq,btq->btq', Wprime_v0, self.space.gbf_quad)
        int_Wprime_v0_gbf = jnp.sum(self.space.quadrature_with_values_mapped(Wprime_v0_gbf), axis=1) # shape (b,)
        rhs_linear_system = self.eps * self.space.l2_stiffness_matrix @ v0 + 1/self.eps * int_Wprime_v0_gbf
        # solve the linear system
        w0 = bicgstab(self.space.l2_mass_matrix, rhs_linear_system)[0]
        self.initial_condition = jnp.hstack((v0, w0))
    
    def B(self, u):
        u1 = u[...,self.space.mesh.num_boundary_edges:] # shape (..., num_boundary_edges), contains nabla w dot normal on boundary edges
        u2 = u[...,:self.space.mesh.num_boundary_edges] # shape (..., num_boundary_edges), contains nabla v dot normal on boundary edges
        B_mat = self.space.boundary_mass_matrix @ self.space.boundary_extension_matrix
        return jnp.hstack((self.eps * B_mat @ u2, self.sigma * B_mat @ u1))
        
    def hamiltonian(self, z1, z2):
        # z1 = coeffs to v
        # z2 is not used
        
        # the part eps/2 * \nabla v.T @ \nabla v
        stiffness = self.eps / 2 * z1.T @ self.space.l2_stiffness_matrix @ z1 # not vectorized yet
        
        # evaluate v at quad nodes
        v_quad, _ = self.space.eval_coeffs_quad(z1) # v_quad.shape = (self.space.mesh.num_triangles, self.space.num_quad_nodes)
        W_quad = self.W(v_quad) # W_quad.shape = (self.space.mesh.num_triangles, self.space.num_quad_nodes)
        potential = 1 / self.eps * jnp.sum(self.space.quadrature_with_values_mapped(W_quad)) # integral over W(v)
        
        return stiffness + potential
    
    def default_control(self, t):
        return jnp.zeros((t.shape[0], 2*self.space.mesh.num_boundary_edges))
    
    def get_manufactured_solution(self):
        r"""
        def v_spatial_manufactured(xy):
            x, y = xy[...,0], xy[...,1]
            # return jnp.sin(4*jnp.pi*x/self.space.mesh.Lx) * jnp.cos(4*jnp.pi*y/self.space.mesh.Ly) # 2 periods in both dimensions
            return 2*jnp.exp(-25*((x-self.space.mesh.Lx/2)**2 + (y-self.space.mesh.Ly/2)**2))-1 # gauss bubble
            
        def v_manufactured(xy, t):
            return jnp.sin(t) * v_spatial_manufactured(xy)
        
        def dt_v_manufactured(xy, t):
            return jnp.cos(t) * v_spatial_manufactured(xy)
        
        # compute consistent w - implicitly assumes u1(t_0) = nabla v(t_0) \dot n = 0
        # w is determined via
        # eps * stiffness_matrix @ v + 1/eps * [int_Omega W'(v) phi_i dx]_i=1,...,dim = mass_matrix @ w
        v_manufactured_quad = jax.vmap(lambda t: v_manufactured(self.space.mapped_quad_nodes, t), in_axes=0)
        dt_v_manufactured_quad = jax.vmap(lambda t: dt_v_manufactured(self.space.mapped_quad_nodes, t), in_axes=0)
        """
        
        def v_spatial(xy):
            x, y = xy[...,0], xy[...,1]
            # return jnp.sin(4*jnp.pi*x/self.space.mesh.Lx) * jnp.cos(4*jnp.pi*y/self.space.mesh.Ly) # 2 periods in both dimensions
            return 2*jnp.exp(-25*((x-self.space.mesh.Lx/2)**2 + (y-self.space.mesh.Ly/2)**2)) - 1 # gauss bubble
    
        def v_manufactured(xy, t):
            return (0.1*jnp.sin(t) + 1) * v_spatial(xy)
        
        def dt_v_manufactured(xy, t):
            return 0.1*jnp.cos(t) * v_spatial(xy)
        
        v_manufactured_quad = jax.vmap(lambda t: v_manufactured(self.space.mapped_quad_nodes, t), in_axes=0)
        dt_v_manufactured_quad = jax.vmap(lambda t: dt_v_manufactured(self.space.mapped_quad_nodes, t), in_axes=0)
        projection_vmap = jax.vmap(lambda x: self.space.get_projection_coeffs(x, inner_product='L2'), in_axes=0)
        z1 = lambda t: projection_vmap(v_manufactured_quad(t))
        dt_z1 = lambda t: projection_vmap(dt_v_manufactured_quad(t))
        Wprime_vmap = jax.vmap(lambda x: self.Wprime(x), in_axes=0)
        Wprime_v = lambda t: Wprime_vmap(v_manufactured_quad(t)) # output has shape (..., self.space.mesh.num_triangles, self.space.num_quad_nodes)
        Wprime_v_gbf = lambda t: jnp.einsum('...tq,btq->...btq', Wprime_v(t), self.space.gbf_quad) # output has shape (..., self.dim, self.space.mesh.num_triangles, self.space.num_quad_nodes)
        int_Wprime_v_gbf = lambda t: jnp.sum(self.space.quadrature_with_values_mapped(Wprime_v_gbf(t)), axis=2) # output has shape (..., self.dim, self.space.mesh.num_triangles). axis=2 since time is at axis=0
        # compute consistent w via
        # eps * stiffness_matrix @ v + 1/eps * [int_Omega W'(v) phi_i dx]_i=1,...,dim = mass_matrix @ w # implicitly assumes zero flux of w, but this is irrelevant for manufactured solution
        # compute associated boundary control values
        stiffness_vmap = jax.vmap(lambda x: self.space.l2_stiffness_matrix @ x, in_axes=0)
        rhs_linear_system = lambda t: self.eps * stiffness_vmap(z1(t)) + 1/self.eps * int_Wprime_v_gbf(t)
        # solve the linear system
        solver = jax.vmap(lambda x: bicgstab(self.space.l2_mass_matrix, x)[0], in_axes=0)
        z3 = lambda t: solver(rhs_linear_system(t))
        
        # empty array callable
        _ = lambda t: jnp.zeros((t.shape[0], 0)) # empty array
        
        # z2 is empty array
        z2 = _
        dt_z2 = _
        
        # build initial condition
        t0 = jnp.zeros((1,))
        z1_0 = z1(t0)[0,:]
        z2_0 = z2(t0)[0,:]
        z3_0 = z3(t0)[0,:]
        z0 = jnp.hstack((z1_0, z2_0, z3_0))
        
        # build missing control input
        u = jnp.zeros((2*self.space.mesh.num_boundary_edges,))
        
        # # visualize
        # self.space.visualize_coefficient_vector(z1, title='Cahn--Hilliard model: manufactured solution for $v$')
        # self.space.visualize_coefficient_vector(z3, title='Cahn--Hilliard model: manufactured solution for $w$')
        
        # compute g for g_manufactured solution - the latter is a constant map since v_manufactured is constant
        h1 = lambda t: self.nabla_1_ham_vmap(z1(t), z2(t))
        h2 = lambda t: self.nabla_2_ham_vmap(z1(t), z2(t))
        rhs = lambda t: \
            self.J_vmap(dt_z1(t), h2(t), z3(t)) \
            - self.R_vmap(dt_z1(t), h2(t), z3(t)) \
            + self.B(u) # u is constant, no B_vmap needed. otherwise: self.B_vmap(u(t))
        lhs = lambda t: jnp.hstack((h1(t), dt_z2(t), jnp.zeros_like(z3(t))))
        
        manufactured_solution = lambda t: jnp.hstack((z1(t), z2(t), z3(t)))
        control_manufactured_solution = lambda t: jnp.tile(u, (t.shape[0], 1)) # repeats boundary input entries as often as needed, shape (t.shape[0], 2*self.space.mesh.num_boundary_edges)
        g_manufactured_solution = lambda t: lhs(t) - rhs(t)
        
        return z0, manufactured_solution, control_manufactured_solution, g_manufactured_solution
    
class CahnHilliardReducedOrder(CahnHilliard):
    
    def __init__(self, r1: int = 5, r3: int = 5, **kwargs):
        
        super().__init__(**kwargs)
        
        # store reduced order dimensions
        self.r1 = r1
        self.r3 = r3
        
        # simulate the system to obtain snapshot matrix
        T = 1.5
        nt = int(T*100)+1
        tt = jnp.linspace(0, T, nt)
        degree = 3
        s = timer()
        _, zz, dt_zz = projection_method(
            CahnHilliard(**kwargs), # fom
            tt,
            z0=self.initial_condition,
            control=self.default_control,
            degree=degree,
            num_quad_nodes=2*degree,
            num_proj_nodes=2*degree,
            # debug=True,
            )['boundaries']
        e = timer()
        print(f'[cahn hilliard rom] full order simulation took {e-s:.2f} seconds')
        Q1 = zz[:,:self.dims[0]].T # snapshot matrix for z1 states - time index in second position
        Q3 = zz[:,self.dims[0]:].T # snapshot matrix for z3 states - time index in second position
        U1, _, _ = jnp.linalg.svd(Q1, full_matrices=False)
        U3, _, _ = jnp.linalg.svd(Q3, full_matrices=False)
        self.V1 = U1[:,:self.r1]
        self.V3 = U3[:,:self.r3]
        
        # build reduced order matrix
        self.V2 = jnp.eye(0) # self.dims[1] == 0 for cahn hilliard model
        self.V = jax.scipy.linalg.block_diag(self.V1, self.V2, self.V3)
        
        # update state dimension
        self.dims = (self.V1.shape[1], self.V2.shape[1], self.V3.shape[1])
        
        # update initial condition
        self.initial_condition = self.V.T @ self.initial_condition
        
        # set rom flag
        self.is_rom = True
        
    def hamiltonian(self, z1, z2):
        return super().hamiltonian(self.V1 @ z1, self.V2 @ z2)
    
    def J(self, dt_z1, h2, z3):
        return self.V.T @ super().J(self.V1 @ dt_z1, self.V2 @ h2, self.V3 @ z3)
    
    def R(self, dt_z1, h2, z3):
        return self.V.T @ super().R(self.V1 @ dt_z1, self.V2 @ h2, self.V3 @ z3)
        
    def B(self, u):
        return self.V.T @ super().B(u)

    

if __name__ == '__main__':
    
    from helpers.other import mpl_settings
    # mpl_settings()
    
    import matplotlib
    matplotlib.rc('axes.formatter', useoffset=False)
    
    jax.config.update("jax_enable_x64", True)
    
    
    # ch = CahnHilliard()
    # # # test J and R
    # # J = ch.J_matrix.todense()
    # # R = ch.R_matrix.todense()
    # # plot_matrix(J, 'J')
    # # plot_matrix(R, 'R')
    # # dprint(jnp.linalg.norm(J+J.T)) # should be zero
    # # dprint(jnp.linalg.norm(R-R.T)) # should be zero
    #
    # # # test hamiltonian
    # # z1 = jnp.zeros((ch.space.dim,)).at[ch.space.dim//2+1].set(1.0)
    # # # ch.space.visualize_coefficient_vector(z1)
    # # h = ch.hamiltonian(z1,None)
    # # dprint(h.shape)
    # # dprint(h)
    #
    # # # test gradient of hamiltonian
    # nabla_h = ch.nabla_1_ham
    # z1 = jnp.zeros((ch.space.dim,)).at[ch.space.dim//2+1].set(1.0)
    # n = nabla_h(z1,None)
    # dprint(n.shape)
    # dprint(n)
    # # compare with naive implementation
    # def naive_nabla_h(z1, z2):
    #     stiffness = ch.eps * ch.space.l2_stiffness_matrix @ z1
    #     v_quad, _ = ch.space.eval_coeffs_quad(z1)
    #     Wprime_quad = ch.Wprime(v_quad)
    #     Wprime_gbf = jnp.einsum('tq,btq->btq', Wprime_quad, ch.space.gbf_quad)
    #     potential = 1 / ch.eps * jnp.sum(ch.space.quadrature_with_values_mapped(Wprime_gbf), axis=1)
    #     return stiffness + potential
    # naive_n = naive_nabla_h(z1,None)
    # dprint(naive_n.shape)
    # dprint(naive_n)
    # dprint(jnp.linalg.norm(n-naive_n))
    # # compare with projected implementation
    # def non_discrete_nabla_ham(v_at_quad_nodes, nabla_v_at_quad_nodes):
    #     """
    #     computes the vector
    #
    #     [ int_Omega eps * nabla v.T * nabla gbf_i + eps^-1 * gbf_i * Wprime(v) dx ]_{i=1,...,dim}
    #     =
    #     [ < nabla ham(v), phi_i > ]_{i=1,...,dim}
    #     =
    #     [ < proj nabla ham(v), phi_i > ]_{i=1,...,dim}
    #     =
    #     mass_matrix @ coeffs(proj nabla ham(v))
    #
    #     ---> the projection is happening implicitely and does not need to be included
    #     """
    #     Wprime_quad = ch.Wprime(v_at_quad_nodes) # shape (t, q)
    #     Wprime_gbf_quad = jnp.einsum('tq,btq->btq', Wprime_quad, ch.space.gbf_quad) # shape (b, t, q)
    #     potential = 1 / ch.eps * jnp.sum(ch.space.quadrature_with_values_mapped(Wprime_gbf_quad), axis=1) # shape (b,)
    #     nabla_v_grad_gbf_quad = jnp.einsum('tqn,btqn->btq', nabla_v_at_quad_nodes, ch.space.grad_gbf_quad) # shape (b, t, q)
    #     stiffness = ch.eps * jnp.sum(ch.space.quadrature_with_values_mapped(nabla_v_grad_gbf_quad), axis=1) # shape (b,)
    #     return potential + stiffness
    # def projected_nabla_h(z1, z2):
    #     """
    #     z1 = coeffs to v
    #     z2 = unused
    #     """
    #     # compute v at quad nodes and grad v at quad nodes
    #     v_quad, nabla_v_quad = ch.space.eval_coeffs_quad(z1)
    #     # compute nabla_1 ham coeffs
    #     n = non_discrete_nabla_ham(v_quad, nabla_v_quad)
    #     return n
    # proj_n = projected_nabla_h(z1,None)
    # dprint(proj_n.shape)
    # dprint(proj_n)
    # dprint(jnp.linalg.norm(n-proj_n))
    
    # simulate cahn hilliard system
    ch = CahnHilliard()
    
    z0_manufactured_solution, manufactured_solution, control_manufactured_solution, g_manufactured_solution = ch.get_manufactured_solution() # todo: remove
    
    # simulation settings
    T = 1.5
    nt = int(T*100)+1
    tt = jnp.linspace(0, T, nt+1)
    # control = ch.default_control
    # z0 = ch.initial_condition
    degree = 2
    
    # run simulation with projection method
    s = timer()
    proj_solution = projection_method(
        ebm=ch,
        tt=tt,
        z0=z0_manufactured_solution,
        control=control_manufactured_solution,
        degree=degree,
        num_quad_nodes=degree+1,
        num_proj_nodes=2*degree,
        g_manufactured_solution=g_manufactured_solution,
        debug=True,
        )
    e = timer()
    print(f'\nprojection method took {e-s:.2f} seconds')
    zz_proj = proj_solution['boundaries'][1]
    
    # # visualize energy balance error
    # from helpers.errors import energy_balance_error
    # eb_error = energy_balance_error(
    #     proj_solution,
    #     ch,
    #     control,
    #     )
    # print(f'maximum energy balance error = {jnp.max(eb_error):.2e}')
    # import matplotlib.pyplot as plt
    # from helpers.other import mpl_settings
    # mpl_settings()
    # plt.semilogy(tt[1:], eb_error, label='cahn hilliard')
    # plt.legend()
    # plt.ylim(1.5e-18, 1.5e-3)
    # plt.title('energy balance error')
    # plt.xlabel('$t$')
    # plt.ylabel('error')
    # plt.tight_layout()
    # plt.show()
    
    
    
    
    
    #
    # zz_proj_1 = zz_proj[:, :ch.dims[0]]
    # zz_proj_2 = zz_proj[:, ch.dims[0]:ch.dims[0]+ch.dims[1]]
    # plt.semilogy(tt, ch.hamiltonian_vmap(zz_proj_1, zz_proj_2), label='cahn hilliard hamiltonian')
    # plt.legend()
    # plt.title('cahn hilliard hamiltonian')
    # plt.xlabel('$t$')
    # plt.ylabel('$\mathcal{H}(t)$')
    # plt.tight_layout()
    # plt.show()
    #
    # ch.space.visualize_coefficient_vector(
    #     zz_proj[0, :ch.dims[0]],
    #     title=f'$v({0*T:.1f})$, projection method, $\\varepsilon={ch.eps:.2f}, \\sigma={ch.sigma:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     zz_proj[1*nt//4, :ch.dims[0]],
    #     title=f'$v({1*T/4:.1f})$, projection method, $\\varepsilon={ch.eps:.2f}, \\sigma={ch.sigma:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     zz_proj[2*nt//4, :ch.dims[0]],
    #     title=f'$v({2*T/4:.1f})$, projection method, $\\varepsilon={ch.eps:.2f}, \\sigma={ch.sigma:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     zz_proj[3*nt//4, :ch.dims[0]],
    #     title=f'$v({3*T/4:.1f})$, projection method, $\\varepsilon={ch.eps:.2f}, \\sigma={ch.sigma:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     zz_proj[-1, :ch.dims[0]],
    #     title=f'$v({T:.1f})$, projection method, $\\varepsilon={ch.eps:.2f}, \\sigma={ch.sigma:.2f}$',
    #     )
    
    d1, d2, d3 = ch.dims
    
    zz_man = manufactured_solution(tt)
    
    error_1 = zz_proj[:, :d1] - zz_man[:, :d1]
    error_2 = zz_proj[:, d1:] - zz_man[:, d1:]
    
    # ch.space.visualize_coefficient_vector(
    #     error_1[0,:],
    #     title=f'error in $v$ at time ${0:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     error_1[1*nt//4, :],
    #     title=f'error in $v$ at time ${1*T/4:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     error_1[2*nt//4, :],
    #     title=f'error in $v$ at time ${2*T/4:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     error_1[3*nt//4, :],
    #     title=f'error in $v$ at time ${3*T/4:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     error_1[-1, :],
    #     title=f'error in $v$ at time ${T:.2f}$',
    #     )
    #
    # ch.space.visualize_coefficient_vector(
    #     error_2[0,:],
    #     title=f'error in $w$ at time ${0:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     error_2[1*nt//4, :],
    #     title=f'error in $w$ at time ${1*T/4:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     error_2[2*nt//4, :],
    #     title=f'error in $w$ at time ${2*T/4:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     error_2[3*nt//4, :],
    #     title=f'error in $w$ at time ${3*T/4:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     error_2[-1, :],
    #     title=f'error in $w$ at time ${T:.2f}$',
    #     )
    
    z1 = zz_proj[:, :d1]
    z3 = zz_proj[:, d1:]
    #
    # ch.space.visualize_coefficient_vector(
    #     z1[0,:],
    #     title=f'computed $v$ at time ${0:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     z1[1*nt//4, :],
    #     title=f'computed $v$ at time ${1*T/4:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     z1[2*nt//4, :],
    #     title=f'computed $v$ at time ${2*T/4:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     z1[3*nt//4, :],
    #     title=f'computed $v$ at time ${3*T/4:.2f}$',
    #     )
    # ch.space.visualize_coefficient_vector(
    #     z1[-1, :],
    #     title=f'computed $v$ at time ${T:.2f}$',
    #     )
    #
    ch.space.visualize_coefficient_vector(
        z3[0,:],
        title=f'computed $w$ at time ${0:.2f}$',
        )
    ch.space.visualize_coefficient_vector(
        z3[1*nt//4, :],
        title=f'computed $w$ at time ${1*T/4:.2f}$',
        )
    ch.space.visualize_coefficient_vector(
        z3[2*nt//4, :],
        title=f'computed $w$ at time ${2*T/4:.2f}$',
        )
    ch.space.visualize_coefficient_vector(
        z3[3*nt//4, :],
        title=f'computed $w$ at time ${3*T/4:.2f}$',
        )
    ch.space.visualize_coefficient_vector(
        z3[-1, :],
        title=f'computed $w$ at time ${T:.2f}$',
        )
    
    import matplotlib.pyplot as plt
    plt.plot(tt, jnp.linalg.norm(z3, axis=1), label=r'norm $w(t)$')
    plt.xlabel(r'time $t$')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    import matplotlib.pyplot as plt
    plt.plot(tt, jnp.linalg.norm(jnp.einsum('nm,...m->...n', ch.space.l2_stiffness_matrix.todense(), z3), axis=1), label=r'norm $K w(t)$')
    plt.xlabel(r'time $t$')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
