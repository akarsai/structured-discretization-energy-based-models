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
import pickle

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
        # reconstruct discrete v_h at quad nodes from v0 coefficients to avoid disc
        v0_quad_discrete = jnp.einsum('b,btq->tq', v0, self.space.gbf_quad)
        Wprime_v0 = self.Wprime(v0_quad_discrete) # shape (self.space.mesh.num_triangles, self.space.num_quad_nodes)
        Wprime_v0_gbf = jnp.einsum('tq,btq->btq', Wprime_v0, self.space.gbf_quad)
        int_Wprime_v0_gbf = jnp.sum(self.space.quadrature_with_values_mapped(Wprime_v0_gbf), axis=1) # shape (b,)
        rhs_linear_system = self.eps * self.space.l2_stiffness_matrix @ v0 + 1/self.eps * int_Wprime_v0_gbf
        # solve the linear system
        w0 = bicgstab(self.space.l2_mass_matrix, rhs_linear_system, tol=1e-14)[0]
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
        vh_quad = lambda t: jnp.einsum('...b,btq->...tq', z1(t), self.space.gbf_quad) # recompute vh from z1 coefficients
        dt_z1 = lambda t: projection_vmap(dt_v_manufactured_quad(t))
        Wprime_vmap = jax.vmap(lambda x: self.Wprime(x), in_axes=0)
        Wprime_vh = lambda t: Wprime_vmap(vh_quad(t)) # output has shape (..., self.space.mesh.num_triangles, self.space.num_quad_nodes)
        Wprime_vh_gbf = lambda t: jnp.einsum('...tq,btq->...btq', Wprime_vh(t), self.space.gbf_quad) # output has shape (..., self.dim, self.space.mesh.num_triangles, self.space.num_quad_nodes)
        int_Wprime_vh_gbf = lambda t: jnp.sum(self.space.quadrature_with_values_mapped(Wprime_vh_gbf(t)), axis=2) # output has shape (..., self.dim, self.space.mesh.num_triangles). axis=2 since time is at axis=0
        # compute consistent w via
        # eps * stiffness_matrix @ v + 1/eps * [int_Omega W'(v) phi_i dx]_i=1,...,dim = mass_matrix @ w # implicitly assumes zero flux of w, but this is irrelevant for manufactured solution
        # compute associated boundary control values
        stiffness_vmap = jax.vmap(lambda x: self.space.l2_stiffness_matrix @ x, in_axes=0)
        rhs_linear_system = lambda t: self.eps * stiffness_vmap(z1(t)) + 1/self.eps * int_Wprime_vh_gbf(t)
        # solve the linear system
        solver = jax.vmap(lambda x: bicgstab(self.space.l2_mass_matrix, x, tol=1e-14)[0], in_axes=0)
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
    
    def __init__(
            self,
            picklepath: str = None,
            r1: int = 5,
            r3: int = 5,
            **kwargs,
            ):
        
        super().__init__(**kwargs)
        
        # store reduced order dimensions
        self.r1 = r1
        self.r3 = r3
        
        # simulate the system to obtain snapshot matrix
        T = 1.5
        nt = int(T*100)+1
        tt = jnp.linspace(0, T, nt)
        degree = 3
        num_quad_nodes = 2*degree
        num_proj_nodes = 2*degree
        picklename = f'{picklepath}_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}' # needs to be updated
        
        try: # try to skip also the evaluation
            with open(f'{picklename}.pickle','rb') as f:
                proj_solution = pickle.load(f)['proj_solution']
            print(f'\tFOM result was loaded')

        except FileNotFoundError:
            s = timer()
            proj_solution = projection_method(
                CahnHilliard(**kwargs), # fom
                tt,
                z0=self.initial_condition,
                control=self.default_control,
                degree=degree,
                num_quad_nodes=2*degree,
                num_proj_nodes=2*degree,
                # debug=True,
                )
            e = timer()
            print(f'[cahn hilliard rom] full order simulation took {e-s:.2f} seconds')
            
            # save file
            if picklepath is not None: # save at valid path
                with open(f'{picklename}.pickle','wb') as f:
                    pickle.dump({'proj_solution': proj_solution},f)
                print(f'\tFOM result was written')
        
        _, zz, dt_zz = proj_solution['boundaries']
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
    
    
    
    
    
    
    
    
    
