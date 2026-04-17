#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a doubly nonlinear parabolic equation
# as an example of a nonlinear system where the projection in
# time can not be omitted.
#
#


import jax
import jax.numpy as jnp


# plotting
import matplotlib.pyplot as plt

# file saving
import pickle

# timing
from timeit import default_timer as timer

# energy based model
from helpers.energy_based_model import EnergyBasedModel_LinearJR, EnergyBasedModel

# discretization
from main.space_discretization import AnsatzSpace1D
from main.time_discretization import projection_method, implicit_midpoint

# sparse computations
from jax.scipy.sparse.linalg import bicgstab

# helpers
from helpers.other import dprint


class DoublyNonlinearParabolic(EnergyBasedModel):
    """
    implements
    
        dt z = dx beta( dx alpha(z) ) + u_1
    
    on Omega = [0,L] with boundary conditions
    
        beta(dx alpha(z)) = u_2 on partial Omega
        
    with
        
        alpha(z) = (|z|^{q-1} + eps_alpha) z
        beta(z) = (|z|^{p-2} + eps_beta) z
    
    here, eps_alpha and eps_beta are small regularization parameters.
    
    we consider u_1 = 0 and u_2 = 0.
    """
    
    def __init__(
            self,
            mesh_settings: dict = None,
            p: int | float = 1.5,
            q: int | float = 1.5,
            nx: int = 25,
            eps_alpha: float = 0.0,
            eps_beta: float = 1e-307, # smallest number that can be represented, check sys.float_info.min. this is to avoid jax compilation issues!
            ):
        
        # save parameters
        self.p = p
        self.q = q
        self.eps_alpha = eps_alpha # regularization parameter for alpha
        self.eps_beta = eps_beta # regularization parameter for beta
        
        # define ansatz space for space discretization
        if mesh_settings is None:
            mesh_settings = {
                'L': 1.0,
                }
        
        mesh_settings['n'] = nx
        
        self.space = AnsatzSpace1D(
            mesh_settings=mesh_settings,
            inner_product=f'W1,{self.p}',
            )
        
        super().__init__()
        
        # set PDE flag
        self.was_pde = True
        
        # set dimensions
        self.dims = (0, self.space.dim, 0) # 0 for z1 and z3 dims
        
        # initial condition - fits default manufactured solution
        self.initial_condition = self.space.get_projection_coeffs(jnp.cos(4*jnp.pi*self.space.mapped_quad_nodes/self.space.mesh.L), inner_product='L2')
        
    def J(self, dt_z1, h2, z3):
        return jnp.zeros_like(h2)
        
    def R(self, dt_z1, h2, z3):
        """
        Compute the nonlinear operator:
        <r(v), φ_i> = ∫_Ω |∇v|^{p-2} ∇v · ∇φ_i dx
        
        for v = h2
        """
        
        # apply inverse mass matrix to preserve structure
        M2inv_h2, _ = bicgstab(self.space.l2_mass_matrix, h2,
                            tol=1e-14, atol=1e-14, maxiter=1000)
        
        # evaluate v and grad v at quadrature points
        _, grad_v_quad = self.space.eval_coeffs_quad(M2inv_h2)
        # shapes: v_quad (num_elements, num_quad_nodes)
        #         grad_v_quad (num_elements, num_quad_nodes)
        
        # compute |∇v|^{p-2}
        grad_v_norm = jnp.abs(grad_v_quad)  # in 1D, |∇v| = |dv/dx|
        
        
        # compute |∇v|^{p-2} ∇v
        r_grad_v = (grad_v_norm + self.eps_beta)**(self.p - 2.0) * grad_v_quad
        # shape: (num_elements, num_quad_nodes)
        
        # multiply with ∇φ_i and integrate
        # r_grad_v * grad_gbf_i, summed over quadrature points
        r_grad_v_grad_gbf = jnp.einsum('eq,beq->beq', r_grad_v, self.space.grad_gbf_quad)
        # shape: (dim, num_elements, num_quad_nodes)
        
        # integrate using quadrature
        r_v = jnp.sum(self.space.quadrature_with_values_physical(r_grad_v_grad_gbf), axis=1)
        # shape: (dim,)
        
        # apply inverse mass matrix to preserve structure
        M2inv_r_v, _ = bicgstab(self.space.l2_mass_matrix, r_v,
                            tol=1e-14, atol=1e-14, maxiter=1000)
        
        return M2inv_r_v
    
    def hamiltonian(self, z1, z2):
        """
        Energy functional:
        H(z) = (1/(q+1)) ∫_Ω |z|^{q+1} dx
        
        Args:
            z1: not used (empty)
            z2: coefficients for z
            
        Returns:
            scalar energy value
        """
        # evaluate z at quadrature points
        z_quad, _ = self.space.eval_coeffs_quad(z2)
        # shape: (num_elements, num_quad_nodes)
        
        # compute 1/(q+1) * |z|^{q+1} + 1/2 * eps_alpha z^2    -> alpha = (|z|^{q-1} + eps_alpha) z
        z_abs = jnp.abs(z_quad)
        integrand = (1/(self.q + 1)) * z_abs**(self.q + 1.0) \
                    + 1/2 * self.eps_alpha * z_quad**2
        # shape: (num_elements, num_quad_nodes)
        
        # integrate using quadrature
        energy = jnp.sum(self.space.quadrature_with_values_physical(integrand))
        
        return energy
     
    def B(self, u):
        """
        Control input operator for 1D with Neumann boundary conditions:
        <G u(t), φ> := ∫_Ω u_1(t) φ dx - u_2(L) φ(L) + u_2(0) φ(0)
        
        The boundary terms come from integration by parts in the weak formulation.
        
        Args:
            u: control input, shape (..., dim + 2)
               u[..., :dim] = u_1 (distributed control coefficients)
               u[..., dim] = u_2(0) (left boundary flux control)
               u[..., dim+1] = u_2(L) (right boundary flux control)
            
        Returns:
            B_u: vector of shape (..., dim)
        """
        # Single control input
        u_1_coeffs = u[:self.space.dim]  # distributed control
        u_2_left = u[self.space.dim]      # left boundary control u_2(0)
        u_2_right = u[self.space.dim+1]   # right boundary control u_2(L)
        
        # Domain integral: ∫_Ω u_1 φ_i dx
        domain_contribution = self.space.l2_mass_matrix @ u_1_coeffs
        
        # Boundary contribution from weak formulation
        # In 1D: u_2(L) * φ_i(L) - u_2(0) * φ_i(0)
        boundary_contribution = jnp.zeros(self.space.dim)
        
        # Left boundary (x=0): subtract u_2(0) because of outward normal direction
        # For left boundary, outward normal is -1, so we get u_2(0) * φ_i(0)
        boundary_contribution = boundary_contribution.at[0].set(u_2_left)
        
        # Right boundary (x=L): add u_2(L) because of outward normal direction
        # For right boundary, outward normal is +1, so we get -u_2(L) * φ_i(L)
        boundary_contribution = boundary_contribution.at[-1].set(-u_2_right)
        
        b = domain_contribution + boundary_contribution
        
        # apply inverse mass matrix to preserve structure
        M2inv_b, _ = bicgstab(self.space.l2_mass_matrix, b,
                            tol=1e-14, atol=1e-14, maxiter=1000)
        
        return M2inv_b
    
    def default_control(self, t):
        return jnp.zeros((t.shape[0], self.space.dim+2)) # distributed control part u1 and boundary control part u2

    def get_manufactured_solution(self):
        
        def z_spatial(x):
            return jnp.cos(4*jnp.pi*x/self.space.mesh.L)
            
        tscale = 50
        
        def z_manufactured(xy, t):
            return jnp.exp(-tscale * t) * z_spatial(xy)
        
        def dt_z_manufactured(xy, t):
            return - tscale * jnp.exp(-tscale * t) * z_spatial(xy)
        
        z_manufactured_quad = jax.vmap(lambda t: z_manufactured(self.space.mapped_quad_nodes, t), in_axes=0)
        dt_z_manufactured_quad = jax.vmap(lambda t: dt_z_manufactured(self.space.mapped_quad_nodes, t), in_axes=0)
        projection_vmap = jax.vmap(lambda x: self.space.get_projection_coeffs(x, inner_product='L2'), in_axes=0)
        z2 = lambda t: projection_vmap(z_manufactured_quad(t))
        dt_z2 = lambda t: projection_vmap(dt_z_manufactured_quad(t))
        
        # empty array callable
        _ = lambda t: jnp.zeros((t.shape[0], 0)) # empty array
        
        # z1 and z3 are empty array
        z1 = z3 = dt_z1 = _
        
        # build initial condition
        t0 = jnp.zeros((1,))
        z1_0 = z1(t0)[0,:]
        z2_0 = z2(t0)[0,:]
        z3_0 = z3(t0)[0,:]
        z0 = jnp.hstack((z1_0, z2_0, z3_0))
        
        # build zero control input
        u = jnp.zeros((self.space.dim+2,))
        
        # compute g for g_manufactured solution - the latter is a constant map since v_manufactured is constant
        h1 = lambda t: self.nabla_1_ham_vmap(z1(t), z2(t))
        h2 = lambda t: self.nabla_2_ham_vmap(z1(t), z2(t))
        rhs = lambda t: \
            self.J_vmap(dt_z1(t), h2(t), z3(t)) \
            - self.R_vmap(dt_z1(t), h2(t), z3(t)) \
            + self.B(u) # u is constant, no B_vmap needed. otherwise: self.B_vmap(u(t))
        lhs = lambda t: jnp.hstack((h1(t), dt_z2(t), jnp.zeros_like(z3(t))))
        
        manufactured_solution = lambda t: jnp.hstack((z1(t), z2(t), z3(t)))
        control_manufactured_solution = lambda t: jnp.tile(u, (t.shape[0], 1)) # repeats boundary input entries as often as needed, shape (t.shape[0], ncontrol)
        g_manufactured_solution = lambda t: lhs(t) - rhs(t)
        
        return z0, manufactured_solution, control_manufactured_solution, g_manufactured_solution
    
    def visualize_solution(
        self,
        tt: jnp.ndarray,
        zz: jnp.ndarray,
        vmin: float = None,
        vmax: float = None,
        colorbarticks: list = None,
        title: str = None,
        savepath: str = None,
        interpolation: str = 'gaussian',
        ):
        """
        Visualize the solution at different time steps using imshow.
        
        Parameters
        ----------
        interpolation : str, optional
            Interpolation method for imshow. Options include:
            'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', etc.
            Default is 'bilinear'.
        """
        fig, ax = plt.subplots()
    
        # Get spatial and temporal extents
        x_min, x_max = self.space.mesh.vertices.min(), self.space.mesh.vertices.max()
        t_min, t_max = tt.min(), tt.max()
        
        # imshow displays data with origin at top-left by default
        # extent: [left, right, bottom, top]
        im = ax.imshow(
            zz,
            aspect='auto',
            origin='lower',
            extent=[x_min, x_max, t_min, t_max],
            vmin=vmin,
            vmax=vmax,
            cmap='viridis',
            interpolation=interpolation
        )
        
        ax.set_xlabel('space $x$')
        ax.set_ylabel('time $t$')
        plt.colorbar(im, ax=ax, label='$z(t,x)$', ticks=colorbarticks)
        
        if savepath is not None:
            fig.tight_layout()
            fig.savefig(savepath + '.pgf')  # save as pgf
            fig.savefig(savepath + '.png')  # save as png
            print(f'figure saved under savepath {savepath} (as pgf and png)')
        
        if title is not None:
            ax.set_title(title)
    
        fig.tight_layout()
        fig.show()
    
        return fig
    
class DoublyNonlinearParabolicReducedOrder(DoublyNonlinearParabolic):
    
    def __init__(
            self,
            reduced_order: int = 5,
            picklepath: str = None,
            **kwargs,
            ):
        
        super().__init__(**kwargs)
        
        # store reduced order dimension
        self.reduced_order = reduced_order
        
        # simulate the system to obtain snapshot matrix
        T = 0.1
        nt = 501 # fine discretization
        tt = jnp.linspace(0, T, nt)
        degree = 4
        num_quad_nodes = 2*degree
        num_proj_nodes = 2*degree
        picklename = f'{picklepath}_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}' # needs to be updated

        try: # try to skip also the evaluation
            with open(f'{picklename}.pickle','rb') as f:
                proj_solution = pickle.load(f)['proj_solution']
            print(f'\tFOM result was loaded')

        except FileNotFoundError: # evaluation was not done before
            s = timer()
            proj_solution = projection_method(
                ebm=DoublyNonlinearParabolic(**kwargs),
                tt=tt,
                z0=self.initial_condition,
                control=self.default_control,
                degree=degree,
                num_quad_nodes=num_quad_nodes,
                num_proj_nodes=num_proj_nodes,
                debug=False,
                )
            e = timer()
            print(f'[doubly nonlinear parabolic rom] full order simulation took {e-s:.2f} seconds')

            # save file
            if picklepath is not None: # save at valid path
                with open(f'{picklename}.pickle','wb') as f:
                    pickle.dump({'proj_solution': proj_solution},f)
                print(f'\tFOM result was written')
    
        _, zz, dt_zz = proj_solution['boundaries']
        Q2 = zz.T # snapshot matrix for z2 states - time index in second position
        U2, _, _ = jnp.linalg.svd(Q2, full_matrices=False)
        self.V2 = U2[:,:self.reduced_order]
        
        # build reduced order matrix
        self.V1 = self.V3 = jnp.eye(0) # self.dims[0] == self.dims[2] == 0
        self.V = jax.scipy.linalg.block_diag(self.V1, self.V2, self.V3)
        
        # update state dimension
        self.dims = (self.V1.shape[1], self.V2.shape[1], self.V3.shape[1])
        
        # update initial condition
        self.initial_condition = self.V.T @ self.initial_condition
        
        # set rom flag
        self.is_rom = True
    
    def visualize_solution(
            self,
            zz: jnp.ndarray,
            **kwargs,
            ):
        
        Vzz = jnp.einsum('nr,tr->tn', self.V, zz)
        
        return super().visualize_solution(
            zz=Vzz,
            **kwargs,
            )
        
    def hamiltonian(self, z1, z2):
        return super().hamiltonian(self.V1 @ z1, self.V2 @ z2)
    
    def J(self, dt_z1, h2, z3):
        return self.V.T @ super().J(self.V1 @ dt_z1, self.V2 @ h2, self.V3 @ z3)
    
    def R(self, dt_z1, h2, z3):
        return self.V.T @ super().R(self.V1 @ dt_z1, self.V2 @ h2, self.V3 @ z3)
        
    def B(self, u):
        return self.V.T @ super().B(u)
