#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements the AC/DC converter example
#
#


import jax
import jax.numpy as jnp

# energy based model
from helpers.energy_based_model import EnergyBasedModel_LinearJR

# newton solver
from helpers.newton import newton

# other
from helpers.other import dprint
from timeit import default_timer as timer

class ACDC(EnergyBasedModel_LinearJR):

    def __init__(
            self,
            ):
        
        # matrices were obtained from personal correspondence with T. Reis
        self.A_V = jnp.array([-1, 0, 0, 0]).reshape((-1,1))
        self.A_R = jnp.array([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, -1, 0],
            [0, -1, 0, 0, -1]
            ])
        self.A_C = jnp.array([0, -1, 0, 1]).reshape((-1,1))
        self.A_CV = jnp.hstack((self.A_C, self.A_V))
        self.G = jnp.eye(5) # current voltage relation at resistances
        
        self.A = jnp.block([
            [jnp.zeros((2,2)), self.A_CV.T],
            [-self.A_CV, -self.A_R @ self.G @ self.A_R.T]
            ])
        
        J_matrix = 1/2 * (self.A - self.A.T)
        R_matrix = - 1/2 * (self.A + self.A.T)
        self.B_matrix = jnp.zeros((6,1)).at[1,0].set(-1)
        
        super().__init__(J_matrix=J_matrix, R_matrix=R_matrix)
        
        self.dims = (1, 0, 5)
        
        self.u_s_init = - jnp.ones((1,))
        q_c_init = jnp.ones((1,))
        i_s_init = jnp.ones((1,))
        phi_init, _, _, _ = jnp.linalg.lstsq(self.A_CV.T, jnp.hstack((self.nabla_1_ham(q_c_init, None), self.u_s_init)))
        
        self.initial_condition = jnp.hstack((q_c_init, i_s_init, phi_init))
        
    def hamiltonian(self, z1, z2):
        # z2 is not needed
        return 1/2 * z1[0]**2 + 1/2 * z1[0]**4
    
    def B(self, u):
        return self.B_matrix @ u
    
    def default_control(self, t):
        return (jnp.sin(t) + self.u_s_init).reshape((-1,1))
    
    def get_manufactured_solution(self):
        
        def control_manufactured_solution(t):
            return jnp.sin(t).reshape((-1,1)) - 1
        
        def q_c_manufactured(t):
            return jnp.cos(t).reshape((-1,1))
    
        def dt_q_c_manufactured(t):
            return -jnp.sin(t).reshape((-1,1))
        
        def i_s_manufactured(t):
            return jnp.sin(t).reshape((-1,1))
        
        def manufactured_solution(t):
            # solve for phi via least squares problem
            phi = lambda s: jnp.linalg.lstsq(self.A_CV.T, jnp.hstack((self.nabla_1_ham(q_c_manufactured(s).reshape((-1,)), None), control_manufactured_solution(s).reshape((-1,)))))[0]
            phi_vmap = jax.vmap(phi, in_axes=0)
            # stack
            return jnp.hstack((q_c_manufactured(t), i_s_manufactured(t), phi_vmap(t)))
            
        def g_manufactured_solution(t):
            manu = manufactured_solution(t)
            u = control_manufactured_solution(t)
            q_c_manu = q_c_manufactured(t)
            dt_q_c_manu = dt_q_c_manufactured(t)
            i_s_manu = i_s_manufactured(t)
            phi_manu = manu[:,2:]
            lhs = jnp.hstack((self.nabla_1_ham_vmap(q_c_manu, None), jnp.zeros((t.shape[0],5))))
            rhs = jnp.einsum('nm,...m->...n', self.J_matrix - self.R_matrix, jnp.hstack((dt_q_c_manu, i_s_manu, phi_manu))) + self.B_vmap(u)
            return lhs - rhs
            
        z0 = manufactured_solution(jnp.array([0.]).reshape((-1,1)))[0,:]
        
        return z0, manufactured_solution, control_manufactured_solution, g_manufactured_solution
    
    
    
if __name__ == "__main__":
    
    jax.config.update("jax_enable_x64", True)
    
    from main.time_discretization import projection_method
    
    import matplotlib.pyplot as plt
    from helpers.other import mpl_settings
    mpl_settings()
    
    # create ACDC converter model
    ebm = ACDC()
    
    # simulation settings
    T = 5
    nt = 100
    tt = jnp.linspace(0,T,nt+1)
    
    # # run simulation
    # proj_sol = projection_method(
    #     ebm=ebm,
    #     tt=tt,
    #     z0=ebm.initial_condition,
    #     control=ebm.default_control,
    #     degree=3,
    #     num_quad_nodes=None,
    #     num_proj_nodes=None,
    #     debug=True,
    #     )
    # zz_proj = proj_sol['boundaries'][1]
    #
    # # plot
    # plt.plot(tt, zz_proj, label=[f'$z_{i}$' for i in range(6)])
    # plt.legend()
    # # plt.scatter(tt, 0*tt, marker='o')
    # plt.tight_layout()
    # plt.show()
    #
    # run simulation for manufactured solution
    z0_manufactured_solution, manufactured_solution, control_manufactured_solution, g_manufactured_solution = ebm.get_manufactured_solution()
    k = 3
    proj_sol = projection_method(
        ebm=ebm,
        tt=tt,
        z0=z0_manufactured_solution,
        control=control_manufactured_solution,
        degree=k,
        num_quad_nodes=None,
        num_proj_nodes=2*k,
        debug=True,
        g_manufactured_solution=g_manufactured_solution,
        )
    zz_proj = proj_sol['boundaries'][1]

    # plot computed solution
    plt.plot(tt, zz_proj, label=[f'$z_{i}$' for i in range(6)])
    plt.legend()
    # plt.scatter(tt, 0*tt, marker='o')
    plt.title('computed solution')
    plt.tight_layout()
    plt.show()
    
    # plot manufactured solution
    plt.plot(tt, manufactured_solution(tt), label=[f'$z_{i}$' for i in range(6)])
    plt.legend()
    # plt.scatter(tt, 0*tt, marker='o')
    plt.title('manufactured solution')
    plt.tight_layout()
    plt.show()
    
    # plot error
    plt.plot(tt, zz_proj - manufactured_solution(tt), label=[f'$e_{i}$' for i in range(6)])
    plt.legend()
    plt.title('error')
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    