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

class NonlinearCircuit(EnergyBasedModel_LinearJR):

    def __init__(
            self,
            ):
        
        self.A_C = jnp.array([
            [0, 0],
            [0,-1],
            [0, 0],
            [0, 0],
            [1, 1]
            ])
        self.A_S = jnp.array([0, 0, 0, 0, -1]).reshape((-1,1))
        self.A_R = jnp.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1]
            ])
        self.A_L = jnp.array([
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
            [0, 0]
            ])
        self.A_CLS = jnp.hstack((self.A_C, self.A_L, self.A_S))
        self.G = jnp.eye(3) # current voltage relation at resistances
        
        self.A = jnp.block([
            [jnp.zeros((5,5)), self.A_CLS.T],
            [-self.A_CLS, -self.A_R @ self.G @ self.A_R.T]
            ])
        
        J_matrix = 1/2 * (self.A - self.A.T)
        R_matrix = - 1/2 * (self.A + self.A.T)
        self.B_matrix = jnp.zeros((10,1)).at[4,0].set(-1)
        
        super().__init__(J_matrix=J_matrix, R_matrix=R_matrix)
        
        self.dims = (2, 2, 6)
        
        self.u_s_init = jnp.zeros((1,)) # u_s(t) = sin(t), u_s(0) = 0
        q_c_init = jnp.zeros((2,))
        psi_l_init = jnp.zeros((2,))
        i_s_init = - jnp.ones((1,)) # i_s(0) = - dot u_s(0) = -cos(0) = -1
        phi_init = jnp.zeros((5,))
        # derivative initial conditions - needed for the radau solver
        dt_q_c_init = -jnp.eye(2)[:,0]  # consistent
        dt_psi_l_init = 0 * dt_q_c_init # consistent
        dt_i_s_init = jnp.zeros((1,)) # does not matter
        dt_phi_init = jnp.zeros((5,)) # does not matter
        
        self.initial_condition = jnp.hstack((q_c_init, psi_l_init, i_s_init, phi_init))
        self.derivative_initial_condition = jnp.hstack((dt_q_c_init, dt_psi_l_init, dt_i_s_init, dt_phi_init))
        
    def hamiltonian(self, z1, z2):
        # z1 = q_c, z2 = psi_l
        norm_z1_squared = z1.T @ z1
        norm_z2_squared = z2.T @ z2
        return 1/2 * norm_z1_squared + 1/2 * (norm_z2_squared + norm_z2_squared**2)
    
    def B(self, u):
        return self.B_matrix @ u
    
    def default_control(self, t):
        return (jnp.sin(t) + self.u_s_init).reshape((-1,1))

    def get_manufactured_solution(self):
        
        def control_manufactured_solution(t):
            return jnp.sin(t).reshape((-1,1)) - 1
        
        def q_c_manufactured(t): # 2
            return jnp.repeat(jnp.cos(t)[:, None], 2, axis=1)
    
        def dt_q_c_manufactured(t): # 2
            return -jnp.repeat(jnp.sin(t)[:, None], 2, axis=1)
        
        def psi_l_manufactured(t): # 2
            return jnp.repeat(jnp.cos(t)[:, None], 2, axis=1)
        
        def dt_psi_l_manufactured(t): # 2
            return -jnp.repeat(jnp.sin(t)[:, None], 2, axis=1)
        
        def i_s_manufactured(t): # 1
            return jnp.sin(t).reshape((-1,1))
        
        def phi_manufactured(t): # 5
            return jnp.repeat(jnp.sin(t)[:, None], 5, axis=1)
        
        def manufactured_solution(t):
            # no least squares solve
            return jnp.hstack((q_c_manufactured(t), psi_l_manufactured(t), i_s_manufactured(t), phi_manufactured(t)))
            
        def g_manufactured_solution(t):
            manu = manufactured_solution(t)
            u = control_manufactured_solution(t)
            q_c_manu = q_c_manufactured(t)
            dt_q_c_manu = dt_q_c_manufactured(t)
            psi_l_manu = psi_l_manufactured(t)
            dt_psi_l_manu = dt_psi_l_manufactured(t)
            i_s_manu = i_s_manufactured(t)
            phi_manu = phi_manufactured(t)
            lhs = jnp.hstack((self.nabla_1_ham_vmap(q_c_manu, psi_l_manu), dt_psi_l_manu, jnp.zeros((t.shape[0],6))))
            rhs = jnp.einsum('nm,...m->...n', self.J_matrix - self.R_matrix, jnp.hstack((dt_q_c_manu, self.nabla_2_ham_vmap(q_c_manu, psi_l_manu), i_s_manu, phi_manu))) + self.B_vmap(u)
            return lhs - rhs
            
        z0 = manufactured_solution(jnp.array([0.]).reshape((-1,)))[0,:]
        
        return z0, manufactured_solution, control_manufactured_solution, g_manufactured_solution
    
    