import jax
import jax.numpy as jnp
import matplotlib

class EnergyBasedModel:
    """
    implements the energy based model class used in the paper
    
    [ nabla_1 ham ]                [    dt z1    ]
    [    dt z2    ]  =  ( J - R )  [ nabla_2 ham ] + B u
    [      0      ]                [     z3      ]
    
    we use the notation
    
    dt_z1 = d/dt z1
    h2 = nabla_z2 ham(z1,z2)
    
    the B operator is state independent in all examples.
    """

    def __init__(
            self,
            initial_condition = None,
            info = None,
            ):

        self.initial_condition = initial_condition
        self.info = info
        self.dims = None
        self.was_pde = False
        self.is_rom = False
        
    def J(self, dt_z1, h2, z3):
        raise NotImplementedError
    
    def R(self, dt_z1, h2, z3):
        raise NotImplementedError
    
    def B(self, u):
        raise NotImplementedError
        
    def hamiltonian(self, z1, z2):
        raise NotImplementedError
    
    @property
    def nabla_1_ham(self):
        return jax.grad(self.hamiltonian, argnums=0)
    
    @property
    def nabla_2_ham(self):
        return jax.grad(self.hamiltonian, argnums=1)
    
    # vmap everything
    @property
    def J_vmap(self):
        return jax.vmap(self.J, in_axes=0)

    @property
    def R_vmap(self):
        return jax.vmap(self.R, in_axes=0)
    
    @property
    def B_vmap(self):
        return jax.vmap(self.B, in_axes=0)
    
    @property
    def hamiltonian_vmap(self):
        return jax.vmap(self.hamiltonian, in_axes=(0,0))
    
    @property
    def nabla_1_ham_vmap(self):
        return jax.vmap(self.nabla_1_ham, in_axes=0)

    @property
    def nabla_2_ham_vmap(self):
        return jax.vmap(self.nabla_2_ham, in_axes=0)
    
    def visualize_hamiltonian(
            self,
            tt: jnp.ndarray,
            zz: jnp.ndarray | list[jnp.ndarray],
            label: str | list[str] = r'solution',
            title: str = None,
            savepath: str = None,
            ):
        import matplotlib.pyplot as plt
        
        # if isinstance(tt, jnp.ndarray): tt = [tt]
        if isinstance(zz, jnp.ndarray): zz = [zz]
        if isinstance(label, str): label = [label]
        
        d1, d2 = self.dims[0], self.dims[1]
        
        fig, ax = plt.subplots()
        
        for i, _zz in enumerate(zz):
            _zz1 = _zz[:,:d1]
            _zz2 = _zz[:,d1:d1+d2]
            hh = self.hamiltonian_vmap(_zz1, _zz2)
            ax.plot(tt, hh, label=label[i])
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\mathcal{H}(z_1(t), z_2(t))$')
        ax.legend()
        if savepath is not None:
            fig.tight_layout()
            fig.savefig(savepath+'.pgf')
            print(f'saved figure to {savepath}.pgf')
        if title is not None:
            ax.set_title(title)
            fig.tight_layout()
        plt.show()
        return
    
class EnergyBasedModel_LinearJR(EnergyBasedModel):
    
    def __init__(
            self,
            J_matrix,
            R_matrix,
            **kwargs,
            ):
        
        self.J_matrix = J_matrix
        self.R_matrix = R_matrix
        
        super().__init__(**kwargs)
        
    def J(self, dt_z1, h2, z3):
        return self.J_matrix @ jnp.hstack((dt_z1, h2, z3))
    
    def R(self, dt_z1, h2, z3):
        return self.R_matrix @ jnp.hstack((dt_z1, h2, z3))
    
class EnergyBasedModel_LinearJRQ(EnergyBasedModel_LinearJR):
    
    def __init__(
            self,
            Q_matrix,
            **kwargs,
            ):
        
        self.Q_matrix = Q_matrix
        
        super().__init__(**kwargs)
        
    def hamiltonian(self, z1, z2):
        z12 = jnp.hstack((z1,z2))
        return 1/2 * z12.T @ self.Q_matrix @ z12
    

if __name__ == '__main__':
    
    ebm = EnergyBasedModel()
    ebm.hamiltonian(None, None)