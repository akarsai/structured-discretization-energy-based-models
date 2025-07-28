from jax.experimental import sparse
import jax.numpy as jnp
from helpers.other import dprint

def sparse_frobenius_norm(sparse_array: sparse.BCOO | sparse.BCSR) -> jnp.ndarray:
    # sparse_array = sparse_array.sum_duplicates()
    # without the line above, the code is not correct.
    # we still do not use it as this leads to difficulties with jax.jit
    return jnp.sqrt(jnp.einsum('...,...->', sparse_array.data, sparse_array.data))

def bcoo_diagonal(M: sparse.BCOO) -> jnp.ndarray:
    M = M.sum_duplicates()
    row, col = M.indices.T
    mask = row == col
    return jnp.zeros(M.shape[0]).at[row[mask]].add(M.data[mask])


if __name__ == "__main__":
    
    from main.space_discretization import AnsatzSpace
    
    space = AnsatzSpace(
        mesh_settings={'nx': 3, 'ny': 3}
        )
    
    # I = jnp.eye(n)
    # I_sparse = sparse.BCOO.fromdense(I)
    M = space.l2_mass_matrix
    
    norm = jnp.linalg.norm(M.todense())
    sp_norm = sparse_frobenius_norm(M)
    
    dprint((norm - sp_norm)/norm, format='.5f')
    
    diag = bcoo_diagonal(M)
    dprint(1/diag)