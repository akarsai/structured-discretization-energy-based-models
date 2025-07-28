#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a piecewise linear ansatz space
# for 2D finite elements using a uniform triangulation
#
#

# essentials
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jax.scipy.sparse.linalg import bicgstab
import numpy as np
import basix # pip install fenics-basix

# plotting
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata

from helpers.triangle import map_points_to_triangle, get_triangle_jacobian_and_area, get_triangle_quadrature_points_and_weights
from helpers.other import dprint, plot_matrix, style


class Mesh:

    def __init__(
            self,
            cell_type: str = 'triangle',
            Lx: float = 1.0,
            Ly: float = 1.0,
            nx: int = 10,
            ny: int = 10,
            ):
        
        assert Lx == Ly, 'only square meshes are supported'
        assert nx == ny, 'only uniform meshes are supported'

        self.cell_type = cell_type
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        
        # calculate mesh width
        self.hx = self.Lx / self.nx
        self.hy = self.Ly / self.ny
        self.h = self.hx # only for square uniform meshes

        # assert cell_type
        if self.cell_type != 'triangle':
            raise ValueError('only triangle cells are supported')

        # calculate vertex coordinates
        x = jnp.linspace(0, self.Lx, self.nx + 1)
        y = jnp.linspace(0, self.Ly, self.ny + 1)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        self.vertices = jnp.column_stack((X.flatten(), Y.flatten()))

        # calculate triangle vertex indices, two triangles per quadrilateral
        triangle_vertices_indices_list = []
        for j in range(self.ny):
            for i in range(self.nx):
                v0 = j * (self.nx + 1) + i
                v1 = v0 + 1
                v2 = v0 + (self.nx + 1)
                v3 = v2 + 1
                triangle_vertices_indices_list.append([v0, v1, v2]) # first triangle
                triangle_vertices_indices_list.append([v1, v2, v3]) # second triangle
        self.triangle_vertices_indices_list = triangle_vertices_indices_list

        # calculate triangle vertices list
        self.triangle_vertices_list = jnp.array([[self.vertices[l] for l in k] for k in triangle_vertices_indices_list])
        
        # get number of triangles
        self.num_triangles = self.triangle_vertices_list.shape[0] # or self.nx * self.ny * 2
        
        # get total number of (unique) triangle vertices
        self.num_vertices = self.vertices.shape[0] # or self.nx * self.ny * 4, or jnp.max(self.triangle_vertices_indices_list) + 1
        
        # boundary edges and boundary vertices
        self.boundary_edges = self.get_boundary_edges()
        self.num_boundary_edges = self.boundary_edges.shape[0]
        # exclude a vertex on each edge to ensure uniqueness.
        self.lower_vertices = jnp.arange(nx+1)[:-1] # lower boundary (y=0) nodes: 0, 1, ..., nx
        self.upper_vertices = jnp.arange(ny*(nx+1), ny*(nx+1)+nx+1)[1:] # upper boundary (y=Ly) nodes: ny*(nx+1), ..., ny*(nx+1)+nx
        self.left_vertices = jnp.arange(0, (ny+1)*(nx+1), nx+1)[1:] # left boundary (x=0) nodes: 0, nx+1, 2(nx+1), ..., ny(nx+1)
        self.right_vertices = jnp.arange(nx, (ny+1)*(nx+1), nx+1)[:-1] # right boundary (x=Lx) nodes: nx, nx+(nx+1), ..., nx+ny(nx+1)
        # combine all boundary nodes. this has no duplicates
        self.boundary_vertices = jnp.sort(jnp.concatenate((self.lower_vertices,self.right_vertices,self.upper_vertices,self.left_vertices)))
        self.num_boundary_vertices = self.boundary_vertices.size
        
        # create lists of attached triangles
        self.lower_edge_triangles = [2*i for i in range(self.nx)]
        self.upper_edge_triangles = [2*((self.ny-1)*self.nx+i)+1 for i in range(self.nx)]
        self.left_edge_triangles = [2*(j*self.nx) for j in range(self.ny)]
        self.right_edge_triangles = [2*((j+1)*self.nx-1)+1 for j in range(self.ny)]
        
        # # debug prints
        # dprint(self.lower_vertices)
        # dprint(self.upper_vertices)
        # dprint(self.left_vertices)
        # dprint(self.right_vertices)
        # dprint(self.lower_edge_triangles)
        # dprint(self.upper_edge_triangles)
        # dprint(self.left_edge_triangles)
        # dprint(self.right_edge_triangles)
        
    def get_boundary_edges(self):
        """Calculate boundary edges for structured rectangular mesh"""
        boundary_edges = []
        nx, ny = self.nx, self.ny
        
        # Bottom boundary (y=0)
        for i in range(nx):
            boundary_edges.append((i, i+1))
            
        # Right boundary (x=Lx)
        for j in range(ny):
            node = nx + j*(nx+1)
            boundary_edges.append((node, node + (nx+1)))
            
        # Top boundary (y=Ly)
        top_start = ny*(nx+1)
        for i in range(nx):
            boundary_edges.append((top_start + i + 1, top_start + i))
            
        # Left boundary (x=0)
        for j in range(ny):
            node = (j+1)*(nx+1)
            boundary_edges.append((node, node - (nx+1)))
            
        return jnp.array(boundary_edges)
        
    def map_local_points_to_mesh(self, local_points: jnp.ndarray) -> jnp.ndarray:
        num_points = local_points.shape[0]
        mapped_points = np.zeros((self.num_triangles, num_points, 2))
        for triangle_index, triangle_vertices in enumerate(self.triangle_vertices_list):
            # map quadrature points to triangle
            mapped = map_points_to_triangle(triangle_vertices, local_points)
            mapped_points[triangle_index,:,:] = mapped
        return jnp.array(mapped_points) # return has shape (t, p, 2) = (num_triangles, num_points, 2)

    def show(self, suppress=False):

        fig, ax = plt.subplots()

        # convert JAX arrays to numpy
        vertices = np.array(self.vertices)
        triangle_vertices_indices_list = np.array(self.triangle_vertices_indices_list)

        # plot mesh
        for triangle_vertices_indices in triangle_vertices_indices_list:
            # get vertices of the element
            triangle_vertices = vertices[triangle_vertices_indices]
            # close the triangle
            triangle_vertices = np.vstack([triangle_vertices, triangle_vertices[0]])
            # plot edges
            ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], 'k-', alpha=0.1, linewidth=0.5)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('finite element mesh')
        ax.set_aspect('equal')

        if not suppress:
            plt.tight_layout()
            plt.show()

        return fig, ax

class AnsatzSpace:

    def __init__(
            self,
            mesh: Mesh = None,
            mesh_settings: dict = None,
            degree: int = 1,
            inner_product: str = 'H1'
            ):

        if mesh is None:
            if mesh_settings is None:
                mesh_settings = {}
            mesh = Mesh(**mesh_settings)
            
        self.mesh = mesh
        self.degree = degree
        self.inner_product = inner_product # inner product, L2 or H1

        if self.mesh.cell_type != 'triangle':
            raise ValueError('only triangle cells are supported')

        if not self.degree == 1:
            raise ValueError('degree must be 1')

        if self.inner_product not in ['H1', 'L2']:
            raise ValueError('inner product must be H1 or L2')

        # create basis functions on one triangle using basix
        self.tri_basis = basix.create_element(
            basix.ElementFamily.P,
            basix.CellType.triangle,
            self.degree
        )
        # self.tri_basis.dim = number of basis functions on each element
        self.dim = self.mesh.num_vertices # total dimension of the ansatz space
        
        # create lookup table for triangle vertex indices
        self.triangle_vertex_indices_lookup = jnp.array(self.mesh.triangle_vertices_indices_list, dtype=jnp.int32)
        # usage: self.triangle_vertex_indices_lookup[triangle_idx, basis_idx] = index of basis_idx-th vertex of triangle_idx-th triangle

        # calculate quadrature points and weights for quadrature on the unit triangle
        self.quad_nodes_unit_triangle, self.quad_weights_unit_triangle = get_triangle_quadrature_points_and_weights()
        self.num_quad_nodes = self.quad_weights_unit_triangle.size
        # map them to the mesh and create a global list of quad points
        self.mapped_quad_nodes = self.mesh.map_local_points_to_mesh(self.quad_nodes_unit_triangle)
        
        # store edge midpoints on unit triangle
        self.edge_midpoints_unit_triangle = jnp.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        
        # # calculate fine points in unit triangle for visualization
        # self.fine_points_unit_triangle = basix.create_lattice(self.tri_basis.cell_type, 5, basix.LatticeType.equispaced, exterior=True)
        # self.num_fine_points = self.fine_points_unit_triangle.shape[0]
        # # map them to the mesh and create a global list of finely spaced points
        # self.mapped_fine_points = self.mesh.map_local_points_to_mesh(self.fine_points_unit_triangle)

        # calculate local basis functions on quadrature points on the unit triangle
        # lbf = local basis functions
        self.lbf_unit_triangle, self.grad_lbf_unit_triangle = self.get_local_basis_functions(
            self.quad_nodes_unit_triangle
            )

        # prepare for jacobian compensation when evaluating gradients
        # this simplification only works because the triangulation is uniform!
        jacobian_even, self.triangle_area = get_triangle_jacobian_and_area(self.mesh.triangle_vertices_list[0])
        jacobian_odd, _ = get_triangle_jacobian_and_area(self.mesh.triangle_vertices_list[1])
        self.jacobian_inv_even = jnp.linalg.inv(jacobian_even) # even triangles: multiply with first inverse
        self.jacobian_inv_odd = jnp.linalg.inv(jacobian_odd) # odd triangles: multiply with second inverse

        # compute grad lbf_i on the quadrature points of even and odd triangles - again, assumes uniform triangulation
        self.grad_lbf_even = jnp.einsum('nd,bpd->bpn', self.jacobian_inv_even.T, self.grad_lbf_unit_triangle)
        self.grad_lbf_odd = jnp.einsum('nd,bpd->bpn', self.jacobian_inv_odd.T, self.grad_lbf_unit_triangle)
        # for even odd rule in jnp.where
        self.triangle_even = (jnp.arange(self.mesh.num_triangles) % 2 == 0)

        # calculate quadrature quadrature weights for triangles in mesh
        # since the triangles are uniform, it suffices to calculate this once
        self.quad_weights_mapped = 2 * self.triangle_area * self.quad_weights_unit_triangle
        
        # precompute global basis functions on gauss points for quadrature and edge midpoints for boundary control
        # shapes: (self.dim, self.mesh.num_triangles, self.num_quad_nodes), (self.dim, self.mesh.num_triangles, self.num_quad_points, 2)
        self.gbf_quad, self.grad_gbf_quad = self.get_global_basis_functions(self.quad_nodes_unit_triangle)
        # shapes: (self.dim, self.mesh.num_triangles, 3), (self.dim, self.mesh.num_triangles, 3, 2)
        self.gbf_edge, self.grad_gbf_edge = self.get_global_basis_functions(self.edge_midpoints_unit_triangle)
        
        # precompute grad_gbf \dot normal, shape (self.dim, self.num_boundary_edges//4)
        self.lower_grad_gbf_normal = - self.grad_gbf_edge[:,self.mesh.lower_edge_triangles,0,1] # 0 for lower edge, minus sign and index 1 since outward normal vector is [0, -1]
        self.upper_grad_gbf_normal = self.grad_gbf_edge[:,self.mesh.upper_edge_triangles,0,1] # 0 for lower edge, index 1 since outward normal vector is [0, 1]
        self.left_grad_gbf_normal = - self.grad_gbf_edge[:,self.mesh.left_edge_triangles,1,0] # 1 for left edge, minus sign and index 0 since outward unit normal vector is [-1, 0]
        self.right_grad_gbf_normal = self.grad_gbf_edge[:,self.mesh.right_edge_triangles,1,0] # 1 for right edge, index 0 since outward unit normal vector is [1, 0]
        
        # stack to form matrix that can be used to compute boundary edge values from coefficients
        self.boundary_grad_matrix = jnp.hstack((self.lower_grad_gbf_normal, self.right_grad_gbf_normal, self.upper_grad_gbf_normal, self.left_grad_gbf_normal)).T
        # # debug prints
        # dprint(self.lower_grad_gbf_normal.shape)
        # dprint(self.upper_grad_gbf_normal.shape)
        # dprint(self.left_grad_gbf_normal.shape)
        # dprint(self.right_grad_gbf_normal.shape)
        # dprint(self.boundary_grad_matrix.shape)
        
        # assemble sparse mass matrices
        self.l2_mass_matrix = self.get_l2_mass_matrix()
        self.l2_stiffness_matrix = self.get_l2_stiffness_matrix()
        self.mass_matrix = self.l2_mass_matrix
        if self.inner_product == 'H1':
            self.mass_matrix = self.mass_matrix + self.l2_stiffness_matrix
        self.boundary_mass_matrix = self.get_boundary_mass_matrix()
        self.boundary_extension_matrix = self.get_boundary_extension()

    def get_local_basis_functions(
            self,
            points_in_unit_triangle: jnp.ndarray
            ) -> (jnp.ndarray, jnp.ndarray):

        basis_functions_on_unit_triangle_points = self.tri_basis.tabulate(n=1, x=np.array(points_in_unit_triangle)).squeeze(-1) # shape (3, num_points, self.tri_basis.dim) ---- 3 = function values + x derivative values + y derivative values
        basis_functions_on_unit_triangle_points = jnp.einsum('abc->cba', basis_functions_on_unit_triangle_points) # swap axes to match my intuition and cast to jax array, new shape (self.tri_basis.dim, num_points, 3)
        # self.tri_basis_functions_on_unit_triangle[i,j,:] (value, x derivative, y derivative) of i-th basis function on the j-th gauss point
        values = basis_functions_on_unit_triangle_points[:,:,0] # shape (self.tri_basis.dim, num_points)
        grad = basis_functions_on_unit_triangle_points[:,:,1:] # gradients of the basis functions at gauss points, shape (self.tri_basis.dim, num_points, 2)

        return values, grad

    def get_global_basis_functions(
            self,
            points_in_unit_triangle: jnp.ndarray,
            ):
        """
        Args:
            points_in_unit_triangle: points on the unit triangle (0,0), (1,0), (0,1) which are mapped to the local triangles and used to compute the global basis functions.

        Returns:
            gbf_values: global basis functions values on the mapped points, shape (self.dim, self.mesh.num_triangles, num_points)
            gbf_grad: global basis functions values on the mapped points, shape (self.dim, self.mesh.num_triangles, num_points, 2)
        """
        
        assert points_in_unit_triangle.shape[1] == 2
        num_points = points_in_unit_triangle.shape[0]
        
        # get basis functions and gradient on triangle points
        lbf_values, lbf_grad = self.get_local_basis_functions(points_in_unit_triangle)
        # values shape: (num_local_basis, num_points)
        # grad shape: (num_local_basis, num_points, 2)
        
        scaled_grad_lbf_even = jnp.einsum('nd,bpd->bpn', self.jacobian_inv_even.T, lbf_grad)
        scaled_grad_lbf_odd = jnp.einsum('nd,bpd->bpn', self.jacobian_inv_odd.T, lbf_grad)
        
        # convert to jnp array for vectorized operations
        triangles = jnp.array(self.mesh.triangle_vertices_indices_list)  # shape: (num_triangles, 3) or (num_triangles, K)
        
        num_triangles = self.mesh.num_triangles
        num_local_basis = self.tri_basis.dim  # Number of basis functions per triangle
        num_global_basis = self.dim  # Assuming global basis functions correspond to vertices
        
        # initialize arrays for all basis functions
        gbf_values = jnp.zeros((num_global_basis, num_triangles, num_points))
        gbf_grad = jnp.zeros((num_global_basis, num_triangles, num_points, 2))
        
        # Define the body function for the triangle loop
        def triangle_loop_body(triangle_idx, arrays):
            gbf_values, gbf_grad = arrays
            
            # Get global indices of local basis functions for this triangle
            global_indices = triangles[triangle_idx]
            
            # Define the body function for the local basis loop
            def local_basis_loop_body(local_idx, arrays):
                gbf_values, gbf_grad = arrays
                global_idx = global_indices[local_idx]
                
                # Set values for this global basis function
                gbf_values = gbf_values.at[global_idx, triangle_idx, :].set(lbf_values[local_idx, :])
                
                # Set gradients for this global basis function
                grad = jnp.where(
                    self.triangle_even[triangle_idx],
                    scaled_grad_lbf_even[local_idx, :, :],
                    scaled_grad_lbf_odd[local_idx, :, :]
                )
                gbf_grad = gbf_grad.at[global_idx, triangle_idx, :, :].set(grad)
                
                return gbf_values, gbf_grad
            
            # Use fori_loop for the local basis functions
            return jax.lax.fori_loop(
                0, num_local_basis, local_basis_loop_body, (gbf_values, gbf_grad)
            )
        
        # Use fori_loop for the triangles
        gbf_values, gbf_grad = jax.lax.fori_loop(
            0, num_triangles, triangle_loop_body, (gbf_values, gbf_grad)
        )
        
        return gbf_values, gbf_grad
    
    @partial(jax.jit, static_argnums=(0,))
    def quadrature_with_values_mapped(
            self,
            values_at_quad_nodes: jnp.ndarray,
            ):
        return jnp.einsum('...q,q->...', values_at_quad_nodes, self.quad_weights_mapped) # q = self.num_quad_nodes
    
    def get_l2_mass_matrix_naive(self):
        gbf_mat = jnp.einsum('ntq,mtq->nmtq', self.gbf_quad, self.gbf_quad)
        return jnp.sum(self.quadrature_with_values_mapped(gbf_mat), axis=-1)

    def get_l2_mass_matrix(self):
        """
        efficient L2 mass matrix assembly using precomputed local matrices
        for uniform triangulation. optimized with jax.lax.fori_loop.
        """
        
        # local mass matrix pattern (scaled by area)
        local_mass = jnp.array([[2.0, 1.0, 1.0],
                                [1.0, 2.0, 1.0],
                                [1.0, 1.0, 2.0]]) * (self.triangle_area / 12.0)
    
        # preallocate data and indices for sparse matrix
        num_triangles = self.mesh.num_triangles
        data = jnp.tile(local_mass.flatten(), num_triangles)
        
        # create an initial empty indices array with the right shape
        indices_shape = (num_triangles * 9, 2)  # 9 entries per triangle (3x3), each with row and column
        indices = jnp.zeros(indices_shape, dtype=jnp.int32)
        
        # define the function to be executed for each triangle
        def body_fun(t, idx_array):
            nodes = self.triangle_vertex_indices_lookup[t]
            row_indices = jnp.repeat(nodes, 3)
            col_indices = jnp.tile(nodes, 3)
            
            # create the current triangle's 9 pairs of indices
            current_indices = jnp.stack([row_indices, col_indices], axis=1)
            
            # calculate the start index for this triangle in the flattened array
            start_idx = t * 9
            
            # use dynamic_update_slice for trace-compatible updates
            return jax.lax.dynamic_update_slice(
                idx_array,
                current_indices,
                (start_idx, 0)
            )
        
        # use fori_loop to fill the indices array
        indices = jax.lax.fori_loop(
            0,                  # lower bound (inclusive)
            num_triangles,      # upper bound (exclusive)
            body_fun,           # body function
            indices             # initial value
        )
        
        # create BCOO sparse matrix
        M = sparse.BCOO(
            (data, indices),
            shape=(self.dim, self.dim),
            indices_sorted=True,
            unique_indices=True,
            )
        
        return M
    
    def get_l2_stiffness_matrix_naive(self):
        gbf_grad_mat = jnp.einsum('ntqd,mtqd->nmtq', self.grad_gbf_quad, self.grad_gbf_quad)
        return jnp.sum(self.quadrature_with_values_mapped(gbf_grad_mat), axis=-1)
    
    def get_l2_stiffness_matrix(self):
        """
        Efficient L2 stiffness matrix assembly using precomputed gradients and quadrature,
        optimized with jax.lax.fori_loop for trace compatibility.
        """
        
        # For triangular P1 elements, the local stiffness matrix for reference element
        # has a constant pattern that needs to be scaled by the appropriate factors
        # based on the actual triangle geometry.
        
        # Preallocate data and indices for sparse matrix
        num_triangles = self.mesh.num_triangles
        num_local_basis = self.tri_basis.dim  # Should be 3 for P1 triangles
        entries_per_triangle = num_local_basis * num_local_basis  # 9 entries for 3x3 local matrix
        
        # Prepare arrays for BCOO format
        data = jnp.zeros(num_triangles * entries_per_triangle)
        indices = jnp.zeros((num_triangles * entries_per_triangle, 2), dtype=jnp.int32)
        
        # Define the function to compute the local stiffness matrix for a triangle
        def compute_local_stiffness(t):
            # Get global indices for this triangle
            nodes = self.triangle_vertex_indices_lookup[t]
            
            # Prepare local stiffness matrix for this triangle
            local_stiffness = jnp.zeros((num_local_basis, num_local_basis))
            
            # Choose the appropriate precomputed gradient based on even/odd triangle
            grad_lbf = jnp.where(
                self.triangle_even[t],
                self.grad_lbf_even,
                self.grad_lbf_odd
            )
            
            # Compute inner products of gradients and scale by quadrature weights
            for i in range(num_local_basis):
                for j in range(num_local_basis):
                    # Inner product of gradients at quadrature points
                    product = jnp.einsum('pd,pd->p', grad_lbf[i], grad_lbf[j])
                    # Scale by quadrature weights and add to local stiffness matrix
                    local_stiffness = local_stiffness.at[i, j].set(
                        jnp.sum(product * self.quad_weights_mapped)
                    )
            
            return local_stiffness, nodes
        
        # Define the body function for the triangle loop
        def body_fun(t, arrays):
            data, indices = arrays
            
            # Compute local stiffness matrix and get global node indices
            local_stiffness, nodes = compute_local_stiffness(t)
            
            # Flatten the local stiffness matrix
            local_data = local_stiffness.flatten()
            
            # Create row and column indices for this triangle
            row_indices = jnp.repeat(nodes, num_local_basis)
            col_indices = jnp.tile(nodes, num_local_basis)
            current_indices = jnp.stack([row_indices, col_indices], axis=1)
            
            # Calculate start index for this triangle in the flattened arrays
            start_idx = t * entries_per_triangle
            
            # Update data array
            data = jax.lax.dynamic_update_slice(data, local_data, (start_idx,))
            
            # Update indices array
            indices = jax.lax.dynamic_update_slice(indices, current_indices, (start_idx, 0))
            
            return data, indices
        
        # Use fori_loop to build the data and indices arrays
        data, indices = jax.lax.fori_loop(
            0,              # lower bound (inclusive)
            num_triangles,  # upper bound (exclusive)
            body_fun,       # body function
            (data, indices) # initial values
        )
        
        # Create BCOO sparse matrix
        K = sparse.BCOO(
            (data, indices),
            shape=(self.dim, self.dim),
            indices_sorted=True,
            unique_indices=True,
            )
        
        return K
    
    def get_boundary_mass_matrix(self):
        """
        Efficient boundary mass matrix assembly using precomputed edge contributions
        for uniform meshes. Optimized with jax.lax.fori_loop.
        """
        # Local boundary mass matrix pattern (scaled by edge length)
        local_boundary_mass = jnp.array([[2.0, 1.0],
                                         [1.0, 2.0]]) * (self.mesh.h / 6.0)
    
        # Get boundary edges from mesh
        boundary_edges = self.mesh.boundary_edges
        num_edges = boundary_edges.shape[0]
    
        # Preallocate data and indices for sparse matrix
        data = jnp.tile(local_boundary_mass.flatten(), num_edges)
        indices_shape = (num_edges * 4, 2)  # 4 entries per edge (2x2)
        indices = jnp.zeros(indices_shape, dtype=jnp.int64)
    
        # Define edge processing function
        def body_fun(e, idx_array):
            nodes = boundary_edges[e]
            row_indices = jnp.repeat(nodes, 2)
            col_indices = jnp.tile(nodes, 2)
            
            current_indices = jnp.stack([row_indices, col_indices], axis=1)
            start_idx = e * 4
            
            return jax.lax.dynamic_update_slice(
                idx_array,
                current_indices,
                (start_idx, 0)
            )
    
        # Process all boundary edges
        indices = jax.lax.fori_loop(
            0, num_edges, body_fun, indices
        )
    
        # Create BCOO sparse matrix
        MB = sparse.BCOO(
            (data, indices),
            shape=(self.dim, self.dim),
            indices_sorted=True,
            unique_indices=True,
            )
        
        return MB
        
    def get_boundary_extension(self):
        """
        creates extension matrix R that maps boundary nodes to full space nodes
        """
        num_total = self.dim
        
        data = jnp.ones(self.mesh.num_boundary_edges)
        indices = jnp.stack((jnp.arange(self.mesh.num_boundary_edges), self.mesh.boundary_vertices), axis=1)
        
        R = sparse.BCOO(
            (data, indices),
            shape=(self.mesh.num_boundary_edges, num_total),
            indices_sorted=True,
            )
    
        # transpose of restriction is extension
        return R.T
    
    @partial(jax.jit, static_argnums=(0,3))
    def get_projection_coeffs(
            self,
            u_values_on_triangle_quadpoints: jnp.ndarray,
            grad_u_values_on_triangle_quadpoints: jnp.ndarray = None,
            inner_product: str = None,
            ):
        """
        calculates the coefficients of the projection onto the ansatz space
        with respect to the chosen inner product. the projection is
        characterized by

        < u, phi_i > = < projection(u), phi_i >

        for all basis function phi_i. the coefficients of projection(u)
        are given by

        coeffs = M^-1 [ <u, phi_i> ]_i=1,...

        where M is the mass matrix M = [ <phi_j, phi_i> ]_i,j=1,...

        here, we use the H1 inner product.

        Args:
            u_values_on_triangle_quadpoints: values of the function u on the mapped triangle quadrature points
            grad_u_values_on_triangle_quadpoints: values of \nabla u on the mapped triangle quadrature points

        Returns:
            coefficients of projection(u)
        """

        # checks
        assert u_values_on_triangle_quadpoints.size == self.mesh.num_triangles * self.num_quad_nodes
        if inner_product is None:
            inner_product = self.inner_product
        assert inner_product in ['H1', 'L2']
        if inner_product == 'H1':
            assert grad_u_values_on_triangle_quadpoints.size == self.mesh.num_triangles * self.num_quad_nodes * 2 # 2 because gradient
        else:
            assert grad_u_values_on_triangle_quadpoints is None
            grad_u_values_on_triangle_quadpoints = jnp.zeros((self.mesh.num_triangles, self.num_quad_nodes, 2))
        
        # reshape if necessary
        u_values_on_triangle_quadpoints = u_values_on_triangle_quadpoints.reshape((self.mesh.num_triangles, self.num_quad_nodes))
        if inner_product == 'H1':
            grad_u_values_on_triangle_quadpoints = grad_u_values_on_triangle_quadpoints.reshape((self.mesh.num_triangles, self.num_quad_nodes, 2))

        # build inner product vector
        # l2 part: multiply with gbf_i and integrate
        u_gbf = jnp.einsum('tq,btq->btq', u_values_on_triangle_quadpoints, self.gbf_quad)
        l2_integral = jnp.sum(self.quadrature_with_values_mapped(u_gbf), axis=1) # shape (b,), summed over the individual triangles

        # h1 part - only if necessary
        def get_h1_integral():
            # multiply with grad gbf_i
            grad_u_grad_gbf = jnp.einsum('tqn,btqn->btq', grad_u_values_on_triangle_quadpoints, self.grad_gbf_quad)
            # integrate on single triangles and sum over the individual triangles to get full integral
            return jnp.sum(self.quadrature_with_values_mapped(grad_u_grad_gbf), axis=1) # shape (b,)

        h1_integral = jax.lax.cond(
            inner_product == 'H1',
            get_h1_integral,
            lambda: jnp.zeros((self.dim,)),
        )

        inner_product_vector = (l2_integral + h1_integral).reshape((-1,))

        # solve linear system with jax.scipy.sparse.linalg.bicgstab
        mass_matrix = self.get_l2_mass_matrix()
        if inner_product == 'H1':
            mass_matrix = mass_matrix + self.get_l2_stiffness_matrix()
        coeffs, info = bicgstab(mass_matrix, inner_product_vector, tol=1e-14, atol=1e-14, maxiter=1000)

        # # Check if solution converged - this does not yet work since info is not yet implemented in bicgstab
        # if info != 0:
        #     jax.debug.print(f'{style.warning}BiCGSTAB did not converge{style.end}')

        return coeffs
    
    def eval_coeffs(
            self,
            coeffs: jnp.ndarray,
            points_in_unit_triangle: jnp.ndarray,
            ):
        """
        evaluate the function corresponding to the coefficient vector at the points self.mesh.map_local_points_to_mesh(points_in_unit_triangle)
        
        Args:
            coeffs: coefficients, shape (self.dim,)
            points_in_unit_triangle: points in unit_triangle, shape (num_points, 2)

        Returns:
            tuple (values, grad), shape (self.mesh.num_triangles, num_points) and (self.mesh.num_triangles, num_points, 2)
        """
        
        # special cases can be handled directly
        if jnp.allclose(points_in_unit_triangle, self.quad_nodes_unit_triangle):
            return jnp.einsum('b,btp->tp', coeffs, self.gbf_quad), jnp.einsum('b,btpn->tpn', coeffs, self.grad_gbf_quad)
        
        # if jnp.allclose(points_in_unit_triangle, self.fine_points_unit_triangle):
        #     return jnp.einsum('b,btp->tp', coeffs, self.gbf_fine), jnp.einsum('b,btpn->tpn', coeffs, self.grad_gbf_fine)
        
        # general case
        gbf, grad_gbf = self.get_global_basis_functions(points_in_unit_triangle)
        return jnp.einsum('b,btp->tp', coeffs, gbf), jnp.einsum('b,btpn->tpn', coeffs, grad_gbf)
    
    def eval_coeffs_quad(
            self,
            coeffs: jnp.ndarray
            ):
        """
        evaluate the function corresponding to the coefficient vector at the points self.mesh.map_local_points_to_mesh(self.quad_nodes_unit_triangle))
        
        Args:
            coeffs: coefficients, shape (self.dim,)

        Returns:
            tuple (values, grad), shape (self.mesh.num_triangles, self.num_quad_nodes) and (self.mesh.num_triangles, self.num_quad_nodes, 2)
        """
        
        return jnp.einsum('b,btp->tp', coeffs, self.gbf_quad), jnp.einsum('b,btpn->tpn', coeffs, self.grad_gbf_quad)

    def visualize_coefficient_vector(
            self,
            coeffs: jnp.ndarray,
            title: str = None,
            vmin: float = None,
            vmax: float = None,
            plot_3d: bool = False,
            savepath: str = None,
            colorbar_label: str = None,
            ):

        assert coeffs.size == self.dim
        
        # assemble u
        # if (not plot_3d) or (self.mesh.nx <= 3 and self.mesh.ny <= 3):
        #     mapped_points = self.mapped_fine_points.reshape((-1,2))
        #     gbf = self.gbf_fine
        # else:
        mapped_points = self.mapped_quad_nodes.reshape((-1,2))
        gbf = self.gbf_quad
        
        u = jnp.einsum('b,btp->tp', coeffs, gbf).reshape((-1,))
        # u has shape (num_triangles*num_points,), fitting to the reshaped mapped points
        
        x = mapped_points[:,0]
        y = mapped_points[:,1]
        
        fig = plt.figure()
        
        if plot_3d:
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_trisurf(x, y, u, cmap='Spectral', alpha=0.8, vmin=vmin, vmax=vmax)
            fig.colorbar(surf, shrink=0.5)
        
        else:
            # triang = tri.Triangulation(x, y)
            # ax = fig.add_subplot(111)
            # # tpc = ax.tripcolor(triang, u, cmap='Spectral', shading='gouraud')
            # tpc = ax.tricontourf(triang, u, cmap='Spectral', levels=100, vmin=vmin, vmax=vmax)
            # ax.set_aspect('equal')
            # ax.set_xlabel('$x$')
            # ax.set_ylabel('$y$')
            # fig.colorbar(tpc, ax=ax)
            
            # Create regular grid for pcolormesh
            # Define regular grid
            xi = np.linspace(x.min(), x.max(), 500)
            yi = np.linspace(y.min(), y.max(), 500)
            Xi, Yi = np.meshgrid(xi, yi) # if indexing='ij' is used, then Zi.T must be used for imshow
            # Interpolate data onto regular grid
            Zi = griddata((x, y), u, (Xi, Yi), method='linear')
            ax = fig.add_subplot(111)
            # tpc = ax.pcolormesh(Xi, Yi, Zi, cmap='Spectral', vmin=vmin, vmax=vmax)
            tpc = ax.imshow(Zi, cmap='viridis', vmin=vmin, vmax=vmax, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='equal')
            # ax.set_aspect('equal')
            ax.set_xlabel('$x$')
            ax.set_xlim(0., self.mesh.Lx)
            ax.set_ylabel('$y$')
            ax.set_ylim(0., self.mesh.Ly)
            # set x and y ticks manually to fix different tickspacing
            num_ticks = 3
            xticks = np.linspace(0., self.mesh.Lx, num_ticks)
            yticks = np.linspace(0., self.mesh.Ly, num_ticks)
            xticklabels = [f'{x:.1f}' for x in xticks]
            yticklabels = [f'{y:.1f}' for y in yticks]
            ax.set_xticks(xticks, labels=xticklabels)
            ax.set_yticks(yticks, labels=yticklabels)
            fig.colorbar(tpc, ax=ax, label=colorbar_label)
            
        if savepath is not None:
            fig.tight_layout()
            fig.savefig(savepath + '.pgf') # save as pgf
            fig.savefig(savepath + '.png') # save as png
            print(f'figure saved under savepath {savepath} (as pgf and png)')

        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        fig.show()

        return

if __name__ == '__main__':

    jax.config.update("jax_enable_x64", True)

    from helpers.other import mpl_settings
    # mpl_settings(fontsize=18)
    mpl_settings()

    # create ansatz space
    space = AnsatzSpace()
    
    MB = space.boundary_mass_matrix.todense()
    plot_matrix(MB, title='boundary mass matrix')
    # dprint(MB.shape)
    # dprint(space.dim)
    #
    # R = space.boundary_extension_matrix.todense()
    # plot_matrix(R, title='boundary extension matrix')
    # dprint(R.shape)
    
    # consistency test
    ones = jnp.ones((space.dim,))
    boundary_gradients = space.boundary_grad_matrix @ ones
    dprint(jnp.linalg.norm(boundary_gradients)) # should be zero
    boundary_integral = ones.T @ MB @ ones  # should equal perimeter of domain
    dprint(boundary_integral)
    dprint(2*space.mesh.Lx + 2*space.mesh.Ly)
    
    M = space.get_l2_mass_matrix_naive()
    dprint(jnp.linalg.norm(M - space.l2_mass_matrix.todense()))

    K = space.get_l2_stiffness_matrix_naive()
    dprint(jnp.linalg.norm(K - space.l2_stiffness_matrix.todense()))

    # # setup coefficient vector
    # coeffs = jnp.zeros((space.dim,))
    # coeffs = coeffs.at[4].set(0.8)
    # coeffs = coeffs.at[1].set(0.5)
    # coeffs = coeffs.at[2].set(1.0)
    #
    # # # test visualization
    # space.visualize_coefficient_vector(coeffs)
    #
    # # test get_projection_coeffs
    # # obtain values on triangles
    # u_values_on_triangle_quadpoints = jnp.einsum('b,btq->tq', coeffs, space.gbf_quad)
    # grad_u_values_on_triangle_quadpoints = jnp.einsum('b,btqn->tqn', coeffs, space.grad_gbf_quad)
    # projection_coeffs = space.get_projection_coeffs(u_values_on_triangle_quadpoints, grad_u_values_on_triangle_quadpoints).reshape((space.dim,))
    # dprint(coeffs)
    # dprint(projection_coeffs)
    # dprint(jnp.linalg.norm(coeffs - projection_coeffs))












    pass

