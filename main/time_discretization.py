#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements time discretization methods suitable for
# automatic differentiation via jax
#
# currently, the following methods are implemented:
# - implicit euler (order of convergence 1)
# - implicit midpoint with linear interpolation of the
#   control (crank nicolson, order of convergence 2)
# - a custom discrete gradient method suitable for
#   passive systems
# - bdf4 (order of convergence 4)
# - the projection based method
#

# jax
import jax
import jax.numpy as jnp
from jax import jit, jacobian
import jax.lax

# custom imports
from helpers.newton import newton, newton_sparse, newton_lineax
from helpers.other import style
# for projection method
from scipy.special import roots_legendre
from helpers.energy_based_model import EnergyBasedModel
from helpers.legendre import scaled_legendre, scaled_legendre_on_boundaries
from helpers.gauss import gauss_quadrature_with_values, project_with_gauss

def projection_method(
        ebm: EnergyBasedModel,
        tt: jnp.ndarray,
        z0: jnp.ndarray,
        control: callable,
        degree: int,
        num_quad_nodes: int = None,
        num_proj_nodes: int = None,
        debug: bool = False,
        g_manufactured_solution: callable = None,
    ) -> jnp.array:
    """
    computes an approximate solution of
    
    [ nabla_1 ham ]                [    dt z1    ]
    [    dt z2    ]  =  ( J - R )  [ nabla_2 ham ] + B u + g,    z(0) = z_0
    [      0      ]                [     z3      ]
    
    on given timesteps in the time horizon [0,T].

    the method finds an approximate solution ztau such that
    the function ztau satisfies the projection equation and
    is a piecewise polynomial with a specified maximal degree.
    
    note that J and R are not necessarily linear operators.

    :param ebm: energy based model
    :param tt: array of timepoints to be used
    :param z0: initial condition for (z1, z2, z3) that is consistent with the algebraic constraints
    :param control: control function u, callable
    :param degree: degree of piecewise polynomial approximation
    :param num_quad_nodes: number of quadrature nodes used for the quadrature rule for (J-R) eta + B u (optional, default = degree)
    :param num_proj_nodes: number of quadrature nodes used in the projection of eta (optional, default = degree)
    :param debug: debug flag (default False)
    :param g_manufactured_solution: optional function for using a manufactured solution
    :return: solution as a dictionary containing:
                - value of solution at time points [t_0, t_1, ... ]
                # - value of solution at 25 points in each interval [t_0, t_1], [t_1, t_2], ...
                # - value of solution at gauss points in each interval [t_0, t_1], [t_1, t_2], ...
                - array of coefficients for solution of shape (N, (n+1), D)
                    the coefficient for the k-th time interval [t_k, t_{k+1}] is
                    stored in coefflist[k,:,:].
                    evaluation of these coefficients is possible using, e.g.,
                        jnp.einsum('MD,Mt->tD',coefflist[k,:,:],scaled_legendre(n,wanted_points)[0])
                - the degree
                - the number of quadrature nodes for the quadrature rules
                - the number of quadrature nodes for the projection
    """

    # eta^inv is not needed if eta is linear or J and R are state-independent.
    J = ebm.J_vmap
    R = ebm.R_vmap
    B = ebm.B_vmap
    nabla_1_ham = ebm.nabla_1_ham_vmap
    nabla_2_ham = ebm.nabla_2_ham_vmap
    g = g_manufactured_solution
    
    d1, d2, d3 = ebm.dims
    M = tt.shape[0]-1 # number of timepoints
    
    if num_quad_nodes is None:
        num_quad_nodes = degree + 1 # plus one for exact integration of algebraic constraints

    if num_proj_nodes is None:
        num_proj_nodes = degree

    if g is None:
        @jax.profiler.annotate_function
        def g(t):
            return jnp.zeros((t.shape[0],d1+d2+d3))

    # setup variable gauss quadrature for (J-R) term -> nqn nodes (nqn = `num quadrature nodes`)
    nqn_gauss_points, nqn_gauss_weights = roots_legendre(num_quad_nodes)
    nqn_gauss_points, nqn_gauss_weights = jnp.array(nqn_gauss_points), jnp.array(nqn_gauss_weights)
    # get legendre values on nqn_gauss_points (for quadratures Q_i)
    psi_at_nqn_gauss_points, psi_prime_at_nqn_gauss_points = scaled_legendre(degree, nqn_gauss_points) # n in first argument is correct here. shape (n+1, nqn)
    phi_at_nqn_gauss_points = psi_at_nqn_gauss_points[:-1,:] # phi[i,j] = phi_i(tj), one degree less since we only test with derivatives for z1 and z2. shape (n, nqn)
    
    # setup gauss quadrature for projection of eta -> proj_nodes nodes (npn = `num projection nodes`)
    npn_gauss_points, npn_gauss_weights = roots_legendre(num_proj_nodes)
    npn_gauss_points, npn_gauss_weights = jnp.array(npn_gauss_points), jnp.array(npn_gauss_weights)
    # get legendre values on proj_gauss_points (for scalar product in projection of nabla ham)
    psi_at_npn_gauss_points, psi_prime_at_npn_gauss_points = scaled_legendre(degree, npn_gauss_points) # n in first argument is correct here. shape (n+1, npn)
    phi_at_npn_gauss_points = psi_at_npn_gauss_points[:-1,:] # phi[i,j] = phi_i(tj), one degree less since we only test with derivatives for z1 and z2. shape (n, npn)
    
    # get scaled legendre values on -1 and 1
    minus1, plus1, dt_minus1, dt_plus1 = scaled_legendre_on_boundaries(degree)
    
    # function defining the discretization scheme - solutions of the scheme satisfy F=0
    @jax.profiler.annotate_function
    def F(
        coeffs123: jnp.ndarray,
        left_boundary_value_12: jnp.ndarray,
        k: int,
        ) -> jnp.ndarray:
        """
        this function calculates

        F(coeffs1,coeffs2,coeffs3))

        where zeros of F correspond to coefficients of solutions of the
        scheme for the k-th time interval.

        note: most shifts from [-1,1] to [tk,tkp1] are missing here,
        since they do not change the result of the computation. only
        in the calculation of the derivative partial_t z_h, a scaling
        factor is incorporated.

        :param coeffs123: coefficient vectors c^i_0,...,c^i_n for i=1,2,3, corresponding to z1, z2, z3
        :param left_boundary_value_12: value jnp.hstack(z1(tk), z2(tk)) -- no boundary value for z3 to ensure well posedness of the system
        :param k: timestep number
        :return: F(coeffs)
        """
        
        # extract coefficients for z1, z2, z3
        coeffs123 = coeffs123.reshape((degree+1,d1+d2+d3))
        coeffs1 = coeffs123[:,:d1]
        coeffs2 = coeffs123[:,d1:d1+d2]
        coeffs3 = coeffs123[:,d1+d2:]
        # jax.debug.print('coeffs1.shape = {x}', x=coeffs1.shape)
        # jax.debug.print('coeffs2.shape = {x}', x=coeffs2.shape)
        # jax.debug.print('coeffs3.shape = {x}', x=coeffs3.shape)
        # coeffs1[k,d] is the d-th entry of the coefficient belong to p_k for z1, and vice versa for z2 and z3

        tk, tkp1 = tt[k], tt[k+1]

        # find dt_z1 at nqn_gauss_points and dt_z2 at n_gauss_points, d = d1 or d = d2
        dt_z1_nqn = jnp.einsum('Nd,Nt->td', coeffs1, psi_prime_at_nqn_gauss_points) * 2/(tkp1-tk) # factor for compensating for chain rule in shift: f on [-1,1] -> sqrt(2/(b-a)) f((2t - a - b)/(b-a)) on [a,b]
        dt_z2_npn = jnp.einsum('Nd,Nt->td', coeffs2, psi_prime_at_npn_gauss_points) * 2/(tkp1-tk) # factor for compensating for chain rule in shift: f on [-1,1] -> sqrt(2/(b-a)) f((2t - a - b)/(b-a)) on [a,b]

        # find z1, z2, z3 at nqn_gauss_points and z1, z2 at proj_gauss_points
        z3_nqn = jnp.einsum('Nd,Nt->td', coeffs3, psi_at_nqn_gauss_points) # d = d3
        z1_npn = jnp.einsum('Nd,Nt->td', coeffs1, psi_at_npn_gauss_points) # d = d1
        z2_npn = jnp.einsum('Nd,Nt->td', coeffs2, psi_at_npn_gauss_points) # d = d2
        
        # find nabla_1_ham at npn_gauss_points # npn since this is essentially also a projection of \nabla_1 H
        nabla_1_ham_npn = nabla_1_ham(z1_npn, z2_npn) # shape (t, d1)
        
        # find projection of nabla_2_ham at nqn_gauss_points
        proj_nabla_2_ham_nqn = project_with_gauss(npn_gauss_weights, phi_at_npn_gauss_points, nabla_2_ham(z1_npn, z2_npn), evaluate_with=phi_at_nqn_gauss_points)

        # find values of (J - R) [ P_{n-1}[eta(z)] ] at nqn_gauss_points
        JmR_nqn = J(dt_z1_nqn, proj_nabla_2_ham_nqn, z3_nqn) - R(dt_z1_nqn, proj_nabla_2_ham_nqn, z3_nqn)

        # find values of B u at nqn_gauss_points
        shifted_nqn_gauss_points = (tkp1-tk)/2 * nqn_gauss_points + (tk+tkp1)/2
        Bu_nqn = B(control(shifted_nqn_gauss_points))

        # for manufactured solution
        g_nqn = g(shifted_nqn_gauss_points)

        # piece them together
        JmR_nqn = JmR_nqn + Bu_nqn + g_nqn # shape (t,D)

        # left hand side integrals
        lhs_stack_12 = jnp.hstack((nabla_1_ham_npn, dt_z2_npn)) # shapes (t,d1), (t,d2) -> shape (tpn, d1+d2)
        lhs_prod_12 = jnp.einsum('td,nt->tnd', lhs_stack_12, phi_at_npn_gauss_points) # shape (tpn, n, d1+d2)
        lhs_integrals_12 = gauss_quadrature_with_values(npn_gauss_weights, lhs_prod_12, interval=(tk,tkp1)) # shape (n,d1+d2)

        # right hand side integrals
        JmR_test_12 = jnp.einsum('td,nt->tnd', JmR_nqn[:,:d1+d2], phi_at_nqn_gauss_points) # shape (tqn, n, d1+d2)
        JmR_test_3 = jnp.einsum('td,Nt->tNd', JmR_nqn[:,d1+d2:d1+d2+d3], psi_at_nqn_gauss_points) # shape (tqn, n+1, d3)
        
        rhs_integrals_12 = gauss_quadrature_with_values(nqn_gauss_weights, JmR_test_12, interval=(tk,tkp1)) # shape (n,d1+d2)
        rhs_integrals_3 = gauss_quadrature_with_values(nqn_gauss_weights, JmR_test_3, interval=(tk,tkp1)) # shape (n+1,d3)

        # ..._integrals contains the integral
        #   ..._integrals[n,d] = int < ... , phi_{nd} >
        # where phi_{nd} is phi_n in the d-th row

        # assemble to dynamics array
        dynamics_12 = lhs_integrals_12 - rhs_integrals_12 # shape (n,d1+d2)
        dynamics_3 = rhs_integrals_3 # shape (n,d1+d2) -- since lhs_integrals_3 == 0
        
        # enforce continuity
        left_boundary_value_12_with_coeffs = jnp.einsum('Nd,N->d', coeffs123[:,:d1+d2], minus1) # only continuity for z1 and z2, since z3 is tested with one degree higher than the rest
        continuity_12 = left_boundary_value_12 - left_boundary_value_12_with_coeffs
        
        # stack and reshape, this is zero if the coefficients are correct
        ret = jnp.hstack((dynamics_12.reshape((-1,)), dynamics_3.reshape((-1,)), continuity_12.reshape((-1,))))
        return ret
    
    coefflist = jnp.zeros((M,degree+1,d1+d2+d3))
    # coefflist = coefflist.at[0,:,:].set(jnp.ones((n+1,D)).at[1:,:].set(0)) # constant function as initial guess for first time interval

    # setup rootfinder for F
    rootfinder = newton_lineax(f=F, max_iter=10, tol=1e-17, debug=debug) # lineax version
    
    # prepare values of z at timepoints tt
    zz = jnp.zeros((M+1,d1+d2+d3)).at[0,:].set(z0)
    dt_zz_12 = jnp.zeros((M,d1+d2)) # derivatives of z1 and z2 and timepoints tt[1:]

    # # for 25 values in each interval
    # nt_superfine = 25
    # t_superfine = jnp.linspace(-1,1,nt_superfine) # in one interval
    # zz_superfine = jnp.zeros((M,nt_superfine,d1+d2+d3)) # will store z values for all intervals
    # tt_superfine = jnp.zeros((M,nt_superfine)) # will store t values for all intervals
    # v_superfine, v_dsuperfine = scaled_legendre(degree,t_superfine)

    # # for values at gauss points
    # # t_gauss = gauss_points
    # zz_quad_nodes = jnp.zeros((M,num_quad_nodes,d1+d2+d3))
    # tt_quad_nodes = jnp.zeros((M,num_quad_nodes))

    # initialize left_boundary_value
    left_boundary_value_12 = z0[:d1+d2]

    # initialize coefficient guess
    coeff_guess = jnp.ones((degree+1,d1+d2+d3)).at[1:,:].set(0).reshape((-1,))
    # coeff_guess = jnp.vstack( ( jnp.sqrt(2)*z0.reshape(1,D), jnp.zeros((n,D)) ) ).reshape((-1,)) # constant polynomial with initial condition as first guess

    @jax.profiler.annotate_function
    def body_fun(k, tup):
        # coefflist, coeff_guess, left_boundary_value, zz = tup
        # coefflist, coeff_guess, left_boundary_value_12, zz, dt_zz_12, zz_superfine, tt_superfine, zz_quad_nodes, tt_quad_nodes = tup
        coefflist, coeff_guess, left_boundary_value_12, zz, dt_zz_12 = tup
        
        if debug: jax.debug.print(f'{style.info}timestep number = {{k}}{style.end}', k=k+1)

        # find root with newton
        root = rootfinder(coeff_guess, left_boundary_value_12=left_boundary_value_12, k=k)
        coeffs123 = root.reshape((degree+1,d1+d2+d3))
        
        # # print jacobian at coeff_guess and root
        # from helpers.other import plot_matrix
        # plot_matrix(DF(coeff_guess, left_boundary_value=left_boundary_value, k=k), title=f'jacobian at coeff_guess, timestep {k}')
        # plot_matrix(DF(root, left_boundary_value=left_boundary_value, k=k), title=f'jacobian at root, timestep {k}')

        # for values at tt
        zz_kp1 = jnp.einsum('ND,N->D', coeffs123, plus1)
        dt_zz_12_kp1 = jnp.einsum('ND,N->D', coeffs123[:,:d1+d2], dt_plus1) * 2/(tt[k+1] - tt[k])
        zz = zz.at[k+1,:].set(zz_kp1)
        dt_zz_12 = dt_zz_12.at[k+1,:].set(dt_zz_12_kp1)

        # update coefflist and coeff_guess
        coefflist = coefflist.at[k,:,:].set(coeffs123)
        coeff_guess = coeffs123.reshape((-1,))

        # # # for superfine points
        # tk, tkp1 = tt[k], tt[k+1]
        # zz_superfine_in_tk_tkp1 = jnp.einsum('ND,Nt->tD', coeffs, v_superfine)
        # zz_superfine = zz_superfine.at[k,:,:].set(zz_superfine_in_tk_tkp1)
        # tt_superfine = tt_superfine.at[k,:].set((tkp1-tk)/2 * t_superfine + (tk+tkp1)/2)

        # # for gauss points
        # zz_quad_nodes_in_tk_tkp1 = jnp.einsum('ND,Nt->tD', coeffs, psi_at_nqn_gauss_points)
        # zz_quad_nodes = zz_quad_nodes.at[k,:,:].set(zz_quad_nodes_in_tk_tkp1)
        # tt_quad_nodes = tt_quad_nodes.at[k,:].set((tkp1-tk)/2 * nqn_gauss_points + (tk+tkp1)/2)

        # update left boundary value for next iteration
        left_boundary_value_12 = zz_kp1[:d1+d2]
        
        return coefflist, coeff_guess, left_boundary_value_12, zz, dt_zz_12 #, zz_superfine, tt_superfine, zz_quad_nodes, tt_quad_nodes

        # print(f'iteration k={k} done')
    init_val = (coefflist, coeff_guess, left_boundary_value_12, zz, dt_zz_12) #, zz_superfine, tt_superfine, zz_quad_nodes, tt_quad_nodes)
    # coefflist, _, _, zz = jax.lax.fori_loop(0, M, body_fun, init_val)
    # coefflist, _, _, zz, zz_superfine, tt_superfine, zz_quad_nodes, tt_quad_nodes = jax.lax.fori_loop(0, M, body_fun, init_val)
    coefflist, _, _, zz, dt_zz_12 = jax.lax.fori_loop(0, M, body_fun, init_val)
    
    # # plot jacobian at zz[1,:]
    # from helpers.other import plot_matrix
    # DF = jax.jacobian(F, argnums=0)
    # plot_matrix(DF(coefflist[2,:,:].reshape((-1,)), left_boundary_value=zz[2,:], k=3), title=f'jacobian cahn hilliard')
    #
    # # reshape tt_superfine, zz_superfine
    # tt_superfine = tt_superfine.reshape((M*nt_superfine,))
    # zz_superfine = zz_superfine.reshape((M*nt_superfine, d1+d2+d3))
    #
    # # reshape tt_quad_nodes, zz_quad_nodes
    # tt_quad_nodes = tt_quad_nodes.reshape((M*num_quad_nodes,))
    # zz_quad_nodes = zz_quad_nodes.reshape((M*num_quad_nodes, d1+d2+d3))
    
    return {
        'boundaries': (tt, zz, dt_zz_12), # values of solution at time interval boundaries and derivate values on right boundaries (left side derivative)
        # 'superfine': (tt_superfine, zz_superfine, dt_zz_superfine), # values at "superfine" sample points in between boundaries
        # 'quad_nodes': (tt_quad_nodes, zz_quad_nodes, dt_zz_quad_nodes), # values at quadrature nodes
        'coefflist': coefflist, # shape (M,n+1,D)
        # #
        # 'ph_sys': ph_sys,
        'degree': degree,
        'num_quad_nodes': num_quad_nodes,
        'num_proj_nodes': num_proj_nodes,
        # #
        # 'control': u,
        # 'g': g,
        }


