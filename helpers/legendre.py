#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements the legendere polynomials, both in
# unscaled (orthogonal) and unscaled (orthonormal) versions.
# the methods are used throughout this project.
#
#


import jax.numpy as jnp
import jax
# jax.config.update("jax_enable_x64", True)

from scipy.special import roots_legendre

def legendre(n,tt):
    """
    calculates the values of the legendre
    polynomials at the points specified in tt
    up to order n

    in other words, the function returns an array

    A = [[P_0(tt[0]), ..., P_0(tt[-1])],
          ...
         [P_n(tt[0]), ..., P_n(tt[-1])]]

    where A_kj = P_k(tt_j)

    the method returns an additional array
    containing the derivatives at these points

    B = [[P_0'(tt[0]), ..., P_0'(tt[-1])],
          ...
         [P_n'(tt[0]), ..., P_n'(tt[-1])]]

    where B_kj = P_k'(tt_j)

    the method uses the three term recursions

    (k+1) P_{k+1}(t) = (2k+1) t P_k(t) - k P_{k-1}(t)
    d/dt P_{k+1}(t) = (k+1) P_k(t) + t * d/dt P_k(t)

    with P_0(t) = 1 , P_1(t) = t, P_0'(t) = 0, P_1'(t) = 1

    :param n: maximum degree of legendre polynomials
    :param tt: points to evaluate polynomials at
    :return: arrays A and B containing the values and derivative values
    """

    nt = tt.shape[0]

    values_init = jnp.zeros((n+1,nt))
    dvalues_init = jnp.zeros((n+1,nt)) # values of derivative

    P0 = jnp.ones(tt.shape)
    P1 = tt

    # derivatives
    dP0 = jnp.zeros(tt.shape)
    dP1 = jnp.ones(tt.shape)

    values_init = values_init.at[0,:].set(P0)
    values_init = values_init.at[1,:].set(P1)
    dvalues_init = dvalues_init.at[0,:].set(dP0)
    dvalues_init = dvalues_init.at[1,:].set(dP1)

    ### normal python
    # for k in range(1,n):
    #
    #     # handle normal values
    #     Pkm1, Pk = values[k-1,:], values[k,:]
    #     Pkp1 = (2*k+1)/(k+1) * tt * Pk - k/(k+1) * Pkm1
    #     values = values.at[k+1,:].set(Pkp1)
    #
    #     # handle derivative values
    #     dPk = dvalues[k,:]
    #     dPkp1 = (k+1)*Pk + tt * dPk
    #     dvalues = dvalues.at[k+1,:].set(dPkp1)

    ### for jax.lax
    def loop(k,tup):
        values, dvalues = tup

        # handle normal values
        Pkm1, Pk = values[k-1,:], values[k,:]
        Pkp1 = (2*k+1)/(k+1) * tt * Pk - k/(k+1) * Pkm1
        values = values.at[k+1,:].set(Pkp1)

        # handle derivative values
        dPk = dvalues[k,:]
        dPkp1 = (k+1)*Pk + tt * dPk
        dvalues = dvalues.at[k+1,:].set(dPkp1)

        return values, dvalues

    return jax.lax.fori_loop(
        lower=1,
        upper=n,
        body_fun=loop,
        init_val=(values_init, dvalues_init),
        )

def scaled_legendre(n,tt):
    """
    similar to the method `legendre`, but returns
    the values of scaled legendre polynomials

    p_k = sqrt((2*k + 1) / 2) * P_k

    :param n: maximum degree of legendre polynomials
    :param tt: points to evaluate polynomials at
    :return: arrays A and B containing the values and derivative values
    """

    values, dvalues = legendre(n,tt)

    scalings = jnp.sqrt((2*jnp.arange(n+1)+1)/2)

    scaled_values = jnp.einsum('N,Nt->Nt',scalings,values)
    scaled_dvalues = jnp.einsum('N,Nt->Nt',scalings,dvalues)

    ### old code, equivalent but slower
    # for j in range(n+1):
    #     values = values.at[j,:].set(scalings[j]*values[j,:])
    #     dvalues = dvalues.at[j,:].set(scalings[j]*dvalues[j,:])

    return scaled_values, scaled_dvalues

def scaled_legendre_on_boundaries(
        n: int,
        ):
    """
    this function caches the values of the scaled legendre
    polynomials
        p_0, ..., p_n     (that are defined on [-1,1])
    at the values -1, 1, and their derivatives at these points.
    this then allows for a quick computation of a function
        z(t) = sum_{k=0}^{n} c_k p_k(t)
    at the values t = -1, 1, and its derivative at these points.
    the legendre polynomials are scaled here so that they
    form a basis that is L^2 orthonormal
    :param n: maximum number of order
    :return: minus1, plus1, dt_minus1, dt_plus1
        - minus1: values at t = -1
        - plus1: values at t = 1
        - dt_minus1: derivative values at t = -1
        - dt_plus1: derivative values at t = 1
    """
    # scalings are always (2*j + 1) / 2
    scalings = jnp.sqrt((2*jnp.arange(n+1)+1)/2)
    
    # values at -1 and +1
    # for the unscaled polynomials, we have q(-1) = +- 1 and q(1) = 1
    alternating = (1-(jnp.arange(n+1)%2)*2)
    minus1 = scalings*alternating
    plus1 = scalings*jnp.ones((n+1,))
    
    # derivative values at boundaries
    # For Legendre polynomials: P_n'(-1) = (-1)^(n+1) * n*(n+1)/2
    #                          P_n'(1) = n*(n+1)/2
    # Special case: P_0'(x) = 0 everywhere
    k = jnp.arange(n+1)
    
    # Derivative of unscaled Legendre polynomials at boundaries
    # P_0'(x) = 0, so we set the first element to 0
    dt_unscaled_minus1 = jnp.where(
        k == 0,
        0.0,
        (-1)**(k+1) * k * (k+1) / 2
    )
    dt_unscaled_plus1 = jnp.where(
        k == 0,
        0.0,
        k * (k+1) / 2
    )
    
    # Apply scaling to derivatives
    dt_minus1 = scalings * dt_unscaled_minus1
    dt_plus1 = scalings * dt_unscaled_plus1
    
    return minus1, plus1, dt_minus1, dt_plus1

def cache_legendre_values(
        n: int,
        gauss_points: jnp.ndarray | None = None
        ):
    """
    this function caches the values of the scaled legendre
    polynomials

        p_0, ..., p_n     (that are defined on [-1,1])

    at the values -1, 1, and the gauss interpolation points

        x_i, i = 1,..., n+1

    which are the roots of the n+1 th legendre polynomial.
    this then allows for a quick computation of a function

        z(t) = sum_{k=0}^{n} c_k p_k(t)

    at the values t = -1, 1, ti.

    the legendre polynomials are scaled here so that they
    form a basis that is L^2 orthonormal

    :param n: maximum number of order
    :return: 3 tuple consisting of:
        array of values at -1,
        array of values at +1,
        matrix of values at ti     here, m[j,i] = p_j(ti)
    """

    minus1, plus1, _, _ = scaled_legendre_on_boundaries(n)

    # values at roots
    if gauss_points is None:
        gauss_points, _ = roots_legendre(n+1)

    rootvalues, drootvalues = scaled_legendre(n,gauss_points)
    # rootvalues[j,xi] = p_j(xi)
    # drootvalues[j,xi] = p_j'(xi)

    return minus1, plus1, rootvalues, drootvalues

def shift_to_interval(
        fvalues: jnp.ndarray,
        interval: tuple,
        ) -> jnp.ndarray:
    """
    this function scales the function values provided in
    fvalues to the corresponding values on [a,b] in such
    a way that if fvalues comes from a function f that is
    orthonormal on [-1,1] w.r.t. the L2 inner product,
    then the computed function values belong to the
    function that is scaled and shifted to be orthonormal
    w.r.t. the L2 inner product on [a,b].

    the transformed function reads as

    sqrt(2/(b-a)) f( (2t-a-b) / (b-a) )

    the inner transformation in f is not relevant here.
    only the outer scaling is important.

    :param fvalues: values of f (f orthonormal on [-1,1])
    :param interval: interval to transform to
    :return: transformed fvalues according to the formula
    """

    a,b = interval

    return jnp.sqrt(2/(b-a)) * fvalues


if __name__ == '__main__':
    
    from helpers.other import dprint
    jax.config.update('jax_enable_x64', True)

    #### test the transformation matrix
    # a, b = 0, 1
    #
    # # polynomial degree
    # n = 5
    # _, plus1, rootvalues, drootvalues = cache_legendre_values(n)
    # rootvalues_shifted = shift_to_interval(rootvalues, (a,b))
    # drootvalues_shifted = shift_to_interval(drootvalues, (a,b))
    #
    # # get roots of n+1 -th legendre polynomial on [-1,1]
    # roots, _ = roots_legendre(n+1)
    #
    # # shift to interval [a,b]
    # roots_shifted = (b-a)/2 * roots + (a+b)/2
    #
    # # j-th lagrange polynomial with nodes `points`
    # def get_lagrange(points,j):
    #
    #     tj = points[j]
    #
    #     def L_j(t):
    #         prod = 1
    #         for tk in jnp.delete(points,j):
    #             prod *= (t-tk)/(tj-tk)
    #         return prod
    #
    #     return L_j
    #
    # i = 0 # which polynomial?
    # L_normal = get_lagrange(roots, i)
    # L_shifted = get_lagrange(roots_shifted, i)
    #
    # nt = 100
    #
    # # for [-1,1]
    # tt_normal = jnp.linspace(-1,1,nt)
    # T_normal = jnp.linalg.inv(rootvalues.T)
    # values_lagrange = L_normal(tt_normal)
    # # get basis coefficients in legendre basis
    # legendre_coeffs_of_lagrange_poly = T_normal[:,i]
    # # assemble legendre polynomials with coefficients
    # s_leg, s_dleg = scaled_legendre(n,tt_normal)
    # values_legendre_normal = jnp.einsum('M,Mt->t',legendre_coeffs_of_lagrange_poly,s_leg)
    #
    # # for [a,b]
    # tt_shifted = (b-a)/2 * tt_normal + (a+b)/2
    # T_shifted = jnp.linalg.inv(rootvalues_shifted.T)
    # # get basis coefficients in legendre basis
    # legendre_coeffs_of_lagrange_poly_shifted = T_shifted[:,i]
    # # assemble legendre polynomials with coefficients
    # s_leg_shifted = shift_to_interval(s_leg, (a,b))
    # values_legendre_shifted = jnp.einsum('M,Mt->t',legendre_coeffs_of_lagrange_poly_shifted,s_leg_shifted)
    #
    # # check derivative
    # poly = scipy_legendre(n)  # coefficients of n^th degree Legendre polynomial
    # polyd = poly.deriv() # coefficients of derivative of n^th degree Legendre Polynomial
    # eval_d_np = np.polyval(polyd,tt_normal) # evaluate derivative at desired coordinates(s)

    # import matplotlib.pyplot as plt

    # plt.plot(tt_normal, eval_d_np, label='derivative of legendre_n (numpy)')
    # eval_d_my = legendre(n,tt_normal)[1][-1,:]
    # plt.plot(tt_normal, eval_d_my, label='derivative of legendre_n (my method)')
    # plt.legend()
    # plt.title('derivative of n-th legendre polynomial')
    # plt.show()
    #
    # print(f'||eval_d_np-eval_d_my|| = {np.linalg.norm(eval_d_np-eval_d_my)}')



    ### plot
    import matplotlib.pyplot as plt

    # plt.plot(tt_normal,values_lagrange,
    #          label='lagrange polynomial')
    # plt.plot(tt_normal,values_legendre_normal,
    #          label='weighted sum of scaled legendre polynomials')
    # plt.legend()
    # plt.title('on [-1,1]')
    # plt.show()
    #
    # plt.plot(tt_shifted,values_lagrange,
    #          label='lagrange polynomial')
    # plt.plot(tt_shifted,values_legendre_shifted,
    #          label='weighted sum of scaled legendre polynomials')
    # plt.legend()
    # plt.title('shifted to [a,b]')
    # plt.show()




    # zh_fine = jnp.einsum('M,Mt->t',legendre_coeffs_of_lagrange_poly,s_leg)
    # zh = jnp.einsum('M,Mt->t',legendre_coeffs_of_lagrange_poly,rootvalues)
    # zh_dot = jnp.einsum('M,Mt->t',legendre_coeffs_of_lagrange_poly,drootvalues)
    # # E_zh_dot_values = jnp.einsum('xD,tD->tx',E,zh_dot)
    #
    # # find projection of eta at gauss points
    # zh_coeffs = jnp.einsum('Mt,t->M',T_normal,zh) # find zh at gauss points (lagrange coeffs) and transform (to get legendre coeffs)
    # zh_proj_fine = jnp.einsum('m,mt->t',zh_coeffs[:-1],s_leg[:-1,:]) # calculate eta projection at fine timesteps
    # zh_proj = jnp.einsum('m,mt->t',zh_coeffs[:-1],rootvalues[:-1,:]) # calculate eta projection at gauss points


    # plt.plot(tt_normal,values_lagrange,
    #          label='lagrange polynomial')
    # plt.plot(tt_normal,zh_fine,
    #          label='weighted sum of scaled legendre polynomials')
    # plt.plot(tt_normal,zh_proj_fine,
    #          label='projection of weighted sum of scaled legendre polynomials')
    # # plt.plot(roots,zh,
    # #          label='weighted sum of scaled legendre polynomials')
    # # plt.plot(roots,zh_proj,
    # #          label='projection of weighted sum of scaled legendre polynomials')
    # plt.legend()
    # plt.title('on [a,b]')
    # plt.show()

    # ### test orthonormality
    # from scipy.special import roots_legendre
    # from gauss import gauss_quadrature_with_values
    #
    # interval = (-1,1)
    #
    # num_gauss_points = 10
    # gauss_points, gauss_weights = roots_legendre(num_gauss_points)
    # gauss_points = jnp.array(gauss_points)
    # gauss_weights = jnp.array(gauss_weights)
    #
    # pk_values, _ = scaled_legendre(5,gauss_points)
    #
    # print(pk_values.shape)
    #
    # pk_pk_values = jnp.einsum('mt,mt->tm',pk_values, pk_values)
    # L2_norms_pk = gauss_quadrature_with_values(gauss_weights, pk_pk_values, interval=(-1,1))
    # print(f'L2_norms_pk = {L2_norms_pk}')
    
    
    # compare scaled_legendre_on_boundaries and scaled_legendre
    n = 10
    tt = jnp.linspace(-1,1,2)
    values, dt_values = scaled_legendre(n, tt)
    minus1, plus1, dt_minus1, dt_plus1 = scaled_legendre_on_boundaries(n)
    
    error_values = jnp.linalg.norm(values - jnp.stack((minus1, plus1), axis=1))
    error_dt_values = jnp.linalg.norm(dt_values - jnp.stack((dt_minus1, dt_plus1), axis=1))
    
    dprint(error_values)
    dprint(error_dt_values)

