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

def legendre(
        n: int,
        tt: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    calculates the values of the legendre
    polynomials at the points specified in tt
    up to order n

    in other words, the function returns an array

    A = [[P_0(tt[0]), ..., P_0(tt[-1])],
          ...
         [P_n(tt[0]), ..., P_n(tt[-1])]]

    where A_kj = P_k(tt_j). we have A.shape = (n+1, tt.shape)

    the method returns an additional array
    containing the derivatives at these points

    B = [[P_0'(tt[0]), ..., P_0'(tt[-1])],
          ...
         [P_n'(tt[0]), ..., P_n'(tt[-1])]]

    where B_kj = P_k'(tt_j). we have B.shape = (n+1, tt.shape)

    the method uses the three term recursions

    (k+1) P_{k+1}(t) = (2k+1) t P_k(t) - k P_{k-1}(t)
    d/dt P_{k+1}(t) = (k+1) P_k(t) + t * d/dt P_k(t)

    with P_0(t) = 1 , P_1(t) = t, P_0'(t) = 0, P_1'(t) = 1

    :param n: maximum degree of legendre polynomials
    :param tt: points to evaluate polynomials at
    :return: arrays A and B containing the values and derivative values
    """

    values_init = jnp.zeros((n+1,) + tt.shape)
    dvalues_init = jnp.zeros((n+1,) + tt.shape) # values of derivative

    P0 = jnp.ones(tt.shape)
    P1 = tt

    # derivatives
    dP0 = jnp.zeros(tt.shape)
    dP1 = jnp.ones(tt.shape)

    values_init = values_init.at[0].set(P0)
    values_init = values_init.at[1].set(P1)
    dvalues_init = dvalues_init.at[0].set(dP0)
    dvalues_init = dvalues_init.at[1].set(dP1)

    ### for jax.lax
    def loop(k,tup):
        values, dvalues = tup

        # handle normal values
        Pkm1, Pk = values[k-1], values[k]
        Pkp1 = (2*k+1)/(k+1) * tt * Pk - k/(k+1) * Pkm1
        values = values.at[k+1].set(Pkp1)

        # handle derivative values
        dPk = dvalues[k]
        dPkp1 = (k+1)*Pk + tt * dPk
        dvalues = dvalues.at[k+1].set(dPkp1)

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

    scaled_values = jnp.einsum('N,N...->N...',scalings,values)
    scaled_dvalues = jnp.einsum('N,N...->N...',scalings,dvalues)

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
    
    pass

