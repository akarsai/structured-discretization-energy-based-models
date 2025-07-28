#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements helper functions for triangles used in the
# space discretization.
#
#

import jax
import jax.numpy as jnp

@jax.jit
def map_points_to_triangle(
        triangle_vertices: jnp.array,
        points: jnp.array,
        ):
    # see also
    # https://math.stackexchange.com/questions/525732/gaussian-integration-on-triangles

    if points.ndim == 1:
        points = points[None, :]

    v1, v2, v3 = triangle_vertices
    x, y = points[:, 0], points[:, 1]

    # compute mapped points
    mapped_points = (
            v1
            + jnp.einsum('n,m->nm', x, v2 - v1)
            + jnp.einsum('n,m->nm', y, v3 - v1)
            )

    return mapped_points

@jax.jit
def get_triangle_jacobian_and_area(
        triangle_vertices: jnp.array,
        ):

    v1, v2, v3 = triangle_vertices

    # compute jacobian matrix and area of triangle
    jacobian = jnp.column_stack([v2 - v1, v3 - v1])
    area = 1/2 * jnp.abs(jnp.linalg.det(jacobian))

    return jacobian, area

def get_triangle_quadrature_points_and_weights():
    # taken from
    # https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
    # STRANG5, order 6, degree of precision 4
    #
    # the given weights are normalized to 1 (they sum to 1).
    # to account for the area of the unit triangle, they need to be scaled by 1/2.
    #
    # see, e.g., https://mathsfromnothing.au/triangle-quadrature-rules/?i=1
    #
    # A note about quadrature weights, most references normalize the weights so their
    # sum equals one rather than keeping them at the values found in the derivation
    # so the weights reported above are half what most references will give. These
    # calculated weights can be used directly to integrate a function whereas the
    # weights from most references including the table below will need to be multiplied
    # by a half (area of the basis triangle).
    

    points = [
        [0.816847572980459, 0.091576213509771],
        [0.091576213509771, 0.816847572980459],
        [0.091576213509771, 0.091576213509771],
        [0.108103018168070, 0.445948490915965],
        [0.445948490915965, 0.108103018168070],
        [0.445948490915965, 0.445948490915965]
        ]

    weights = [
        0.109951743655322,
        0.109951743655322,
        0.109951743655322,
        0.223381589678011,
        0.223381589678011,
        0.223381589678011
        ]

    return jnp.array(points), 1/2 * jnp.array(weights)