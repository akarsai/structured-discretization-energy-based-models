#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements helper functions to calculate errors used
# in convergence analysis and in visualization
#
#

import jax.numpy as jnp
import jax


from helpers.gauss import gauss_quadrature_with_values, project_with_gauss
from helpers.energy_based_model import EnergyBasedModel
from helpers.legendre import scaled_legendre
from scipy.special import roots_legendre

# loading
import pickle

# timing
from timeit import default_timer as timer

# main methods
from main.time_discretization import projection_method

def eval_proj_solution(
        ebm: EnergyBasedModel,
        tt_ref: jnp.ndarray,
        proj_solution: dict,
        resample_step: int,
        ):
    """
    this method evaluates the spp solution on a given reference
    timeframe. the method assumes that

    tt_ref[::k] == tt_spp

    for some number k, here called "resample_step". another assumption
    is that both tt_ref and tt_spp are equally spaced.

    :param ebm: energy based model for which dynamics should be approximated
    :param tt_ref: reference array of timepoints to sample to
    :param proj_solution: output of projection_method
    :param resample_step: number k such that tt_ref[::k] = tt_spp
    :return: values of spp solution at timepoints specified in tt_ref
    """
    
    d1, d2, d3 = ebm.dims

    tt_proj, zz_proj, dt_zz_12_proj = proj_solution['boundaries']
    coefflist = proj_solution['coefflist']
    degree = proj_solution['degree']

    assert jnp.allclose(tt_ref[::resample_step] - tt_proj, 0)

    tt_j = tt_ref[:resample_step]
    t_j, t_jp1 = tt_proj[0], tt_proj[1]
    tt_j_shift = (2*tt_j - t_j - t_jp1)/(t_jp1 - t_j) # shift to [-1,1]

    # since tt_j_shift will be the same for every interval, we can precompute scaled_legendre once
    # returns psi evaluations (shape: degree+1, t) and their derivatives
    scaled_legendre_values, dt_scaled_legendre_values = scaled_legendre(degree, tt_j_shift)

    M = coefflist.shape[0]
    num_coeffs_12 = (degree+1) * (d1+d2)
    
    # unpack the flattened coefficients into the distinct shapes for z1,z2 and z3
    coeffs12 = coefflist[:, :num_coeffs_12].reshape((M, degree+1, d1+d2))
    coeffs3 = coefflist[:, num_coeffs_12:].reshape((M, degree, d3))

    # calculate values in subintervals with einsum -> for every subinterval at once
    # z1 and z2 use all basis functions (degree + 1)
    values_12_j_all = jnp.einsum('Mkd,kt->Mtd', coeffs12, scaled_legendre_values)
    # z3 uses one degree less (degree), so we drop the highest-order basis function (:-1)
    values_3_j_all = jnp.einsum('Mkd,kt->Mtd', coeffs3, scaled_legendre_values[:-1, :])

    # concatenate them along the dimension axis
    values_j_all = jnp.concatenate((values_12_j_all, values_3_j_all), axis=-1)

    # the values we need are just a reshape away
    values = values_j_all.reshape((-1,d1+d2+d3))

    # the final value is missed, set it manually
    values = jnp.concatenate((values, zz_proj[-1,:][None,:]),axis=0)
    
    # handle derivative (only for z1 and z2)
    if d1+d2 == 0:
        dt_values_12 = jnp.zeros((tt_ref.shape[0]-1, 0))
    else:
        dt_values_12_j_all = jnp.einsum('Mkd,kt->Mtd', coeffs12, dt_scaled_legendre_values) * 2/(t_jp1 - t_j)
        dt_values_12 = dt_values_12_j_all.reshape((-1,d1+d2))
        dt_values_12 = jnp.concatenate((dt_values_12, dt_zz_12_proj[-1,:d1+d2][None,:]),axis=0)
        dt_values_12 = dt_values_12[1:,:] # exclude the first value in the derivative as it is not meaningful

    return values, dt_values_12

def calculate_projection_method_errors(
        ebm: EnergyBasedModel,
        T: float,
        nt_array: jnp.ndarray | list,
        degree: int,
        num_quad_nodes: int,
        num_proj_nodes: int,
        z0: jnp.ndarray,
        control: callable,
        tt_ref: jnp.ndarray,
        zz_ref: jnp.ndarray,
        ref_order_smaller: int,
        g_manufactured_solution: callable = None,
        use_projection: bool = True,
        use_pickle: bool = True,
        nodal_superconvergence: bool = False,
        include_algebraic_error: bool = True,
        picklepath: str = None,
        debug: bool = False,
        ):
    
    # get necessary information from energy based model
    d1, d2, d3 = ebm.dims
    
    if use_pickle: assert picklepath is not None, 'if use_pickle is True, picklepath must be specified'
    
    errors = []

    for k, nt in enumerate(nt_array):
        
        nt = nt.item()

        picklename = f'{picklepath}_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}'
        if not use_projection:
            picklename += '_no_projection'
        
        proj_solution = None
        
        if use_pickle:
            try:
                with open(f'{picklename}.pickle','rb') as f:
                    proj_solution = pickle.load(f)['proj_solution']
                print(f'({degree = }, {num_quad_nodes = }, {num_proj_nodes = }, {nt = })\n\tresult was loaded')
            except FileNotFoundError:
                pass
            
        if proj_solution is None:
            
            tt = jnp.linspace(0,T,nt)

            s_proj = timer()
            proj_solution = projection_method(
                ebm=ebm,
                tt=tt,
                z0=z0,
                control=control,
                degree=degree,
                num_proj_nodes=num_proj_nodes,
                num_quad_nodes=num_quad_nodes,
                g_manufactured_solution=g_manufactured_solution,
                use_projection=use_projection,
                debug=debug,
                )
            e_proj = timer()
            print(f'({degree = }, {num_quad_nodes = }, {num_proj_nodes = }, {nt = })\n\tdone, took {e_proj-s_proj:.2f} seconds')

            # save file
            with open(f'{picklename}.pickle','wb') as f:
                pickle.dump({'proj_solution':proj_solution},f)
            print(f'\tresult was written')
        
        if nodal_superconvergence: # analyze superconvergence

            # eval reference solution on coarse time gitter
            tt, zz_proj, dt_zz_12_proj = proj_solution['boundaries'] # function values at boundaries
            zz_ref_resampled = zz_ref[::2**(k+ref_order_smaller),:]
        
            # compute error in non-algebraic variables
            error_12 = zz_ref_resampled[:,:d1+d2] - zz_proj[:,:d1+d2]
            
            # z3 variable is not supported in nodal_superconvergence
            error_3 = 0.0*zz_ref_resampled[:,d1+d2:]
            
        else:
            
            # eval proj solution on reference gitter for z1 and z2 variables
            zz_proj_on_tt_ref, dt_zz_12_proj_on_tt_ref = eval_proj_solution(
                ebm=ebm,
                tt_ref=tt_ref,
                proj_solution=proj_solution,
                resample_step=2**(k+ref_order_smaller)
                )
            
            # compute errors in z1+z2 and z3 variables
            error_12 = zz_ref[:,:d1+d2] - zz_proj_on_tt_ref[:,:d1+d2 ] # shape (tt_ref, d1+d2)
            error_3 = zz_ref[:,d1+d2:] - zz_proj_on_tt_ref[:,d1+d2:] # shape (tt_ref, d3)


        # compute error in non-algebraic variables
        if ebm.was_pde:
            spatial_norm_12 = jax.vmap(ebm.space.get_norm, in_axes=0)
            
            if hasattr(ebm, 'has_two_components'):
                error_component_1 = spatial_norm_12(error_12[:, :ebm.space.dim])
                error_component_2 = spatial_norm_12(error_12[:, ebm.space.dim:])
                error_12 = error_component_1 + error_component_2
            else:
                error_12 = spatial_norm_12(error_12)
        else:
            error_12 = jnp.linalg.norm(error_12, axis=1)
            
        max_error_12 = float(jnp.max(error_12))

        # compute error in algebraic variables
        if ebm.was_pde and include_algebraic_error:
            spatial_norm_3 = jax.vmap(lambda x: ebm.space.get_norm(x), in_axes=0)
        else:
            spatial_norm_3 = jax.vmap(lambda x: jnp.sqrt(x.T @ jnp.eye(d3) @ x), in_axes=0)

        error_3 = spatial_norm_3(error_3)
        max_error_3 = float(jnp.max(error_3))

        # compute norming in non-algebraic variables
        if d1+d2 != 0:
            zz_ref_12 = zz_ref_resampled[:,:d1+d2] if nodal_superconvergence else zz_ref[:,:d1+d2]
            
            if ebm.was_pde:
                if hasattr(ebm, 'has_two_components'):
                    norming_12 = spatial_norm_12(zz_ref_12[:, :ebm.space.dim]) + spatial_norm_12(zz_ref_12[:, ebm.space.dim:])
                else:
                    norming_12 = spatial_norm_12(zz_ref_12)
                max_error_12 = max_error_12 / float(jnp.max(norming_12))
            else:
                max_error_12 = max_error_12 / float(jnp.max(jnp.linalg.norm(zz_ref_12, axis=1)))
        
        # compute norming in algebraic variables
        if d3 != 0:
            max_norming_3 = jnp.max(spatial_norm_3(zz_ref[:, d1+d2:]))
            max_error_3 = max_error_3 / max_norming_3

        
        # save errors
        e = max_error_12
        if include_algebraic_error:
            e += max_error_3
        errors.append(e)
        
    return errors

def energy_balance_error(
        proj_solution: dict,
        ebm: EnergyBasedModel,
        control: callable,
        relative: bool = True,
        ):

    tt, zz, _ = proj_solution['boundaries']
    degree = proj_solution['degree']
    num_quad_nodes = proj_solution['num_quad_nodes']
    num_proj_nodes = proj_solution['num_proj_nodes']
    
    R = ebm.R_vmap
    B = ebm.B_vmap
    ham = ebm.hamiltonian_vmap
    nabla_2_ham = ebm.nabla_2_ham_vmap

    coefflist = proj_solution['coefflist']

    # dimensions
    d1, d2, d3 = ebm.dims
    M = coefflist.shape[0]

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
    
    # prepare output arrays
    int_supply = jnp.zeros((M,))
    int_dissip = jnp.zeros((M,))

    # loop over time steps
    def body(k, tup):

        int_supply, int_dissip = tup

        tk, tkp1 = tt[k], tt[k+1]

        # Extract the flattened coefficients for the k-th timestep
        coeffs_flat = coefflist[k, :]
        num_coeffs_12 = (degree+1) * (d1+d2)
        
        # Unpack them into their respective variables and shapes
        coeffs12 = coeffs_flat[:num_coeffs_12].reshape((degree+1, d1+d2))
        coeffs1 = coeffs12[:, :d1]
        coeffs2 = coeffs12[:, d1:d1+d2]
        coeffs3 = coeffs_flat[num_coeffs_12:].reshape((degree, d3))

        # find dt_z1 at nqn_gauss_points and dt_z2 at n_gauss_points, d = d1 or d = d2
        dt_z1_nqn = jnp.einsum('Nd,Nt->td', coeffs1, psi_prime_at_nqn_gauss_points) * 2/(tkp1-tk) # factor for compensating for chain rule in shift: f on [-1,1] -> sqrt(2/(b-a)) f((2t - a - b)/(b-a)) on [a,b]

        # find z1, z2, z3 at nqn_gauss_points and z1, z2 at proj_gauss_points
        z1_npn = jnp.einsum('Nd,Nt->td', coeffs1, psi_at_npn_gauss_points) # d = d1
        z2_npn = jnp.einsum('Nd,Nt->td', coeffs2, psi_at_npn_gauss_points) # d = d2
        z3_nqn = jnp.einsum('nd,nt->td', coeffs3, phi_at_nqn_gauss_points) # d = d3
        
        # find projection of nabla_2_ham at nqn_gauss_points
        proj_nabla_2_ham_nqn = project_with_gauss(npn_gauss_weights, phi_at_npn_gauss_points, nabla_2_ham(z1_npn, z2_npn), evaluate_with=phi_at_nqn_gauss_points)
        
        # define test vector [dt_z1, proj_nabla_2_ham, z3] on nqn_gauss_points
        dt_z1__proj_nabla_2_ham__z3_nqn = jnp.hstack((dt_z1_nqn, proj_nabla_2_ham_nqn, z3_nqn)) # shape (t, d1+d2+d3)
        
        # find values of (J - R) [ P_{n-1}[eta(z)] ] at nqn_gauss_points
        dissip_nqn = jnp.einsum('tD,tD->t', dt_z1__proj_nabla_2_ham__z3_nqn, R(dt_z1_nqn, proj_nabla_2_ham_nqn, z3_nqn))

        # find values of B u at nqn_gauss_points and multiply with test vector [dt_z1, proj_nabla_2_ham, z3]
        shifted_nqn_gauss_points = (tkp1-tk)/2 * nqn_gauss_points + (tk+tkp1)/2
        Bu_nqn = B(control(shifted_nqn_gauss_points))
        supply_nqn = jnp.einsum('tD,tD->t', dt_z1__proj_nabla_2_ham__z3_nqn, Bu_nqn)

        # calculate integrals
        int_supply_k = gauss_quadrature_with_values(nqn_gauss_weights, supply_nqn, interval=(tk,tkp1))
        int_dissip_k = gauss_quadrature_with_values(nqn_gauss_weights, dissip_nqn, interval=(tk,tkp1))
        
        # store
        int_supply = int_supply.at[k].set(int_supply_k)
        int_dissip = int_dissip.at[k].set(int_dissip_k)

        return int_supply, int_dissip

    int_supply, int_dissip = jax.lax.fori_loop( 0, M, body, (int_supply, int_dissip) ) # this is way faster than a python loop

    # calculate energy balance
    HH = ham(zz[:, :d1], zz[:, d1:d1+d2])
    Hdiff = HH[1:] - HH[:-1]
    integral = int_supply - int_dissip
    error = jnp.abs(Hdiff - integral)

    if relative:
        error = error / jnp.max(jnp.abs(Hdiff))
    
    return error


if __name__ == '__main__':
    pass