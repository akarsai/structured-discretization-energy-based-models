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
        resample_step: float,
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
    scaled_legendre_values, dt_scaled_legendre_values = scaled_legendre(degree, tt_j_shift)

    # calculate values in subintervals with einsum -> for every subinterval at once
    values_j_all = jnp.einsum('MkD,kt->MtD', coefflist, scaled_legendre_values)

    # the values we need are just a reshape away
    values = values_j_all.reshape((-1,d1+d2+d3))

    # the final value is missed, set it manually
    values = jnp.concatenate((values, zz_proj[-1,:][None,:]),axis=0)
    
    # handle derivative
    if d1+d2 == 0:
        dt_values_12 = jnp.zeros((tt_ref.shape[0]-1, 0))
    else:
        dt_values_12_j_all = jnp.einsum('Mkd,kt->Mtd', coefflist[:,:,:d1+d2], dt_scaled_legendre_values) * 2/(t_jp1 - t_j)
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
        g_tt_ref: jnp.ndarray,
        B_u_tt_ref: jnp.ndarray,
        ref_order_smaller: int,
        g_manufactured_solution: callable = None,
        use_pickle: bool = True,
        nodal_superconvergence: bool = False,
        include_algebraic_error: bool = True,
        picklepath: str = None,
        debug: bool = False,
        ):
    
    # get necessary information from energy based model
    d1, d2, d3 = ebm.dims
    J = ebm.J_vmap
    R = ebm.R_vmap
    B = ebm.B_vmap
    nabla_2_ham = ebm.nabla_2_ham_vmap
    
    if use_pickle: assert picklepath is not None, 'if use_pickle is True, picklepath must be specified'
    
    errors = []
    num_Delta_t_steps = len(nt_array)

    for k, nt in enumerate(nt_array):
        
        nt = nt.item()

        picklename = f'{picklepath}_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}'
        
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
                debug=debug,
                )
            e_proj = timer()
            print(f'({degree = }, {num_quad_nodes = }, {num_proj_nodes = }, {nt = })\n\tdone, took {e_proj-s_proj:.2f} seconds')

            # save file
            with open(f'{picklename}.pickle','wb') as f:
                pickle.dump({'proj_solution':proj_solution},f)
            print(f'\tresult was written')
        
        # zz_proj_init = proj_solution['boundaries'][1][1,:]
        # zz_proj_mid = proj_solution['boundaries'][1][nt//2,:]
        # zz_proj_end = proj_solution['boundaries'][1][-1,:]
        # d1, d2, d3 = ebm.dims
        # ebm.space.visualize_coefficient_vector(zz_proj_init[:d1], title='$v$ start')
        # ebm.space.visualize_coefficient_vector(zz_proj_init[d1:], title='$w$ start')
        # ebm.space.visualize_coefficient_vector(zz_proj_mid[:d1], title='$v$ mid')
        # ebm.space.visualize_coefficient_vector(zz_proj_mid[d1:], title='$w$ mid')
        # ebm.space.visualize_coefficient_vector(zz_proj_end[:d1], title='$v$ end')
        # ebm.space.visualize_coefficient_vector(zz_proj_end[d1:], title='$w$ end')
        
        if nodal_superconvergence: # analyze superconvergence

            # eval reference solution on coarse time gitter
            tt, zz_proj, dt_zz_12_proj = proj_solution['boundaries'] # function values at boundaries
            zz_ref_resampled = zz_ref[::2**(k+ref_order_smaller),:]
        
            # compute error in non-algebraic variables
            error_12 = zz_ref_resampled[:,:d1+d2] - zz_proj[:,:d1+d2]
            
            # compute the defect in the algebraic constraint
            z1 = zz_proj[1:,:d1] # 1: in time axis to match other dimensions (no derivative at initial time point)
            z2 = zz_proj[1:,d1:d1+d2]
            z3 = zz_proj[1:,d1+d2:]
            dt_z1 = dt_zz_12_proj[:,:d1]
            h2 = nabla_2_ham(z1, z2)
            g = g_manufactured_solution(tt[1:])
            u = control(tt[1:])
            rhs = J(dt_z1, h2, z3) - R(dt_z1, h2, z3) + B(u) + g
            error_3 = rhs[:,d1+d2:]
            
        else:
            
            tt, zz_proj, dt_zz_12_proj = proj_solution['boundaries'] # function values at boundaries

            # eval proj solution on reference gitter
            zz_proj_on_tt_ref, dt_zz_12_proj_on_tt_ref = eval_proj_solution(
                ebm=ebm,
                tt_ref=tt_ref,
                proj_solution=proj_solution,
                resample_step=2**(k+ref_order_smaller)
                )
        
            # compute error in non-algebraic variables
            error_12 = zz_ref[:,:d1+d2] - zz_proj_on_tt_ref[:,:d1+d2]
            
            # # compute the defect in the algebraic constraint, only in time grid points
            # if include_algebraic_error:
            #     z1 = zz_proj[1:,:d1] # 1: in time axis to match other dimensions (no derivative at initial time point)
            #     z2 = zz_proj[1:,d1:d1+d2]
            #     z3 = zz_proj[1:,d1+d2:]
            #     dt_z1 = dt_zz_12_proj[:,:d1]
            #     h2 = nabla_2_ham(z1, z2)
            #     g = g_tt_ref[::2**(k+ref_order_smaller),:][1:,:]
            #     B_u = B_u_tt_ref[::2**(k+ref_order_smaller),:][1:,:]
            #     rhs = J(dt_z1, h2, z3) - R(dt_z1, h2, z3) + B_u + g
            #     error_3 = rhs[:,d1+d2:]
            # else: # we can skip the calculation
            #     error_3 = jnp.zeros((tt_ref, d3))
            
            error_3 = zz_ref[:,d1+d2:] - zz_proj_on_tt_ref[:,d1+d2:]

        # compute relative error in non-algebraic variables
        if ebm.was_pde:
            mass_norm = jax.vmap(lambda x: x.T @ ebm.space.mass_matrix @ x, in_axes=0)
            error_12 = jnp.sqrt(mass_norm(error_12)) # only works if z1 or z2 is non existent
        else:
            error_12 = jnp.linalg.norm(error_12, axis=1) # norms along axis 1, since axis 0 are the time points
        if not jnp.isclose(jnp.max(error_12), 0): # exclude d1+d2 = 0 case
            if ebm.was_pde:
                norming_12 = jnp.sqrt(mass_norm(zz_ref[:,:d1+d2])) # only works if z1 or z2 is non existent
            else:
                norming_12 = jnp.max(jnp.linalg.norm(zz_ref[:,:d1+d2], axis=1))
            error_12 = error_12 / norming_12
        
        # compute error in algebraic variables - not relative, since reference is zero
        # error_3 = jnp.linalg.norm(error_3, axis=1) # norms along axis 1, since axis 0 are the time points
        
        # compute relative error in algebraic variables
        if ebm.was_pde:
            mass_norm = jax.vmap(lambda x: x.T @ ebm.space.mass_matrix @ x, in_axes=0)
            error_3 = jnp.sqrt(mass_norm(error_3)) # only works if z1 or z2 is non existent
        else:
            error_3 = jnp.linalg.norm(error_3, axis=1) # norms along axis 1, since axis 0 are the time points
        if not jnp.isclose(jnp.max(error_3), 0): # exclude d3 = 0 case
            if ebm.was_pde:
                norming_3 = jnp.sqrt(mass_norm(zz_ref[:,d1+d2:])) # only works if z1 or z2 is non existent
            else:
                norming_3 = jnp.max(jnp.linalg.norm(zz_ref[:,d1+d2:], axis=1))
            error_3 = error_3 / norming_3
        
        # visualize
        # import matplotlib.pyplot as plt
        # fig_err, ax_err = plt.subplots()
        # ax_err.semilogy(error_3)
        # fig_err.suptitle(f'error_3 over time, {nt=}')
        # fig_err.show()
        
        # compute errors
        max_error_12 = float(jnp.max(error_12))
        min_error_3 = float(jnp.min(error_3))
        max_error_3 = float(jnp.max(error_3))
        avg_error_3 = float(jnp.average(error_3))
        
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

        coeffs123 = coefflist[k,:,:]
        coeffs1 = coeffs123[:,:d1]
        coeffs2 = coeffs123[:,d1:d1+d2]
        coeffs3 = coeffs123[:,d1+d2:]

        # find dt_z1 at nqn_gauss_points and dt_z2 at n_gauss_points, d = d1 or d = d2
        dt_z1_nqn = jnp.einsum('Nd,Nt->td', coeffs1, psi_prime_at_nqn_gauss_points) * 2/(tkp1-tk) # factor for compensating for chain rule in shift: f on [-1,1] -> sqrt(2/(b-a)) f((2t - a - b)/(b-a)) on [a,b]

        # find z1, z2, z3 at nqn_gauss_points and z1, z2 at proj_gauss_points
        z3_nqn = jnp.einsum('Nd,Nt->td', coeffs3, psi_at_nqn_gauss_points) # d = d3
        z1_npn = jnp.einsum('Nd,Nt->td', coeffs1, psi_at_npn_gauss_points) # d = d1
        z2_npn = jnp.einsum('Nd,Nt->td', coeffs2, psi_at_npn_gauss_points) # d = d2
        
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