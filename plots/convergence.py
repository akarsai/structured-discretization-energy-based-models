#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file is used to generate the convergence plots in the publication.
#
#

import jax
import jax.numpy as jnp

from helpers.errors import calculate_projection_method_errors

# plotting
import matplotlib.pyplot as plt

# examples
from examples.cahn_hilliard import CahnHilliard
from examples.acdc import ACDC

def varying_degree(
        kind: str,
        T: float,
        degrees: list[int],
        num_quad_nodes_list: list[int] = None,
        num_proj_nodes_list: list[int] = None,
        nodal_superconvergence: bool = False,
        save: bool = True,
        use_pickle: bool = True,
        legend_loc: str = 'best',
        include_algebraic_error: bool = False,
        savepath_suffix: str = '',
        debug: bool = False,
        ):
    """
    creates the varying_degree plots
    """

    # general settings
    base_Delta_t = 1e-3
    num_Delta_t_steps = 9
    Delta_t_array = jnp.array([2**i * base_Delta_t for i in range(num_Delta_t_steps)])
    ref_order_smaller = 3 # by which order of magnitude should the reference solution be smaller than the smallest tested Delta t? # change to zero if using scheme as reference solution

    # prepare input parameters
    assert kind in ['cahn_hilliard', 'acdc']
    
    # default num_quad_nodes_list and num_proj_nodes_list
    if num_quad_nodes_list is None:
        num_quad_nodes_list = [None for k in degrees]
    if num_proj_nodes_list is None:
        num_proj_nodes_list = [None for k in degrees]
    assert len(num_quad_nodes_list) == len(degrees)
    assert len(num_proj_nodes_list) == len(degrees)

    # setup energy based model
    ebm = None
    if kind == 'cahn_hilliard':
        ebm = CahnHilliard()
        num_quad_nodes_list = [2*k for k in degrees]
        num_proj_nodes_list = [2*k for k in degrees]
    elif kind == 'acdc':
        ebm = ACDC()
        num_proj_nodes_list = [2*k for k in degrees]
        
    # get desired manufactured solution and associated inputs
    z0_manufactured_solution, manufactured_solution, control_manufactured_solution, g_manufactured_solution = ebm.get_manufactured_solution()
    # z0 = ebm.initial_condition
    # control = ebm.default_control
    
    print(f'\n\n--- running varying_degree for {kind} system ---')
    
    # to save pickle files and figures
    savepath = f'{SAVEPATH}/figures/{kind}'
    picklepath = f'{SAVEPATH}/pickle/{kind}_manu'

    # convert Delta_t values to nt values
    N = (T/Delta_t_array[-1]).astype(int)
    nt_array = jnp.flip(jnp.array([N * 2**(order) + 1 for order in range(num_Delta_t_steps)]))
    nt_ref = 2**(ref_order_smaller+num_Delta_t_steps-1) * N + 1

    # convert back to get "real" Delta_t corresponding to the nt_spp values
    Delta_t_array = T/nt_array
    Delta_t_ref = T/nt_ref

    print(f'\nnt_ref = {nt_ref}\nDelta_t_ref = {Delta_t_ref:e}')
    print(f'\nnt_array = {nt_array}\nDelta_t_array = {Delta_t_array}\n')

    # obtain reference solution and corresponding inputs
    tt_ref = jnp.linspace(0,T,nt_ref)
    # # alternative approach: reference solution is computed with same scheme of higher order
    # kmax = max(degrees) + 1
    # nqn_max = None
    # npn_max = None
    # if kind == 'cahn_hilliard':
    #     nqn_max = 2*kmax
    #     npn_max = 2*kmax
    # elif kind == 'acdc':
    #     npn_max = 2*kmax
    # picklename = f'{picklepath}_n{kmax}_qn{nqn_max}_pn{npn_max}_M{nt_ref}'
    # ref_solution = None
    # if use_pickle:
    #     try:
    #         with open(f'{picklename}.pickle','rb') as f:
    #             ref_solution = pickle.load(f)['proj_solution']
    #         print(f'(degree = {kmax}, num_quad_nodes = {nqn_max}, num_proj_nodes = {npn_max}, nt = {nt_ref})\n\tresult was loaded')
    #     except FileNotFoundError:
    #         pass
    # if ref_solution is None:
    #     s_proj = timer()
    #     ref_solution = projection_method(
    #         ebm=ebm,
    #         tt=tt_ref,
    #         z0=z0,
    #         control=control,
    #         degree=kmax,
    #         num_proj_nodes=npn_max,
    #         num_quad_nodes=nqn_max,
    #         debug=debug,
    #         )
    #     e_proj = timer()
    #     print(f'(degree = {kmax}, num_quad_nodes = {nqn_max}, num_proj_nodes = {npn_max}, nt = {nt_ref})\n\tdone, took {e_proj-s_proj:.2f} seconds')
    #     # save file
    #     with open(f'{picklename}.pickle','wb') as f:
    #         pickle.dump({'proj_solution':ref_solution},f)
    #     print(f'\tresult was written')
    # zz_ref = ref_solution['boundaries'][1]
    zz_ref = manufactured_solution(tt_ref)
    B_u_tt_ref = ebm.B_vmap(control_manufactured_solution(tt_ref))
    g_tt_ref = g_manufactured_solution(tt_ref)

    ### calculate and plot in one go
    fig, ax = plt.subplots()

    # spp calculation + plot
    all_errors = {}
    all_errors_array = jnp.zeros((len(Delta_t_array),len(degrees)))

    for index, degree in enumerate(degrees):

        errors_for_this_degree = calculate_projection_method_errors(
            ebm=ebm,
            T=T,
            nt_array=nt_array,
            degree=degree,
            num_quad_nodes=num_quad_nodes_list[index],
            num_proj_nodes=num_proj_nodes_list[index],
            z0=z0_manufactured_solution,
            control=control_manufactured_solution,
            tt_ref=tt_ref,
            zz_ref=zz_ref,
            B_u_tt_ref=B_u_tt_ref,
            g_tt_ref=g_tt_ref,
            ref_order_smaller=ref_order_smaller,
            g_manufactured_solution=g_manufactured_solution,
            use_pickle=use_pickle,
            nodal_superconvergence=nodal_superconvergence,
            include_algebraic_error=include_algebraic_error,
            picklepath=picklepath,
            debug=debug,
            )
        all_errors[degree] = errors_for_this_degree
        all_errors_array = all_errors_array.at[:,index].set(jnp.flip(jnp.array(errors_for_this_degree)))

        print(f'{degree = } done\n')

        # marker_data, marker_fit = markerlist[index]
        marker_data, marker_fit = '.', 'none'

        ax.loglog(Delta_t_array, errors_for_this_degree,
                   label=f'$k = {degree}$',
                   marker=marker_data,
                   linewidth=3.0,
                   markersize=10.0,
                   color=plt.cm.tab20(2*index))

        # add linear fit
        slope = degree+1
        if include_algebraic_error:
            slope = degree
        if nodal_superconvergence:
            slope = 2*degree
        c = errors_for_this_degree[-5]/Delta_t_array[-5]**(slope) # find coefficient to match Delta_t^p to curves
        ax.loglog(
            Delta_t_array, c * Delta_t_array**(slope),
            label=f'$\\tau^{{{slope}}}$',
            linestyle='--',
            marker=marker_fit,
            markersize=7,
            linewidth=4.0,
            color=plt.cm.tab20(2*index + 1),
            zorder=0,
            )

    print('\n----\n')

    # set plot properties
    ax.legend(loc=legend_loc)
    ax.set_xlabel('step size $\\tau$')
    ax.set_ylim(1.5e-18, 1.5e-1)
    
    ylabeltext = r'$\errorstate^{\text{non-alg}}$'
    if include_algebraic_error:
        ylabeltext = r'$\errorstate$'

    if nodal_superconvergence:
        # ylabeltext = '$\\frac{\\max\\limits_{t_0, \dots, t_m} \| z(t) - z_{\\tau}(t) \|}{\\max\\limits_{t_0, \dots, t_m} \| z(t)\|}$'
        ylabeltext = 'nodal error'

    ax.set_ylabel(ylabeltext)

    # saving the figure
    if save:
        savepath += f'_varying_degree{savepath_suffix}'
        if nodal_superconvergence: savepath += '_nodal_superconvergence'
        fig.tight_layout() # call tight_layout to be safe
        fig.savefig(savepath + '.pgf') # save as pgf
        fig.savefig(savepath + '.png') # save as png
        print(f'\n\nfigure saved under {savepath} (as pgf and png)')

    # showing the figure
    fig.tight_layout()
    fig.show()
    
    return all_errors


if __name__ == '__main__':
    
    jax.config.update("jax_enable_x64", True) # activate double precision
    
    # plotting
    from helpers.other import mpl_settings
    mpl_settings(fontsize=20)
    
    # set savepath
    SAVEPATH = './results'
    
    # cahn hilliard
    varying_degree(
        kind='cahn_hilliard',
        T=1.5,
        degrees=[2,3,4],
        save=True,
        use_pickle=True,
        include_algebraic_error=False,
        legend_loc='upper left',
        )
    
    # ACDC converter
    varying_degree(
        kind='acdc',
        T=5,
        degrees=[2,3,4],
        save=True,
        use_pickle=True,
        include_algebraic_error=True,
        legend_loc='lower right',
        )
    
    # ACDC converter - non algebraic variables only
    varying_degree(
        kind='acdc',
        T=5,
        degrees=[2,3,4],
        save=True,
        use_pickle=True,
        include_algebraic_error=False,
        savepath_suffix='_nonalg',
        legend_loc='lower right',
        )









