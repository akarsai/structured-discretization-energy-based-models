#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file is used to generate the convergence plots in the publication.
#
#

# jax
import jax.numpy as jnp

# utility
import pickle
from itertools import product
from timeit import default_timer as timer

# time discretization method
from main.time_discretization import projection_method

# error calculation
from helpers.errors import calculate_projection_method_errors

# plotting
import matplotlib.pyplot as plt

# examples
from examples.nonlinear_circuit import NonlinearCircuit
from examples.doubly_nonlinear_parabolic import DoublyNonlinearParabolic
from examples.cahn_hilliard import CahnHilliard

# eoc table code
from helpers.other import generate_eoc_table_tex_code

def varying_degree(
        kind: str,
        T: float,
        degrees: list[int],
        ebm_kwargs: dict = None,
        num_quad_nodes_list: list[int] = None,
        num_proj_nodes_list: list[int] = None,
        base_Delta_t: float = 1e-3,
        num_Delta_t_steps: int = 9,
        ref_order_smaller: int = 3,
        nodal_superconvergence: bool = False,
        use_projection: bool = True,
        save: bool = True,
        use_pickle: bool = True,
        legend_loc: str = 'best',
        include_algebraic_error: bool = False,
        savepath_suffix: str = '',
        eoc_table = False,
        with_legend = True,
        debug: bool = False,
        ):
    """
    creates the varying_degree plots
    """

    # general settings
    Delta_t_array = jnp.array([2**i * base_Delta_t for i in range(num_Delta_t_steps)])

    # prepare input parameters
    assert kind in ['nonlinear_circuit', 'doubly_nonlinear_parabolic', 'cahn_hilliard']
    if ebm_kwargs is None:
        ebm_kwargs = {}
    
    # default num_quad_nodes_list and num_proj_nodes_list
    if num_quad_nodes_list is None:
        num_quad_nodes_list = [None for k in degrees]
    if num_proj_nodes_list is None:
        num_proj_nodes_list = [None for k in degrees]
    assert len(num_quad_nodes_list) == len(degrees)
    assert len(num_proj_nodes_list) == len(degrees)

    # setup energy based model
    ebm = None
    if kind == 'nonlinear_circuit':
        ebm = NonlinearCircuit()
        num_proj_nodes_list = [2*k for k in degrees] # while this would be correct, we omit this to demonstrate that even with inexact projection the convergence is good
    elif kind == 'cahn_hilliard':
        ebm = CahnHilliard(**ebm_kwargs)
        num_quad_nodes_list = [2*k for k in degrees]
        num_proj_nodes_list = [2*k for k in degrees]
    elif kind == 'doubly_nonlinear_parabolic':
        ebm = DoublyNonlinearParabolic(**ebm_kwargs)
        num_quad_nodes_list = [2*k for k in degrees]
        num_proj_nodes_list = [2*k for k in degrees]
        
    # get desired manufactured solution and associated inputs
    z0_manufactured_solution, manufactured_solution, control_manufactured_solution, g_manufactured_solution = ebm.get_manufactured_solution()
    # z0 = ebm.initial_condition
    # control = ebm.default_control
    
    print(f'\n\n--- running varying_degree for {kind} system ---')
    if ebm_kwargs:
        print(f'--- settings: {ebm_kwargs} ---')
        
    # to save pickle files and figures
    savepath = f'{SAVEPATH}/figures/{kind}'
    picklepath = f'{SAVEPATH}/pickle/{kind}'
    
    # add ebm_kwargs to savepath to prevent duplicates
    ebm_kwargs_string = ''
    if ebm_kwargs:
        ebm_kwargs_string = '_' + '_'.join([f'{k}{v}' for k, v in ebm_kwargs.items()])
        savepath += ebm_kwargs_string
        picklepath += ebm_kwargs_string
    
    # note that manufactured solutions are used
    picklepath += '_manu'

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
    zz_ref = manufactured_solution(tt_ref)

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
            ref_order_smaller=ref_order_smaller,
            # manufactured_solution=manufactured_solution, # only needed for L2 error in z3 variable
            g_manufactured_solution=g_manufactured_solution,
            use_projection=use_projection,
            use_pickle=use_pickle,
            nodal_superconvergence=nodal_superconvergence,
            include_algebraic_error=include_algebraic_error,
            picklepath=picklepath,
            debug=debug,
            )
        all_errors[degree] = errors_for_this_degree
        all_errors_array = all_errors_array.at[:,index].set(jnp.flip(jnp.array(errors_for_this_degree)))

        # print(f'{errors_for_this_degree = }\n')
        print(f'{degree = } done\n')

        # marker_data, marker_fit = markerlist[index]
        marker_data, marker_fit = '.', 'none'

        ax.loglog(
            Delta_t_array, errors_for_this_degree,
            label=f'$k = {degree}$',
            marker=marker_data,
            linewidth=3.0,
            markersize=10.0,
            # alpha=0.6,
            color=plt.cm.tab20(2*index),
            )

        # add linear fit
        if kind in ['nonlinear_circuit']:
            if degree % 2 == 0:
                slope = degree
            else:
                slope = degree + 1
            if include_algebraic_error:
                slope = slope - 1
        else:
            slope = degree+1
            if include_algebraic_error:
                slope = degree
            if nodal_superconvergence:
                slope = 2*degree
        
        slope_index = -5
        if kind in ['doubly_nonlinear_parabolic']:
            slope_index = -2
        if kind in ['nonlinear_circuit']:
            slope_index = -3
        c = errors_for_this_degree[slope_index]/Delta_t_array[slope_index]**(slope) # find coefficient to match Delta_t^p to curves
        ax.loglog(
            Delta_t_array, c * Delta_t_array**(slope),
            label=f'$\\tau^{{{slope}}}$',
            linestyle='--',
            marker=marker_fit,
            markersize=7,
            linewidth=5.0,
            alpha=0.4,
            color=plt.cm.tab20(2*index + 1),
            zorder=0,
            )
        
    print('\n----\n')

    
    # create EOC table code
    if eoc_table:
        eoc_table_tex_code = generate_eoc_table_tex_code(
            tau_list=jnp.flip(Delta_t_array),
            k_list=jnp.array(degrees),
            error_list=all_errors_array,
            with_average=True,
            )

    # set plot properties
    if with_legend:
        ax.legend(loc=legend_loc)
    ax.set_xlabel('step size $\\tau$')
    ax.set_ylim(1.5e-18, 1.5e-1)
    if kind in ['doubly_nonlinear_parabolic']:
        ax.set_ylim(1.5e-14, 1.5e-1)
    elif kind in ['nonlinear_circuit'] and include_algebraic_error:
        ax.set_ylim(1.5e-13, 1.5e-0)
    
    ylabeltext = r'$\errorstate^{\mathrm{non-alg}}$'
    if include_algebraic_error or kind in ['toda', 'rigid_body', 'doubly_nonlinear_parabolic']:
        ylabeltext = r'$\errorstate$'

    if nodal_superconvergence:
        # ylabeltext = '$\\frac{\\max\\limits_{t_0, \dots, t_m} \| z(t) - z_{\\tau}(t) \|}{\\max\\limits_{t_0, \dots, t_m} \| z(t)\|}$'
        ylabeltext = r'$\errorstatenodal$'


    ax.set_ylabel(ylabeltext)

    # saving the figure
    if save:
        savepath += f'_varying_degree{savepath_suffix}'
        if nodal_superconvergence: savepath += '_nodal_superconvergence'
        if not use_projection: savepath += '_no_projection'
        fig.tight_layout() # call tight_layout to be safe
        fig.savefig(savepath + '.pgf', bbox_inches='tight', pad_inches=0.01) # save as pgf
        fig.savefig(savepath + '.png', bbox_inches='tight', pad_inches=0.01) # save as png
        print(f'\n\nfigure saved under {savepath} (as pgf and png)')
        
        if eoc_table:
            with open(savepath + '_eoc.tex', 'w') as f:
                f.write(eoc_table_tex_code)
            print(f'eoc table saved under {savepath}_eoc.tex')

    # showing the figure
    fig.tight_layout()
    fig.show()
    
    return all_errors



if __name__ == '__main__':
    
    import jax
    jax.config.update("jax_enable_x64", True) # activate double precision
    
    # plotting
    from helpers.other import mpl_settings
    mpl_settings(fontsize=18)
    
    # set savepath
    SAVEPATH = './results'
    

    # nonlinear circuit
    varying_degree(
        kind='nonlinear_circuit',
        T=1,
        degrees=[1,2,3,4],
        save=True,
        use_pickle=True,
        include_algebraic_error=True,
        legend_loc='upper left',
        )

    # nonlinear circuit - non algebraic variables only
    varying_degree(
        kind='nonlinear_circuit',
        T=1,
        degrees=[1,2,3,4],
        save=True,
        use_pickle=True,
        include_algebraic_error=False,
        savepath_suffix='_nonalg',
        legend_loc='upper left',
        )
    
    # doubly nonlinear parabolic
    plist = [1.5]
    qlist = [1.5, 3]
    for (p,q) in product(plist, qlist):
        # temporal convergence
        varying_degree(
            kind='doubly_nonlinear_parabolic',
            T=0.1,
            degrees=[2,3,4],
            ebm_kwargs={'p': p, 'q': q, 'nx': 25},
            base_Delta_t=1e-4,
            num_Delta_t_steps=7,
            ref_order_smaller=3,
            save=True,
            use_pickle=True,
            eoc_table=False,
            with_legend=False if (p,q)==(1.5,1.5) else True,
            )
        # temporal convergence without projection
        varying_degree(
            kind='doubly_nonlinear_parabolic',
            T=0.1,
            degrees=[2,3,4],
            ebm_kwargs={'p': p, 'q': q, 'nx': 25},
            base_Delta_t=1e-4,
            num_Delta_t_steps=7,
            ref_order_smaller=3,
            use_projection=False,
            save=True,
            use_pickle=True,
            eoc_table=False,
            with_legend=False if (p,q)==(1.5,1.5) else True,
            )

    # cahn hilliard - non algebraic variables only
    varying_degree(
        kind='cahn_hilliard',
        T=1.5,
        degrees=[2,3,4],
        save=True,
        use_pickle=True,
        include_algebraic_error=False,
        legend_loc='upper left',
        savepath_suffix='_nonalg',
        )

    # cahn hilliard - all variables
    varying_degree(
        kind='cahn_hilliard',
        T=1.5,
        degrees=[2,3,4],
        save=True,
        use_pickle=True,
        include_algebraic_error=True,
        legend_loc='upper left',
        )
    
    






