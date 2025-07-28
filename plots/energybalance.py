#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file is used to generate the energy balance plots in the publication.
#
#

import jax
import jax.numpy as jnp

# main methods
from main.time_discretization import projection_method

# plotting
import matplotlib.pyplot as plt

# file saving
import pickle

# timing
from timeit import default_timer as timer

# error calculation
from helpers.errors import energy_balance_error

# examples
from examples.cahn_hilliard import CahnHilliard, CahnHilliardReducedOrder
from examples.acdc import ACDC

def energybalance(
        kind: str,
        T: float,
        degrees: list[int],
        nt: int = None,
        num_quad_nodes_list: list[int] = None,
        num_proj_nodes_list: list[int] = None,
        save: bool = True,
        use_pickle: bool = True,
        legend_loc: str = 'best',
        debug: bool = False,
        with_rom: bool = False,
        ):
    """
    creates the energybalance plots
    """

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
    ebm_fom = None # overwrite this
    if kind == 'cahn_hilliard':
        ebm_fom = CahnHilliard()
        ebm_rom = CahnHilliardReducedOrder()
        num_quad_nodes_list = [2*k for k in degrees] # although k+1 should be sufficient, the newton solver has difficulties for k+1 quadnodes. we therefore default to a higher number of quadnodes.
        num_proj_nodes_list = [2*k for k in degrees]
    elif kind == 'acdc':
        ebm_fom = ACDC()
        num_proj_nodes_list = [2*k for k in degrees]

    print(f'\n\n--- testing energybalance for {kind} system ---')

    # to save pickle files and figures
    savepath = f'{SAVEPATH}/figures/{kind}'
    picklepath = f'{SAVEPATH}/pickle/{kind}'

    # set up time interval
    if nt is None: nt = int(T*100)+1
    tt = jnp.linspace(0,T, nt) # t_i = i * T/nt
    
    if with_rom:
        orderlist = ['fom', 'rom']
        fig, (ax_fom, ax_rom) = plt.subplots(ncols=2, figsize=(11,4), layout='constrained')
        fig.get_layout_engine().set(wspace=0.1, h_pad=0.05)
        
    else:
        orderlist = ['fom']
        # set up plot environment
        fig, ax_fom = plt.subplots()
    
    for order in orderlist:
        print(f'\n--- {order = } ---')
        ebm = ebm_rom if order == 'rom' else ebm_fom
        picklepath_order = f'{picklepath}_rom' if order == 'rom' else picklepath
        ax = ax_rom if order == 'rom' else ax_fom
    
        ax.set_xlabel('time $t$')
        ax.set_ylabel(r'$\errorenergy$')
        ax.set_ylim(1.5e-18, 1.5e-3)
        if kind == 'acdc':
            ax.set_ylim(1.5e-21, 1.5e-6)
        
        for index, degree in enumerate(degrees):
    
            num_proj_nodes = num_proj_nodes_list[index]
            num_quad_nodes = num_quad_nodes_list[index]
            picklename = f'{picklepath_order}_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}'
            
            proj_solution = None
            
            if use_pickle:
                try: # try to skip also the evaluation
                    with open(f'{picklename}.pickle','rb') as f:
                        proj_solution = pickle.load(f)['proj_solution']
                    print(f'({degree = }, {num_quad_nodes = }, {num_proj_nodes = }, {nt = })\n\tresult was loaded')
                except FileNotFoundError:
                    pass
            
            if proj_solution is None: # evaluation was not done before
                s = timer()
                proj_solution = projection_method(
                    ebm=ebm,
                    tt=tt,
                    z0=ebm.initial_condition,
                    control=ebm.default_control,
                    degree=degree,
                    num_quad_nodes=num_quad_nodes,
                    num_proj_nodes=num_proj_nodes,
                    debug=debug,
                    )
                e = timer()
                print(f'({degree = }, {num_quad_nodes = }, {num_proj_nodes = }, {nt = })\n\ttook {e-s:.2f} seconds')
    
                # save file
                with open(f'{picklename}.pickle','wb') as f:
                    pickle.dump({'proj_solution': proj_solution},f)
                print(f'\tresult was written')
    
            eb_error = energy_balance_error(
                proj_solution,
                ebm,
                ebm.default_control,
                )
    
            ax.semilogy(
                tt[1:], eb_error,
                label=rf'$k = {degree},~ \projnodes = {num_proj_nodes}$' if order=='fom' else None,
                color=plt.cm.tab20(2*index),
                linewidth=3.0,
                )
    
            # test influence of projection nodes for cahn hilliard model
            if kind in ['cahn_hilliard']:
    
                num_proj_nodes -= 1
                picklename = f'{picklepath_order}_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}' # needs to be updated
    
                try: # try to skip also the evaluation
                    with open(f'{picklename}.pickle','rb') as f:
                        proj_solution = pickle.load(f)['proj_solution']
                    print(f'({degree = }, {num_quad_nodes = }, {num_proj_nodes = }, {nt = })\n\tresult was loaded')
    
                except FileNotFoundError: # evaluation was not done before
                    s = timer()
                    proj_solution = projection_method(
                        ebm=ebm,
                        tt=tt,
                        z0=ebm.initial_condition,
                        control=ebm.default_control,
                        degree=degree,
                        num_quad_nodes=num_quad_nodes,
                        num_proj_nodes=num_proj_nodes,
                        debug=debug,
                        )
                    e = timer()
                    print(f'({degree = }, {num_quad_nodes = }, {num_proj_nodes = }, {nt = })\n\ttook {e-s:.2f} seconds')
    
                    # save file
                    with open(f'{picklename}.pickle','wb') as f:
                        pickle.dump({'proj_solution': proj_solution},f)
                    print(f'\tresult was written')
    
                eb_error = energy_balance_error(
                    proj_solution,
                    ebm,
                    ebm.default_control,
                    )
    
                ax.semilogy(
                    tt[1:], eb_error,
                    label=rf'$k = {degree},~ \projnodes = {num_proj_nodes}$' if order=='fom' else None,
                    color=plt.cm.tab20(2*index),
                    linestyle='dotted',
                    linewidth=3.0,
                    )

    if with_rom:
        fig.legend(loc=legend_loc, ncols=len(degrees), framealpha=1.)
    else:
        ax_fom.legend(loc=legend_loc)

    # save + show
    if save:
        savepath = savepath + '_energybalance'
        if not with_rom:
            fig.tight_layout()
        fig.savefig(savepath + '.pgf') # save as pgf
        fig.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    fig.title = f'error in energy balance for different methods, {kind}'
    if not with_rom:
        fig.tight_layout()
    fig.show()

    return





if __name__ == '__main__':
    
    jax.config.update("jax_enable_x64", True) # activate double precision
    
    # plotting
    from helpers.other import mpl_settings
    mpl_settings(fontsize=20)
    
    # set savepath
    SAVEPATH = './results'
    
    # cahn_hilliard
    energybalance(
        kind='cahn_hilliard',
        T=1.5,
        degrees=[2,3,4],
        save=True,
        use_pickle=True,
        legend_loc='outside upper center',
        with_rom=True,
        )
    
    # cahn_hilliard
    energybalance(
        kind='acdc',
        T=5,
        degrees=[2,3,4],
        save=True,
        use_pickle=True,
        )

    
    