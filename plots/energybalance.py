#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file is used to generate the energy balance plots in the publication.
#
#

# # add the parent directory to sys.path
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itertools import product

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
from examples.nonlinear_circuit import NonlinearCircuit
from examples.doubly_nonlinear_parabolic import DoublyNonlinearParabolic, DoublyNonlinearParabolicReducedOrder
from examples.cahn_hilliard import CahnHilliard, CahnHilliardReducedOrder

def energybalance(
        kind: str,
        T: float,
        degrees: list[int],
        nt: int = None,
        num_quad_nodes_list: list[int] = None,
        num_proj_nodes_list: list[int] = None,
        ebm_kwargs: dict = None,
        save: bool = True,
        use_pickle: bool = True,
        legend_loc: str = 'best',
        debug: bool = False,
        with_rom: bool = False,
        only_rom: bool = False,
        ):
    """
    creates the energybalance plots
    """

    # prepare input parameters
    assert kind in ['nonlinear_circuit', 'doubly_nonlinear_parabolic', 'cahn_hilliard']
    if ebm_kwargs is None:
        ebm_kwargs = {}
        
    # to save pickle files and figures
    savepath = f'{SAVEPATH}/figures/{kind}'
    picklepath = f'{SAVEPATH}/pickle/{kind}'
    
    # add ebm_kwargs to savepath to prevent duplicates
    ebm_kwargs_string = ''
    if ebm_kwargs:
        ebm_kwargs_string = '_' + '_'.join([f'{k}{v}' for k, v in ebm_kwargs.items()])
        savepath += ebm_kwargs_string
        picklepath += ebm_kwargs_string
        
    # default num_quad_nodes_list and num_proj_nodes_list
    if num_quad_nodes_list is None:
        num_quad_nodes_list = [None for k in degrees]
    if num_proj_nodes_list is None:
        num_proj_nodes_list = [None for k in degrees]
    assert len(num_quad_nodes_list) == len(degrees)
    assert len(num_proj_nodes_list) == len(degrees)
    
    # setup energy based model
    ebm_fom = None # overwrite this
    
    if kind == 'nonlinear_circuit':
        ebm_fom = NonlinearCircuit()
        num_proj_nodes_list = [2*k for k in degrees]
    elif kind == 'doubly_nonlinear_parabolic':
        ebm_fom = DoublyNonlinearParabolic(**ebm_kwargs)
        num_quad_nodes_list = [2*k for k in degrees]
        num_proj_nodes_list = [2*k for k in degrees]
    elif kind == 'cahn_hilliard':
        ebm_fom = CahnHilliard(**ebm_kwargs)
        num_quad_nodes_list = [2*k for k in degrees] # although k+1 should be sufficient, the newton solver has difficulties for k+1 quadnodes. we therefore default to a higher number of quadnodes.
        num_proj_nodes_list = [2*k for k in degrees]
    
    if only_rom:
        savepath += '_rom'

    print(f'\n\n--- testing energybalance for {kind} system ---')
    if ebm_kwargs:
        print(f'--- settings: {ebm_kwargs} ---')
        
    # set up time interval
    if nt is None: nt = int(T*100)+1
    tt = jnp.linspace(0,T, nt) # t_i = i * T/nt
    
    if with_rom:
        orderlist = ['fom', 'rom']
        fig, (ax_fom, ax_rom) = plt.subplots(ncols=2, figsize=(11,4), layout='constrained')
        fig.get_layout_engine().set(wspace=0.1, h_pad=0.1)
        
    elif only_rom:
        orderlist = ['rom']
        fig, ax_rom = plt.subplots()

    else:
        orderlist = ['fom']
        # set up plot environment
        fig, ax_fom = plt.subplots()
    
    for order in orderlist:
        print(f'\n--- {order = } ---')
        if order == 'rom':
            if kind == 'cahn_hilliard':
                ebm_rom = CahnHilliardReducedOrder(picklepath=picklepath, **ebm_kwargs)
            elif kind == 'doubly_nonlinear_parabolic':
                ebm_rom = DoublyNonlinearParabolicReducedOrder(picklepath=picklepath, **ebm_kwargs) # picklepath to speed up computation
            ebm = ebm_rom
            picklepath_order = f'{picklepath}_rom'
            ax = ax_rom
        else:
            ebm = ebm_fom
            picklepath_order = picklepath
            ax = ax_fom
    
        ax.set_xlabel('time $t$')
        ax.set_ylabel(r'$\errorenergy$')
        ax.set_ylim(1.5e-18, 1.5e-3)
        if kind == 'doubly_nonlinear_parabolic':
            ax.set_ylim(1.5e-21, 1.5e-3)
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
                label=rf'$k = {degree},~ \projnodes = {num_proj_nodes}$' if (order=='fom' or only_rom) else None,
                color=plt.cm.tab20(2*index),
                linewidth=3.0,
                alpha=0.6,
                )
    
            # test influence of projection nodes for cahn hilliard model
            test_projection_influence = False
            add_to_proj_nodes = 0
            step = 0
            max_steps = 0
            
            if kind == 'cahn_hilliard':
                test_projection_influence = True
                add_to_proj_nodes = -1
                max_steps = 1
                
            elif kind == 'toda' and degree in [1,2]:
                test_projection_influence = True
                add_to_proj_nodes = -1
                if degree == 1:
                    max_steps = 2
                else:
                    max_steps = 1
    
            while test_projection_influence and step < max_steps:
                
                num_proj_nodes += add_to_proj_nodes
                
                picklename = f'{picklepath_order}_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}' # needs to be updated
        
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
                    label=rf'$k = {degree},~ \projnodes = {num_proj_nodes}$' if (order=='fom' or only_rom) else None,
                    color=plt.cm.tab20(2*index),
                    linestyle=['--', 'dotted'][step],
                    linewidth=3.0,
                    alpha=0.6,
                    zorder=0,
                    )
                
                step += 1

    if with_rom:
        fig.legend(loc=legend_loc, ncols=len(degrees), framealpha=1.)
    elif only_rom:
        ax_rom.legend(loc=legend_loc)
    else:
        ax_fom.legend(loc=legend_loc)

    # save + show
    if save:
        savepath = savepath + '_energybalance'
        if not with_rom:
            fig.tight_layout()
        fig.savefig(savepath + '.pgf', bbox_inches='tight', pad_inches=0.01) # save as pgf
        fig.savefig(savepath + '.png', bbox_inches='tight', pad_inches=0.01) # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    fig.title = f'relative error in energy balance -- {kind}{ebm_kwargs_string}'
    if not with_rom:
        fig.tight_layout()
    fig.show()

    return





if __name__ == '__main__':
    
    jax.config.update("jax_enable_x64", True) # activate double precision
    
    # plotting
    from helpers.other import mpl_settings
    mpl_settings(fontsize=18)
    
    # set savepath
    SAVEPATH = './results'


    # nonlinear circuit
    energybalance(
        kind='nonlinear_circuit',
        T=5,
        degrees=[1,2,3,4],
        save=True,
        use_pickle=True,
        )
    

    # doubly nonlinear parabolic
    plist = [1.5]
    qlist = [1.5, 3]
    for (p,q) in product(plist, qlist):
        # fom
        energybalance(
            kind='doubly_nonlinear_parabolic',
            T=0.1,
            nt=501,
            degrees=[2,3,4],
            ebm_kwargs={'p': p, 'q': q, 'nx': 50},
            save=True,
            use_pickle=True,
            with_rom=False,
            )

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

    
    
    