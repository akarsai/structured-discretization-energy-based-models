#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file is used to generate the visualizations of
# the hamiltonian of the acdc converter model in the publication
#
#



if __name__ == '__main__':
    
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    
    from timeit import default_timer as timer
    
    from main.time_discretization import projection_method
    from examples.acdc import ACDC
    ebm = ACDC()
    
    import pickle
    import matplotlib.pyplot as plt
    from helpers.other import mpl_settings
    mpl_settings(fontsize=20)
    
    # set savepath
    SAVEPATH = './results'
    
    # simulation settings
    T = 5
    nt = int(T*100)+1
    tt = jnp.linspace(0, T, nt)
    degree = 3
    num_quad_nodes = None
    num_proj_nodes = 2*degree
    
    # set paths
    savepath = f'{SAVEPATH}/figures/acdc'
    picklepath = f'{SAVEPATH}/pickle/acdc'
    picklename = f'{picklepath}_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}'
    
    # set up hamiltonian plots
    fig_ham, ax_ham = plt.subplots()

    # run fom and rom simulations
    # get default control and initial condition
    control = ebm.default_control
    z0 = ebm.initial_condition
    
    # run simulation with projection method
    try: # try to skip also the evaluation
        with open(f'{picklename}.pickle','rb') as f:
            proj_solution = pickle.load(f)['proj_solution']
        print(f'({degree = }, {num_quad_nodes = }, {num_proj_nodes = }, {nt = })\n\tresult was loaded')

    except FileNotFoundError: # evaluation was not done before
        s = timer()
        proj_solution = projection_method(
            ebm=ebm,
            tt=tt,
            z0=z0,
            control=control,
            degree=degree,
            num_quad_nodes=num_quad_nodes,
            num_proj_nodes=num_proj_nodes,
            debug=False,
            )
        e = timer()
        print(f'({degree = }, {num_quad_nodes = }, {num_proj_nodes = }, {nt = })\n\ttook {e-s:.2f} seconds')

        # save file
        with open(f'{picklename}.pickle','wb') as f:
            pickle.dump({'proj_solution': proj_solution},f)
        print(f'\tresult was written')
    
    zz_proj = proj_solution['boundaries'][1]

    zz_proj_1 = zz_proj[:, :ebm.dims[0]]
    zz_proj_2 = zz_proj[:, ebm.dims[0]:ebm.dims[0]+ebm.dims[1]]
    label = r'$\mathcal{H}$'
    ax_ham.plot(tt, ebm.hamiltonian_vmap(zz_proj_1, zz_proj_2), label=label, linewidth=3.0)
    
    # plot hamiltonian
    ax_ham.set_xlabel('time $t$')
    ax_ham.set_ylabel('Hamiltonian')
    ax_ham.legend()
    fig_ham.tight_layout()
    fig_ham.savefig(savepath+'_hamiltonian.pgf')
    fig_ham.savefig(savepath+'_hamiltonian.png')
    print(f'figure saved under savepath {savepath}_hamiltonian (as pgf and png)')
    fig_ham.suptitle('Hamiltonians for AC/DC converter model')
    fig_ham.tight_layout()
    fig_ham.show()