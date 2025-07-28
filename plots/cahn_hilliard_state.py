#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file is used to generate the visualizations of
# the state of the cahn hilliard model in the publication
#
#



if __name__ == '__main__':
    
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    
    from timeit import default_timer as timer
    
    from main.time_discretization import projection_method
    from examples.cahn_hilliard import CahnHilliard, CahnHilliardReducedOrder
    ch_fom = CahnHilliard()
    ch_rom = CahnHilliardReducedOrder()
    
    import pickle
    import matplotlib.pyplot as plt
    from helpers.other import mpl_settings
    mpl_settings(fontsize=20)
    
    # set savepath
    SAVEPATH = './results'
    
    # simulation settings
    T = 1.5
    nt = int(T*100)+1
    tt = jnp.linspace(0, T, nt)
    degree = 3
    num_quad_nodes = 2*degree # degree + 1 leads to difficulties in the newton solver
    num_proj_nodes = 2*degree
    
    # set paths
    savepath_fom = f'{SAVEPATH}/figures/cahn_hilliard'
    savepath_rom = f'{savepath_fom}_rom'
    picklepath = f'{SAVEPATH}/pickle/cahn_hilliard'
    picklename_fom = f'{picklepath}_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}'
    picklename_rom = f'{picklepath}_rom_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}'
    
    # set up hamiltonian plots
    fig_ham, ax_ham = plt.subplots()

    # run fom and rom simulations
    for (ebm, picklename, savepath) in [(ch_fom, picklename_fom, savepath_fom), (ch_rom, picklename_rom, savepath_rom)]:
        
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
        label = r'$\mathcal{H}^h$'
        if ebm.is_rom:
            label = r'$\tilde{\mathcal{H}}$'
        ax_ham.plot(tt, ebm.hamiltonian_vmap(zz_proj_1, zz_proj_2), label=label, linewidth=3.0)
        
        # visualize state
        num_steps = 4
        for step in range(num_steps+1):
            if step == num_steps:
                z_vis = zz_proj[-1, :ebm.dims[0]]
            else:
                z_vis = zz_proj[step*nt//num_steps, :ebm.dims[0]]
            
            if ebm.is_rom:
                z_vis = ebm.V1 @ z_vis
            
            ebm.space.visualize_coefficient_vector(
                z_vis,
                title=f'$v({step*T/num_steps:.1f})$',
                # title=f'$v({step*T/num_steps:.1f})$, projection method, $\\varepsilon={ebm.eps:.2f}, \\sigma={ebm.sigma:.2f}$',
                vmin=.35 if step in [0,1,2,3] else -1.1,
                vmax=.65 if step in [0,1,2,3] else 1.1,
                savepath=savepath+f'_{step}',
                # colorbar_label=r'$v$'
                )

    # plot hamiltonians
    ax_ham.set_xlabel('time $t$')
    ax_ham.set_ylabel('Hamiltonian')
    ax_ham.legend()
    fig_ham.tight_layout()
    fig_ham.savefig(savepath_fom+'_hamiltonian.pgf')
    fig_ham.savefig(savepath_fom+'_hamiltonian.png')
    print(f'figure saved under savepath {savepath_fom}_hamiltonian (as pgf and png)')
    fig_ham.suptitle('Hamiltonians for Cahn--Hilliard model')
    fig_ham.tight_layout()
    fig_ham.show()