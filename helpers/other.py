#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a helper class to style print output using
# ansi codes and a helper function to prepare matplotlib figures
# for publication.
#


import matplotlib
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import re

def vmap2d(
        function: callable,
        ) -> callable:
    """
    vectorize a function over two dimensions, so that it accepts a 2d array as input

    Args:
        function: callable to vectorize

    Returns:
        2d vectorized callable
    """

    return jax.vmap(jax.vmap(function, in_axes=0, out_axes=0), in_axes=0, out_axes=0)


def dprint(var, format: str = ""):
    """
    function for debug printing
    """
    import inspect
    frame = inspect.currentframe().f_back
    call_line = inspect.getframeinfo(frame).code_context[0]
    var_name = call_line.strip()
    var_name = var_name[var_name.find('(')+1:var_name.rfind(')')]
    # clean up var_name if it contains format parameter
    if format != "" and ',' in var_name: var_name = ','.join(var_name.split(',')[:-1])
    print(f'\n{var_name} =\n{var:{format}}')
    return

def mpl_fontsize(
        fontsize: int = None,
        bigger_axis_labels: bool = True,
        ):
    
    if fontsize is not None:
        plt.rcParams.update({"font.size": fontsize})
        
        # make legend font size smaller
        plt.rcParams.update({
            "legend.fontsize": fontsize - 6,
            })
    
        # bigger axis labels if needed
        if bigger_axis_labels:
            plt.rcParams.update({
                "axes.labelsize": fontsize + 4,
                "axes.titlesize": fontsize + 4,
                })
            
    return
    
def mpl_settings(
        figsize: tuple = (5.5,4),
        backend: str = None,
        latex_font: str = 'computer modern',
        fontsize: int = None,
        bigger_axis_labels: bool = True,
        dpi: int = 500,
        ) -> None:
    """
    sets matplotlib settings for latex

    :return: None
    """

    plt.rcParams['figure.dpi'] = dpi
    # default for paper: (5.5,4)
    plt.rcParams['figure.figsize'] = figsize
    plt.rc('text', usetex=True)
    
    preamble = '\n'.join([
                        r'\usepackage{amsmath,amssymb}',
                        r'\newcommand{\projnodes}{s_{\Pi}}',
                        r'\newcommand{\quadnodes}{s_Q}',
                        r'\newcommand{\error}{\mathcal{E}}',
                        r'\newcommand{\errorenergy}{\error_{\text{energy}}}',
                        r'\newcommand{\errorstate}{\error_{\text{state}}}',
                    ])
    
    plt.rc('text.latex', preamble=preamble)

    plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,      # don't setup fonts from rc parameters
            "pgf.preamble": preamble, # the preamble really need to be defined two times ... i do not know why
            "savefig.transparent": True,
            })
    
    mpl_fontsize(fontsize=fontsize, bigger_axis_labels=bigger_axis_labels)

    if latex_font == 'times':
        plt.rc('font',**{'family':'serif','serif':['Times']})
    elif latex_font == 'computer modern':
        plt.rc('font',**{'family':'serif'})

    plt.rc('axes.formatter', useoffset=False)
    # plt.rcParams['savefig.transparent'] = True

    if backend is not None:
        matplotlib.use(backend)
    if backend == 'macosx':
        plt.rcParams['figure.dpi'] = 140

    return

class style:
    info = '\033[38;5;027m'
    success = '\033[38;5;028m'
    warning = '\033[38;5;208m'
    fail = '\033[38;5;196m'
    #
    bold = '\033[1m'
    underline = '\033[4m'
    italic = '\033[3m'
    end = '\033[0m'

def plot_matrix(A,title=''):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(A,cmap='nipy_spectral')
    fig.colorbar(cax)
    plt.title(title)
    plt.show()

    return

def sparse_blockmatrix(
        A: sparse.BCOO,
        B: sparse.BCOO,
        C: sparse.BCOO,
        D: sparse.BCOO,
        ):
    """
    builds the block matrix
    
    [ A  B ]
    [ C  D ]
    """
    AB = sparse.bcoo_concatenate((A,B), dimension=1)
    CD = sparse.bcoo_concatenate((C,D), dimension=1)
    return sparse.bcoo_concatenate((AB,CD), dimension=0)

if __name__ == "__main__":
    pass