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


import shutil
from matplotlib.text import Text

_CUSTOM_MATH_MACROS_0 = {
    r'\projnodes': r's_{\Pi}',
    r'\quadnodes': r's_Q',
    r'\errorenergyabs': r'\mathcal{E}_{\mathrm{energy}}^{\mathrm{abs}}',
    r'\errorenergy': r'\mathcal{E}_{\mathrm{energy}}',
    r'\errorstatenodal': r'\mathcal{E}_{\mathrm{state,nodal}}',
    r'\errorstate': r'\mathcal{E}_{\mathrm{state}}',
    r'\error': r'\mathcal{E}',
    r'\hamc': r'\hat{\mathcal{H}}',
    r'\zc': r'\hat{z}',
    r'\zekf': r'\overline{z}',
}

_CUSTOM_MATH_MACROS_1 = {
    r'\norm': r'\Vert #1 \Vert',
}

_MATH_MACRO_FALLBACK_INSTALLED = False


def _tex_available() -> bool:
    return shutil.which("latex") is not None or shutil.which("pdflatex") is not None


def _replace_one_arg_macro(text: str, macro: str, replacement: str) -> str:
    pos = 0
    out = []

    while True:
        i = text.find(macro, pos)
        if i == -1:
            out.append(text[pos:])
            break

        out.append(text[pos:i])
        j = i + len(macro)

        while j < len(text) and text[j].isspace():
            j += 1

        if j >= len(text) or text[j] != "{":
            out.append(macro)
            pos = i + len(macro)
            continue

        depth = 0
        k = j
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1

        if k >= len(text):
            out.append(text[i:])
            break

        arg = text[j + 1:k]
        arg = _expand_math_macros(arg)
        out.append(replacement.replace("#1", arg))
        pos = k + 1

    return "".join(out)


def _expand_math_macros(text):
    if not isinstance(text, str):
        return text

    out = text
    for _ in range(8):
        old = out

        for macro, replacement in _CUSTOM_MATH_MACROS_1.items():
            out = _replace_one_arg_macro(out, macro, replacement)

        for macro in sorted(_CUSTOM_MATH_MACROS_0, key=len, reverse=True):
            out = out.replace(macro, _CUSTOM_MATH_MACROS_0[macro])

        if out == old:
            break

    return out


def _install_math_macro_fallback() -> None:
    global _MATH_MACRO_FALLBACK_INSTALLED

    if _MATH_MACRO_FALLBACK_INSTALLED:
        return

    original_set_text = Text.set_text

    def _set_text_with_macro_expansion(self, s):
        return original_set_text(self, _expand_math_macros(s))

    Text.set_text = _set_text_with_macro_expansion
    _MATH_MACRO_FALLBACK_INSTALLED = True


def mpl_fontsize(
    fontsize: int = None,
    bigger_axis_labels: bool = True,
):
    if fontsize is None:
        return

    plt.rcParams.update({
        "font.size": fontsize,
        "legend.fontsize": max(fontsize - 6, 1),
    })

    if bigger_axis_labels:
        plt.rcParams.update({
            "axes.labelsize": fontsize + 2,
            "axes.titlesize": fontsize + 2,
        })

    return


def mpl_settings(
    figsize: tuple = (5.5, 4),
    backend: str = None,
    latex_font: str = 'computer modern',
    fontsize: int = None,
    bigger_axis_labels: bool = True,
    dpi: int = 500,
) -> None:
    """
    Sets matplotlib settings for TeX if available, otherwise falls back
    to matplotlib mathtext with custom macro expansion.
    """

    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['savefig.transparent'] = True
    plt.rc('axes.formatter', useoffset=False)

    # Dynamically build the preamble from the custom macro dictionaries
    preamble_lines = [r'\usepackage{amsmath,amssymb}']

    # Add macros with no arguments
    for macro, replacement in _CUSTOM_MATH_MACROS_0.items():
        preamble_lines.append(rf'\newcommand{{{macro}}}{{{replacement}}}')

    # Add macros with 1 argument
    for macro, replacement in _CUSTOM_MATH_MACROS_1.items():
        preamble_lines.append(rf'\newcommand{{{macro}}}[1]{{{replacement}}}')

    preamble = '\n'.join(preamble_lines)

    use_tex = _tex_available()

    if use_tex:
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=preamble)
        plt.rcParams.update(
            {
                'pgf.texsystem': 'pdflatex',
                'pgf.rcfonts': False,
                'pgf.preamble': preamble,
            }
        )
    else:
        plt.rc('text', usetex=False)
        plt.rcParams.update(
            {
                'text.parse_math': True,
                'pgf.preamble': '',
            }
        )
        _install_math_macro_fallback()

    if latex_font == 'times':
        plt.rcParams.update(
            {
                'font.family': 'serif',
                'font.serif': [
                    'Times New Roman',
                    'Times',
                    'Nimbus Roman No9 L',
                    'DejaVu Serif',
                ],
                'mathtext.fontset': 'stix',
            }
        )
    elif latex_font == 'computer modern':
        plt.rcParams.update(
            {
                'font.family': 'serif',
                'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
                'mathtext.fontset': 'cm',
            }
        )

    mpl_fontsize(fontsize=fontsize, bigger_axis_labels=bigger_axis_labels)

    if backend is not None:
        matplotlib.use(backend)
        if backend == 'macosx':
            plt.rcParams['figure.dpi'] = 140

    return


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

def scientific_notation_tex_code(number: float) -> str:

    number_string = f'{number:.2e}'

    # regex pattern to match scientific notation
    pattern = r"([+-]?\d+\.\d+)e([+-]?\d+)"

    # replacement format
    replacement = r"\1 \\cdot 10^{\2}"

    # Perform the replacement
    output = re.sub(pattern, replacement, number_string)

    return output

def generate_eoc_table_tex_code(
        tau_list: jnp.ndarray,
        k_list: jnp.ndarray,
        error_list: jnp.ndarray,
        with_average: bool = True,
        error_threshold: float = 1e-14,
        ) -> str:
    """
    generates latex code for an eoc table.

    the formula for the experimental order of convergence (eoc) is

    eoc = log(error_2/error_1) / log(tau_2/tau_1)

    :param tau_list: jnp.ndarray, list of tau values (time discretization step sizes)
    :param k_list: jnp.ndarray, list of k values (polynomial degrees)
    :param error_list: jnp.ndarray, list of errors, shape (tau, k) (rows: tau, columns: k)
    :param with_average: bool, if True, an average line is added to the table
    :param error_threshold: float, if error is below this threshold, eoc is not shown in the table and does not count towards the average
    :param E_subscript: string, subscript $E_{error_type}$ in the table
    :return: str, latex code
    """

    # first, prepare eoc list
    log_error_div = jnp.log(error_list[1:,:]/error_list[:-1,:])
    log_tau_div = jnp.log(tau_list[1:]/tau_list[:-1])
    eoc_list = jnp.einsum('tk,t->tk', log_error_div, 1/log_tau_div)
    eoc_list = jnp.concatenate((-jnp.inf*jnp.ones((1,k_list.shape[0])), eoc_list), axis=0) # put -1 in the first row where we have no eoc

    # first line of tex code, defining the number of columns
    latex = '\n\\begin{tabular}{|c'
    for k in k_list:
        latex += '|cc'
    latex += '|}\n\\hline\n'

    # second line of tex code, defining the header
    latex += '    \\multirow{2}{\\widthof{$\\tau$}}{$\\tau$}'
    for k_index, k in enumerate(k_list):
        latex += ' & \\multicolumn{2}{c|}{$k='+str(k)+'$}'
    latex += '\\\\\n    '

    # third line of tex code
    for k in k_list:
        latex += f' & $\\errorstate(\\tau)$ & \\eoc'

    # actual content
    latex += '\\\\ \\hline'
    for tau_index, tau in enumerate(tau_list):
        latex += f'\n    ${scientific_notation_tex_code(tau)}$'
        for k_index, k in enumerate(k_list):
            if eoc_list[tau_index, k_index] < 0 or error_list[tau_index-1, k_index] < error_threshold: # if eoc is not available / sensible
                eoc_string = '-'
            else: # if eoc is available, format properly
                eoc_string = f'${eoc_list[tau_index, k_index]:.2f}$'
            latex += f' & ${scientific_notation_tex_code(error_list[tau_index, k_index])}$ & {eoc_string}'
        latex += ' \\\\'

    # average line - old version, counts everything
    # if with_average:
    #     latex += '\n\\hline\n     '
    #     for k_index, k in enumerate(k_list):
    #         eoc_avg = jnp.mean(eoc_list[1:,k_index])
    #         latex += f' & & $\\hspace{{-8pt}}\\diameter {eoc_avg:.2f}$'

    # average line
    if with_average:
        latex += '\n\\hline\n     '
        for k_index, k in enumerate(k_list):
            eoc_avg_list = []
            for tau_index, tau in enumerate(tau_list):
                if error_list[tau_index-1, k_index] > error_threshold and eoc_list[tau_index, k_index] != -jnp.inf:
                    eoc_avg_list.append(eoc_list[tau_index, k_index])
            eoc_avg = jnp.mean(jnp.array(eoc_avg_list))
            latex += f' & & $\\hspace{{-11pt}}\\diameter\\; \\textbf{{{eoc_avg:.2f}}}$'
    latex += '\\\\ \\hline\n'

    # last line
    latex += '\\end{tabular}\n'

    # print(latex)
    return latex

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
