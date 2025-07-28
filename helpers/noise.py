#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a noise generating functions. these are used
# for the initial condition of the Cahn--Hilliard equation.
#
#

import numpy as np
from scipy import interpolate

def generate_perlin_noise_2d(
        shape: tuple[int, int],
        res: tuple[int, int] = (1,1),
        tileable: tuple[bool, bool] = (False, False),
        seed: int = 0,
        ):
    
    # helper functions
    def fade(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    def lerp(a, b, x):
        return a + x*(b - a)
    
    # random number generator
    rng = np.random.default_rng(seed=seed)
    
    # Generate grid coordinates
    delta = (res[0]/shape[0], res[1]/shape[1])
    d = (shape[0]//res[0], shape[1]//res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    
    # Generate random gradients
    angles = 2*np.pi*rng.random((res[0]+1, res[1]+1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    
    # Tile gradients if required
    if tileable[0]: gradients[-1,:] = gradients[0,:]
    if tileable[1]: gradients[:,-1] = gradients[:,0]
    
    # Calculate dot products for grid corners
    g00 = gradients[:-1,:-1].repeat(d[0],0).repeat(d[1],1)
    g10 = gradients[1:,:-1].repeat(d[0],0).repeat(d[1],1)
    g01 = gradients[:-1,1:].repeat(d[0],0).repeat(d[1],1)
    g11 = gradients[1:,1:].repeat(d[0],0).repeat(d[1],1)
    
    # Calculate noise components
    u = fade(grid[:,:,0])
    v = fade(grid[:,:,1])
    
    n00 = np.sum(g00 * grid, axis=2)
    n10 = np.sum(g10 * (grid - np.array([1,0])), axis=2)
    n01 = np.sum(g01 * (grid - np.array([0,1])), axis=2)
    n11 = np.sum(g11 * (grid - np.array([1,1])), axis=2)
    
    # Interpolate noise values
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return (lerp(x1, x2, v) + 1) / 2

def generate_fractal_noise_2d(
        shape: tuple[int, int],
        res: tuple[int, int] = (64,64),
        octaves: int = 4,
        persistence: float = 0.9,
        lacunarity: float = 2.0,
        tileable: tuple[bool, bool] = (False, False),
        seed: int = 0,
        ):
    
    noise = np.zeros(shape)
    frequency = 1.0
    amplitude = 1.0
    
    for _ in range(octaves):
        # Calculate current resolution for this octave
        current_res = (
            max(1, int(res[0] / frequency)),
            max(1, int(res[1] / frequency))
        )
        
        # Generate and accumulate noise layer
        noise += amplitude * generate_perlin_noise_2d(
            shape=shape,
            res=current_res,
            tileable=tileable,
            seed=seed,
        )
        
        # Update frequency and amplitude for next octave
        frequency *= lacunarity
        amplitude *= persistence
        seed += 1
    
    # Normalize to [-1, 1] range
    return noise / (1 - persistence**octaves) * (1 - persistence)

# use interpolation to evaluate fractal noise on arbitrary xy coordiantes
def fractal_noise_on_points(
        xy: np.ndarray,
        x_range: tuple[float, float] = (0.0,1.0),
        y_range: tuple[float, float] = (0.0,1.0),
        fractal_noise_kwargs: dict = None,
        seed: int = 0,
        ):
    
    if fractal_noise_kwargs is None:
        fractal_noise_kwargs = {
            'shape': (512, 512),
            'seed': seed,
            }
    
    # compute fractal noise on grid
    noise = generate_fractal_noise_2d(**fractal_noise_kwargs)

    # arange x and y coords
    nx, ny = noise.shape
    x_coords = np.linspace(x_range[0], x_range[1], nx)
    y_coords = np.linspace(y_range[0], y_range[1], ny)
    
    # create interpolator using RegularGridInterpolator
    interp = interpolate.RegularGridInterpolator(
        (x_coords, y_coords),
        noise,
        method='cubic',
    )
    
    return interp(xy)
    
    
    

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from helpers.other import mpl_settings
    mpl_settings()
    
    npoints = 100
    x = np.linspace(0,1,npoints)
    y = np.linspace(0,1,npoints)
    X, Y = np.meshgrid(x,y, indexing='ij')
    xy = np.stack([X.ravel(), Y.ravel()]).T
    noise = fractal_noise_on_points(xy).reshape((npoints,npoints))
    
    plt.figure()
    plt.imshow(noise, cmap='Spectral', interpolation='lanczos')
    plt.colorbar()
    plt.title('2D fractal noise from interpolation')
    plt.tight_layout()
    plt.show()
    