import numpy as np
from scipy.ndimage import geometric_transform
import h5py

# Input parameters
outfile = 'temp'
materials_file = 'myMaterials.txt'

lenX = 2.0
lenY = 2.0
lenZ = 0.05

dx = 0.005
dy = 0.005
dz = 0.05

nx = int(lenX/dx)
nz = int(lenZ/dz)
ny = int(lenY/dy)

lenT = 100e-9

# layers
max_layers = 3 # not including free space
layer_min_thickness = 30 # in samples
layer_max_thickness = ny//max_layers
free_space_layer_depth_samples = 40 # this is the free space

# source-receiver
receiver_depth = lenY - free_space_layer_depth_samples*dy
source_receiver_increment = 0.02

# folding and dipping
slope_limit = 0.25
Ngaussians = 2

# pipe inclusions
Npipes = 2
max_radius = 25
min_radius = 10
pipe_diameter = 3

def genGuassians2D(Ngaussians, X):
    '''
    Generate a sum of a series of Gaussians
    '''
    Xf = X.astype('float')
    sumGaussians = np.zeros_like(Xf)
    std = 0.5*X.shape[0] * np.random.rand(Ngaussians) # stadnard deviation squared
    b = np.random.randn(Ngaussians) * std
    c = X.shape[0] * np.random.rand(Ngaussians)
    # loop over number of Guassians
    for i in range(Ngaussians):
        sumGaussians += b[i]*np.exp(-np.power(Xf-c[i],2)/(2*np.power(std[i],2)))
    return sumGaussians



def create_circular_mask(h, w, center=None, radius=None):
    '''
    generate a circular mask - from https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    '''
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


# parameters for .in file
with open(outfile+'.in','w') as f:
    f.write('#title: B-scan from a metal cylinder buried in a dielectric half-space\n')
    f.write(f'#domain: {lenX} {lenY} {lenZ}\n')
    f.write(f'#dx_dy_dz: {dx} {dy} {dz}\n')
    f.write(f'#rx: {dx+0.35} {receiver_depth} 0\n')
    f.write(f'#time_window: {lenT}\n')
    f.write(f'#waveform: ricker 1 0.5e9 my_ricker\n')
    f.write(f'#hertzian_dipole: z {dx+0.2} {receiver_depth} 0 my_ricker\n')
    f.write(f'#src_steps: {source_receiver_increment} 0 0\n')
    f.write(f'#rx_steps: {source_receiver_increment} 0 0\n')
    f.write(f'#geometry_objects_read: 0.0 0.0 0.0 {outfile}.hdf5 {materials_file}\n')

# build array of integers
arr = 3*np.ones((nx,ny,nz), dtype=np.int16)
X, Y = np.ogrid[0:nx, 0:ny]
# make first 10 layers free space - which must be first material in file
# # put in new layer
# add new layers until total number of layers is reached or number remaining height is less than 10 samples 
current_layer = 3
current_minimum = free_space_layer_depth_samples
while True:
    if current_layer-2 >= max_layers:
        arr[:,current_minimum:,:] = current_layer
        break
    layer_thickness = np.random.randint(layer_min_thickness,layer_max_thickness)
    if current_minimum+layer_thickness+10>ny:
        arr[:,current_minimum:,:] = current_layer
        break
    else:
        arr[:,current_minimum:current_minimum+layer_thickness,:] = current_layer
    current_layer += 1
    current_minimum = current_minimum + layer_thickness

# build matrix of shift values
a1 = 2*slope_limit*np.random.rand() - slope_limit # limit slope to +-0.25
# define centre trace to not move
c1 = -a1 * int(nx/2)
# c1 = -a1 * int(nx/2) -b1 * int(nz/2) # 3D version
# first transform - tilting
S1 = a1 * X + c1
# S1 = a1 * X + b1 * Z + c1 # 3D version

# define folding
# S2 = 1.5*Y/np.max(Y) * genGuassians3D(Ngaussians, X, Z) # 3D version
S2 = 1.5*Y/np.max(Y) * genGuassians2D(Ngaussians, X)

Ynew = Y+S1+S2

# define shift function
def shift_function2D(input_coords):
    return (input_coords[0], Ynew[input_coords[0], input_coords[1]], 0)
# shift_function = lambda input_coords: [input_coords[0], Ynew[input_coords[0], input_coords[1]]]

arr = geometric_transform(arr, shift_function2D, mode='nearest')

# re-insert free space back in
arr[:,:free_space_layer_depth_samples,:] = 0

# add pipes
# pipe
curr_mask = np.zeros_like(arr, dtype=bool)
curr_mask_contents = np.zeros_like(arr, dtype=bool)
completed_pipes = 0
while completed_pipes < Npipes:
    radius = max_radius*np.random.rand()+min_radius # in samples 
    # top of pipe should not be in the first 5 layers. from radius + 5 to ny-7+radius -> total range (ny-7+radius) - (radius+5) = ny-12
    center = [np.random.rand()*(ny-12)+(radius+5), nx * np.random.rand()] 
    mask = create_circular_mask(nx, ny, center, radius)
    if np.max(curr_mask.astype(int)[:,:,0]+mask)>1: # check if pipe masks overlap, if they do then don't use the current pipe mask
        continue
    else:
        curr_mask[:,:,0] += mask
        curr_mask_contents[:,:,0] += create_circular_mask(nx, ny, center, radius - pipe_diameter)
        completed_pipes += 1

arr[curr_mask] = 0 # assume pec is first material in material file
# pipe contents
arr[curr_mask_contents] = 1 # assume water is second material in material file

# write array of integers to file
with h5py.File(outfile+".hdf5", "w") as f:
    arr = arr[:,::-1,:]
    dset = f.create_dataset(data=arr, name="/data", dtype=np.int16)
    dest = f.attrs.create(data=(dx,dy,dz), name="dx_dy_dz", dtype='f')
