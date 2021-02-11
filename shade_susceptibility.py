import numpy as np
import os

def shade_susceptibility(im,labmap,label_path,mag=4):
    labels = load_array_if_path(label_path)

    x = np.linspace(-0.5,0.5,num=im.shape[0])
    y = np.linspace(-0.5,0.5,num=im.shape[1])
    z = np.linspace(-0.5,0.5,num=im.shape[2])

    [X,Y,Z] = np.meshgrid(x,y,z)

    grad_map = np.zeros(im.shape)
    for i in labels:
        r1 = np.random.uniform()
        theta = np.arccos(1-(2*r1))
        r2 = np.random.uniform()
        phi = 2*np.pi*r2
        mag = np.random.uniform(0,mag)

        x = mag*np.cos(phi)*np.sin(theta)
        y = mag*np.sin(phi)*np.sin(theta)
        z = mag*np.cos(theta)

        gradient_field = X*x+Y*y+Z*z
        grad_map[labmap==i] = gradient_field[labmap==i]
    
    im = im+grad_map
    return im

def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var
