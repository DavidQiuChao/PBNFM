import numpy as np
from PIL import Image


def gray_ps(rgb):
    return np.power(np.power(rgb[:,:,0],2.2)*0.2973\
        +np.power(rgb[:,:,1],2.2)*0.6274\
        +np.power(rgb[:,:,2],2.2)*0.0753,1/2.2)+1e-7


def HDR(x, curve_ratio):
    gray_scale = np.expand_dims(gray_ps(x),axis=-1)
    gray_scale_new = np.power(gray_scale,curve_ratio)
    return np.minimum(x*gray_scale_new/gray_scale, 1.0)


def show_bbf_array(im,wb,sdir,max_level,\
        black_level,height,width,tp='sony'):
    im = im.reshape(height,width)
    im = np.maximum(im-black_level,0)/(max_level-black_level)
    im = np.expand_dims(im,axis=2)
    H = im.shape[0]
    W = im.shape[1]
    if tp=='T1pro':
        out = np.concatenate((
            im[1:H:2,0:W:2,:],
            (im[0:H:2,0:W:2,:]+im[1:H:2,1:W:2,:])/2.0,
            im[0:H:2,1:W:2,:]),axis=2)
        wb = wb[:3]
        wb = wb/wb[1]
        out = np.minimum(out*wb,1.0)
    elif tp=='sony':
        out = np.concatenate((
            im[0:H:2,0:W:2,:],
            (im[0:H:2,1:W:2,:]+im[1:H:2,0:W:2,:])/2.0,
            im[1:H:2,1:W:2,:]),axis=2)
        wb = wb[:3]
        wb = wb/wb[1]
        out = np.minimum(out*wb,1.0)
    else:
        return
    out = np.minimum(out*0.2/out[:,:,1].mean(),1.0)
    out = HDR(out,0.35)
    Image.fromarray(np.uint8(out*255)).save(sdir)
