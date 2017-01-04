import numpy as np
import h5py
import skimage.io
from skimage.transform import resize
from scipy.ndimage import zoom


def toHDF5(data_path, input_file, output_file):
    # fixed size to all images
    width = 1360
    height = 800
    with open(input_file, 'r' ) as T :
        lines = T.readlines()
    # If you do not have enough memory split data into
    # multiple batches and generate multiple separate h5 files
    X = np.zeros((len(lines), 3, height, width), dtype='u4')
    y = np.zeros((len(lines), 4), dtype='u4')
    for i, l in enumerate(lines):
        sp = l.split(';')
        img = load_image(data_path + sp[0])
        # resize to fixed size
        # img = resize_image(img, (size, size, 3))
        # Note that the transformation should take img from size-by-size-by-3 and transpose it to 3-by-size-by-size
        # for example
        # RGB->BGR
        img = img[:, :, ::-1]  # RGB => BGR
        img = img.transpose((2, 0, 1))  # in Channel x Height x Width order (switch from H x W x C)
        show_image(img)
        X[i] = img
        y[i, 0] = float(sp[1])
        y[i, 1] = float(sp[2])
        y[i, 2] = float(sp[3])
        y[i, 3] = float(sp[4])
    #todo: update location of this
    with h5py.File('train.h5','w') as H:
        # note the name X given to the dataset!
        H.create_dataset('X', data=X)
        # note the name y given to the dataset!
        H.create_dataset('y', data=y)
    with open(output_file,'w') as L:
        # list all h5 files you are going to use
        L.write(output_file)


def load_image(input_file, color=True):
    img = skimage.img_as_float(skimage.io.imread(input_file, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)


def show_image(img):
    from skimage.viewer import ImageViewer
    img = img.transpose((1, 2, 0))
    img = img[:, :, ::-1]
    viewer = ImageViewer(img.reshape(800, 1360, 3))
    viewer.show()

toHDF5('/Users/silvianac/Downloads/TrainIJCNN2013/', '/Users/silvianac/Downloads/TrainIJCNN2013/gt_mini.txt',
       'out_test.txt')
