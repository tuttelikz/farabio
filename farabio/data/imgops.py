import os
import random
import skimage
import PIL
import numpy as np
from skimage import io
from PIL import Image
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt


PIL.Image.MAX_IMAGE_PIXELS = 933120000

UNITS_MAPPING = [
    (1 << 50, ' PB'),
    (1 << 40, ' TB'),
    (1 << 30, ' GB'),
    (1 << 20, ' MB'),
    (1 << 10, ' KB'),
    (1, (' byte', ' bytes')),
]


def pretty_size(bytes, units=UNITS_MAPPING):
    for factor, suffix in units:
        if bytes >= factor:
            break
    amount = int(bytes / factor)

    if isinstance(suffix, tuple):
        singular, multiple = suffix
        if amount == 1:
            suffix = singular
        else:
            suffix = multiple
    return str(amount) + suffix


class ImgOps:
    """Creates instance of ImgOps class

    Attributes
    ----------
    imgpath : str
        image path of interest
    fsize : str
        file size in human readable format
    img : numpy.array
        image
    h : int
        height
    w : int
        width
    ch : int
        channels
    pmax : int | float
        max pixel value
    pmin : int | float
        min pixel value
    img_r : numpy.array
        red channel image
    img_g : numpy.array
        green channel image
    img_b : numpy.array
        blue channel image

    Methods
    -------
    get_date(self)
        Returns created date and hour info
    print_imginfo(self)
        Prints image information
    slice_img(img, slices, orien)
        Slices image into pieces
    pad_img(self, droi, simg=None)
        Pads image into predefined ROI size
    approx_bcg(self, channel='blue')
        Approximates background image
    blend_img(self, ref_, overlap=0.2, ratio=0.5)
        Blends two images taking the overage of overlap area
    mask_img(self, bw)
        Masks image
    profile_img(self, pt1, pt2)
        Plots image x/y profile between two coordinates
    """

    def __init__(self, imgpath):
        self.imgpath = imgpath
        self.fsize = pretty_size(os.path.getsize(self.imgpath))

        self.img = skimage.io.imread(self.imgpath, plugin='pil')

        self.h = self.img.shape[0]
        self.w = self.img.shape[1]
        self.ch = 1
        if len(self.img.shape) > 2:
            self.ch = self.img.shape[2]

        self.size = (self.h, self.w, self.ch)
        self.pmax = self.img.max()
        self.pmin = self.img.min()

        self.gray = self.img
        if self.ch == 3:
            self.img_r = self.img[:, :, 0]
            self.img_g = self.img[:, :, 1]
            self.img_b = self.img[:, :, 2]
            self.gray = 255*skimage.color.rgb2gray(self.img)

    def get_date(self):
        """Get created date and hour info

        Returns
        -------
        str
            Date and Hour, ex: 200820_191645

        Examples
        ----------
        >>> ImgOps(img_path).get_date()
        """
        img_exif = Image.open(self.imgpath)._getexif()

        if img_exif:
            img_timestamp = img_exif[36868]
            img_day = img_timestamp.split()[0][2:].replace(':', '')
            img_hour = img_timestamp.split()[1].replace(':', '')
            img_date = '_'.join([img_day, img_hour])
            self.img_date = img_date
        else:
            img_date = ""

        return img_date

    def print_imginfo(self):
        """Prints information pm image

        Returns
        -------
        info
            Path, Shape, Intensity range and file size

        Examples
        ----------
        >>> ImgOps(img_path).print_imginfo()
        """

        print("Path:", self.imgpath)
        print("Shape:", self.img.shape)
        print("Intensity Range: [", self.pmin, self.pmax, "]")
        print("File size:", self.fsize)

    def slice_img(self, slices, orien):
        """Slices image into pieces

        Parameters
        ----------
        img : numpy.array
            image
        slices : int
            number of slices
        orien : str
            orientation to cut. 'x': vertical, 'y': horizontal

        Returns
        -------
        img_slices : list of numpy.array
            image slices
        img_slices_info : list of tuples
            represents sliced image dimensions

        Examples
        ----------
        >>> ImgOps(img_path).slice_img(slices=2,orien='x')
        """

        if len(self.img.shape) == 2:
            (h, w) = np.shape(self.img)
        if len(self.img.shape) == 3:
            (h, w, ch) = np.shape(self.img)

        img_slices = []
        if orien == 'y':
            for i in range(slices):
                slice_start = 1 if i == 0 else (i*h) // slices
                slice_end = ((i+1)*h) // slices

                if ch == 1:
                    img_slice = self.img[slice_start:slice_end, :]
                if ch == 3:
                    img_slice = self.img[slice_start:slice_end, :, :]

                img_slices.append(img_slice)
        if orien == 'x':
            for i in range(slices):
                slice_start = 1 if i == 0 else (i*w) // slices
                slice_end = ((i+1)*w) // slices

                img_slice = self.img[:, slice_start:slice_end,
                                     :] if ch == 3 else self.img[:, slice_start:slice_end]

                img_slices.append(img_slice)

        img_slices_info = [np.shape(img_slices[i])
                           for i in range(len(img_slices))]

        return img_slices, img_slices_info

    def pad_img(self, droi, simg=None):
        """Pads image into predefined ROI size

        Parameters
        ----------
        img : numpy.array
            image
        droi : tuple
            desired ROI shape. Ex: (1024, 1024) for gray, (1024, 1024, 3) for RGB
        simg : tuple
            optional. custom definitions for object of interest

        Returns
        ---------
        img_pad : numpy.array
            padded image

        Examples
        ----------
        >>> ImgOps(img_path).pad_img((1024, 1024, 3))
        """

        assert type(droi) == tuple, "param at index 1 must be tuple"

        if simg is None:
            simg = np.shape(self.img)

        dh = droi[0] - simg[0]
        dw = droi[1] - simg[1]

        pad_h = dh // 2
        pad_w = dw // 2

        if len(simg) > 2:
            droi = (*droi, simg[-1])

        img_pad = np.zeros(droi, dtype=self.img.dtype)

        if len(self.img.shape) == 2:
            img_pad = np.zeros(droi, dtype=self.img.dtype)
            img_pad[pad_h:(pad_h+simg[0]), pad_w:(pad_w+simg[-1])] = self.img
        if len(self.img.shape) == 3:
            img_pad[pad_h:(pad_h+simg[0]), pad_w:(pad_w+simg[1]),
                    0] = self.img[:, :, 0]
            img_pad[pad_h:(pad_h+simg[0]), pad_w:(pad_w+simg[1]),
                    1] = self.img[:, :, 1]
            img_pad[pad_h:(pad_h+simg[0]), pad_w:(pad_w+simg[1]),
                    2] = self.img[:, :, 2]

        return img_pad

    def approx_bcg(self, channel='blue'):
        """Approximates background image

        Parameters
        ----------
        channel : str
            colour channel

        Returns
        -------
        img_pad : numpy.array
            approximate background image

        Examples
        ----------
        >>> ImgOps(img_path).approx_bcg(channel='blue')
        """

        min_r, min_g, min_b = [np.min(self.img[:, :, i]) for i in range(0, 3)]
        max_r, max_g, max_b = [np.max(self.img[:, :, i]) for i in range(0, 3)]

        self.img_bcg = np.zeros((self.h, self.w), dtype='uint8')

        if channel == 'red':
            self.img_bcg = np.zeros((self.h, self.w), dtype='uint8')

            for x in range(min_r, max_r):
                img_hull = convex_hull_image(self.img_r > x)
                self.img_bcg[img_hull] = x

        if channel == 'green':
            self.img_bcg = np.zeros((self.h, self.w), dtype='uint8')

            for y in range(min_g, max_g):
                img_hull = convex_hull_image(self.img_r > y)
                self.img_bcg[img_hull] = y

        if channel == 'blue':
            self.img_bcg = np.zeros((self.h, self.w), dtype='uint8')

            for z in range(min_b, max_b):
                img_hull = convex_hull_image(self.img_b > z)
                self.img_bcg[img_hull] = z

        return self.img_bcg

    def blend_img(self, ref_, overlap=0.2, ratio=0.5):
        """Blends two images taking the average of overlap area

        Parameters
        ----------
        ref_ : numpy.array
            image to blend with
        overlap : float
            opt. ratio of overlap area
        ratio : float
            opt. blending ratio

        Returns
        -------
        img_blend : numpy.array
            blended image

        Examples
        ----------
        >>> img_ref = ImgOps('./imgtoblend.png').img
        >>> ImgOps(img_path).blend_img(img_ref, overlap=0.3, ratio=0.5)
        """
        assert self.img.shape == ref_.shape, "Shape must be same with reference image"
        h, w, ch = self.h, self.w, self.ch

        img_blend = np.zeros((h, round((2 - overlap)*w), ch))

        img_blend[:, 0:round((1-overlap)*w), :] = self.img[:,
                                                           0:round((1-overlap)*w), :]
        img_blend[:, w:round((2-overlap)*w), :] = ref_[:,
                                                       round(overlap*w):w, :]
        img_blend[:, round((1-overlap)*w):w, :] = ratio * self.img[:, round(
            (1-overlap)*w):w, :] + (1-ratio) * ref_[:, 0:round(overlap*w), :]

        img_blend = 255 * img_blend / np.amax(img_blend)
        img_blend = img_blend.astype(np.int)

        return img_blend

    @staticmethod
    def blend_imgs(img1, img2, overlap=0.2, ratio=0.5):
        """
        This function blends two images taking the overage of overlap area

        Parameters
        ----------
        img1 : numpy.array
            First image to blend
        img2 : numpy.array
            Second image to blend
        overlap : float
            optional. ratio of overlap area
        ratio : float
            optional. blending ratio

        Returns
        -------
        img_blend : numpy.array
            blended image

        Examples
        ----------
        >>> ImgOps().blend_imgs("img1.jpg","img2.jpg",ratio=0.3)
        """
        h, w, ch = img1.shape

        img_blend = np.zeros((h, round((2 - overlap)*w), ch))

        img_blend[:, 0:round((1-overlap)*w), :] = img1[:,
                                                       0:round((1-overlap)*w), :]
        img_blend[:, w:round((2-overlap)*w), :] = img2[:,
                                                       round(overlap*w):w, :]
        img_blend[:, round((1-overlap)*w):w, :] = ratio * img1[:, round(
            (1-overlap)*w):w, :] + (1-ratio) * img2[:, 0:round(overlap*w), :]

        img_blend = 255 * img_blend / np.amax(img_blend)
        img_blend = img_blend.astype(np.int)

        return img_blend

    def mask_img(self, bw):
        """Masks image

        Parameters
        ----------
        bw : numpy.array
            Binary mask

        Returns
        -------
        img_ov : numpy.array
            Overlay image

        Examples
        ----------
        >>> img_mask = ImgOps('./imgmask.png').img
        >>> ImgOps(img_path).mask_img(img_mask)
        """

        img_ov = self.img
        img_ov[:, :, 0][bw == 0] = 0
        img_ov[:, :, 1][bw == 0] = 0
        img_ov[:, :, 2][bw == 0] = 0

        return img_ov

    @staticmethod
    def random_flip(img, y_random=False, x_random=False,
                    return_param=False, copy=False):
        """Randomly flip an image in vertical or horizontal direction.

        Args:
            img (~numpy.ndarray): An array that gets flipped. This is in CHW format.
            y_random (bool): Randomly flip in vertical direction.
            x_random (bool): Randomly flip in horizontal direction.
            return_param (bool): Returns information of flip.
            copy (bool): If False, a view of :obj:`img` will be returned.

        Returns:
            ~numpy.ndarray or (~numpy.ndarray, dict):
            If :obj:`return_param = False`,
            returns an array :obj:`out_img` that is the result of flipping.
            If :obj:`return_param = True`,
            returns a tuple whose elements are :obj:`out_img, param`.
            :obj:`param` is a dictionary of intermediate parameters whose
            contents are listed below with key, value-type and the description
            of the value.
            * **y_flip** (*bool*): Whether the image was flipped in the\
                vertical direction or not.
            * **x_flip** (*bool*): Whether the image was flipped in the\
                horizontal direction or not.

        """
        y_flip, x_flip = False, False
        if y_random:
            y_flip = random.choice([True, False])
        if x_random:
            x_flip = random.choice([True, False])

        if y_flip:
            img = img[:, ::-1, :]
        if x_flip:
            img = img[:, :, ::-1]

        if copy:
            img = img.copy()

        if return_param:
            return img, {'y_flip': y_flip, 'x_flip': x_flip}
        else:
            return img

    @staticmethod
    def read_image(path, dtype=np.float32, color=True):
        """Read an image from a file.
        This function reads an image from given file. The image is CHW format and
        the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
        order of the channels is RGB.

        Args:
            path (str): A path of image file.
            dtype: The type of array. The default value is :obj:`~numpy.float32`.
            color (bool): The option determines # channels. RGB if :obj:`True`, grayscale for :obj:`False`

        Returns:
            ~numpy.ndarray: An image.

        """

        f = Image.open(path)
        try:
            if color:
                img = f.convert('RGB')
            else:
                img = f.convert('P')
            img = np.asarray(img, dtype=dtype)
        finally:
            if hasattr(f, 'close'):
                f.close()

        if img.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            return img[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            return img.transpose((2, 0, 1))

    def profile_img(self, pt1, pt2):
        """Plots image profile (x and y)

        Parameters
        ----------
        pt1 : tuple
            coordinates of first point (row1, col1)
        pt2 : tuple
            coordinates of second point (row2, col2)

        Returns
        -------
        fig : matplotlib figure
            projection of histograms

        Examples
        --------
        >>> ImgOps(img_path).profile_img((50,50), (100,100))
        """

        # horizontal
        cols = np.arange(pt1[-1], pt2[-1])
        hor_proj = np.sum(self.gray[:, pt1[-1]:pt2[-1]], axis=0)
        # vertical
        rows = np.arange(pt1[0], pt2[0])
        vert_proj = np.sum(self.gray[pt1[0]:pt2[0], :], axis=1)

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(cols, hor_proj)
        axs[0].title.set_text('Horizontal Profile')
        axs[1].plot(rows, vert_proj)
        axs[1].title.set_text('Vertical Profile')
        plt.tight_layout()
        plt.show()

        return fig
