import unittest
from farabio.data.imgops import ImgOps
import matplotlib.pyplot as plt


# you can display on server the result on server using X11 forwarding
class TestImgOps(unittest.TestCase):
    # def test_init(self):
    #     sample_dir = './images/bitmap_font_8_raqm.png'
    #     img_object = ImgOps(sample_dir)

    #     # plt.imshow(dir_object.img)
    #     # plt.show()
    #     print(img_object.h, img_object.w, img_object.ch)

    # def test_get_date(self):
    #     sample_dir = './images/2020-01-13-0056.jpg'
    #     img_object = ImgOps(sample_dir)
    #     print(img_object.get_date())

    # def test_slice_img(self):
    #     sample_dir = './images/bitmap_font_8_raqm.png'
    #     img_object = ImgOps(sample_dir)

    #     img_sl, img_sli = img_object.slice_img(2, 'y')
    #     plt.imshow(img_sl[0])
    #     plt.show()

    # def test_pad_img(self):
    #     sample_dir = './images/bitmap_font_8_raqm.png'
    #     img_object = ImgOps(sample_dir)

    #     pad_img = img_object.pad_img(droi=(500, 500), simg=None)
    #     plt.imshow(pad_img)
    #     plt.show()

    # def test_approx_bcg(self):
    #     #
    #     sample_dir = './images/approx_bcg.jpg'
    #     img_object = ImgOps(sample_dir)

    #     approx_bcg = img_object.approx_bcg(channel='blue')

    #     plt.imshow(approx_bcg)
    #     plt.show()

    # def test_blend_img(self):
    #     sample_img_dir = './images/argb-32bpp_MipMaps-1.png'
    #     sample_ref_dir = './images/bc7-argb-8bpp_MipMaps-1.png'

    #     img_object = ImgOps(sample_img_dir)
    #     ref_img = ImgOps(sample_ref_dir).img

    #     img_blended = img_object.blend_img(ref_img)
    #     plt.imshow(img_blended)
    #     plt.show()

    # def test_mask_img(self):

    #     sample_img_dir = './images/plate_rgb.jpg'
    #     sample_mask_dir = './images/plate_mask.jpg'

    #     img_object = ImgOps(sample_img_dir)
    #     mask_img = ImgOps(sample_mask_dir).img

    #     img_ovr = img_object.mask_img(mask_img)
    #     plt.imshow(img_ovr)
    #     plt.show()

    def test_mask_img(self):

        sample_img_dir = './images/plate_rgb.jpg'
        img_object = ImgOps(sample_img_dir)

        img_object.profile_img((0, 0), (50, 50))


if __name__ == '__main__':
    unittest.main()
