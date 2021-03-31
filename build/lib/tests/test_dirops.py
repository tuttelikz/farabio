import os
import unittest
from farabio.data.dirops import DirOps


class TestDirOps(unittest.TestCase):
    def test_dirinfo(self):
        sample_dir = './images/'
        dir_object = DirOps(sample_dir)
        print(dir_object.dirinfo())

    def test_split_traintest(self):
        sample_dir = './images/'
        dir_object = DirOps(sample_dir)
        test = dir_object.split_traintest()[0]
        print(test)

    def test_lsmedia(self):
        sample_dir = './images/'
        dir_object = DirOps(sample_dir)
        dir_object.lsmedia()

    def test_del_files(self):
        sample_list = ['bitmap_font_1_basic.png', 
        'bitmap_font_2_basic.png', 'bitmap_font_2_raqm.png']

        sample_dir = './images/'
        dir_object = DirOps(sample_dir)
        dir_object.del_files(sample_list, match=False)

        print(os.listdir(sample_dir))


if __name__ == '__main__':
    unittest.main()
