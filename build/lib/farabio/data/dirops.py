import os
import subprocess
import random
import math


class DirOps:
    """Creates instance of DirOps class

    Attributes
    ----------
    path : str
        path of interest
    dirs : list of str
        existing folders at path
    files : list of str
        existing filenames at path
    ndirs : int
        number of only folders at path
    nfiles : int
        number of files at cwd path, except folders
    dirsize : str
        total occupied memory
    titems : int
        total items including subdirectories

    Methods
    -------
    dirinfo(self)
        Prints basic information in a neat format
    split_traintest(self, ratio=0.8, shuffle=True)
        Splits list of files into train and test, shuffles by default
    lsmedia(self)
        Returns media files from path
    """

    def __init__(self, path):
        """Constructor for dirops class

        Parameters
        ----------
        path : str
            path of interest
        """

        self.path = path
        self.subdirs = sorted([name for name in os.listdir(
            path) if os.path.isdir(os.path.join(path, name))])
        self.files = sorted([name for name in os.listdir(
            path) if not os.path.isdir(os.path.join(path, name))])

        self.nsubdirs = len(self.subdirs)
        self.nfiles = len(self.files)
        self.dirsize = subprocess.check_output(
            ['du', '-sh', self.path]).split()[0].decode('utf-8')

        self.titems = 0
        for root, dirs, files in os.walk(self.path):
            self.titems += len(files)

    def dirinfo(self):
        """Prints basic information in a neat format

        Examples
        ----------
        >>> DirOps(dir_path).dirinfo()
        """
        print(f"Info on path: {self.path}")
        print(
            f"Subdirs: {self.nsubdirs} | Files: {self.nfiles} | Recursive files: {self.titems} ")
        print(f"Memory: {self.dirsize}")

    def split_traintest(self, ratio=0.8, shuffle=True):
        """Splits list of files in path into train and test

        Parameters
        ----------
        ratio : float
            ratio to split into train and test
        shuffle : bool
            flag to perform shuffle before split

        Returns
        -------
        train, test : tuple of lists

        Examples
        ----------
        >>> (train_list, test_list) = DirOps(dir_path).split_traintest(ratio=0.7)
        """

        if shuffle is True:
            files = random.sample(self.files, len(self.files))
        else:
            files = self.files

        split_idx = math.floor(ratio*len(files))
        train, test = files[:split_idx], files[split_idx:]

        return (train, test)

    def lsmedia(self):
        """Returns media files from path

        Returns
        -------
        (img_fns, vid_fns, aud_fns) : tuple of lists
            img_fns : list of image files
            vid_fns : list of video files
            aud_fns : list of audio files
        
        Examples
        ----------
        >>> DirOps(dir_path).lsmedia()
        """

        img_fm = (".tif", ".tiff", ".jpg", ".jpeg", ".gif", ".png", ".eps",
                  ".raw", ".cr2", ".nef", ".orf", ".sr2", ".bmp", ".ppm", ".heif")
        vid_fm = (".flv", ".avi", ".mp4", ".3gp", ".mov",
                  ".webm", ".ogg", ".qt", ".avchd")
        aud_fm = (".flac", ".mp3", ".wav", ".wma", ".aac")
        media_fms = {"image": img_fm, "video": vid_fm, "audio": aud_fm}

        def fns(path, media): return [fn for fn in os.listdir(path) if any(
            fn.lower().endswith(media_fms[media]) for ext in media_fms[media])]
        img_fns, vid_fns, aud_fns = fns(self.path, "image"), fns(
            self.path, "video"), fns(self.path, "audio")

        print(f"State of media in '{self.path}'")
        print("Images: ", len(img_fns), " | Videos: ",
              len(vid_fns), "| Audios: ", len(aud_fns))

        return (img_fns, vid_fns, aud_fns)

    def del_files(self, reflist, match=True):
        """Deletes files in current folder that matches / not matches with list

        Parameters
        ----------
        reflist : list
            list of items in folder
        match : bool
            deletes files which match, if True
        
        Examples
        ----------
        >>> sample_list = ['1.png', '3.png', '4.png']
        >>> DirOps(dir_path).del_files(sample_list, match=False)
        """

        if match is True:
            for i, fname in enumerate(self.files):
                if fname in reflist:
                    os.remove(os.path.join(self.path, fname))

        if match is False:
            for i, fname in enumerate(self.files):
                if fname not in reflist:
                    os.remove(os.path.join(self.path, fname))
