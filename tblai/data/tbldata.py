import os


class XenopusData:
    """Predefined variables and dictionaries related to Xenopus
    """
    def __init__(self):

        # 0-level dirs
        self.pth_glob = '/home/DATA_Lia/data_02/DATASET_SUZY/SM_XENOPUS/DATA/'

        # 1-level dirs
        self.pth_boundary = os.path.join(self.pth_glob, 'Boundary')
        self.pth_crop = os.path.join(self.pth_glob, 'Crop')
        self.pth_dataset = os.path.join(self.pth_glob, 'Dataset')
        self.pth_manual = os.path.join(self.pth_glob, 'Manual')
        self.pth_mask = os.path.join(self.pth_glob, 'Mask')
        self.pth_raw = os.path.join(self.pth_glob, 'Raw')
        self.pth_ref = os.path.join(self.pth_glob, 'References')
        self.pth_register = os.path.join(self.pth_glob, 'Register')
        self.pth_roi = os.path.join(self.pth_glob, 'Roi')
        self.pth_segment = os.path.join(self.pth_glob, 'Segment')

        # 2-level dirs
        self.pth_input_512 = os.path.join(self.pth_dataset, 'Input_512')
        self.pth_label_512 = os.path.join(self.pth_dataset, 'Label_512')
        self.pth_train = os.path.join(self.pth_dataset, 'Train')
        self.pth_test = os.path.join(self.pth_dataset, 'Test')
        self.pth_ftrain = os.path.join(self.pth_dataset, 'Train_200909_all')
        self.pth_ftest = os.path.join(self.pth_dataset, 'Test_200909_all')
        self.pth_ftest_attunet = os.path.join(
            self.pth_dataset, 'Test_200909_all')

        # 3-level dirs
        self.pth_dst = os.path.join(self.pth_dataset, 'Input')
        self.pth_train_img = os.path.join(self.pth_train, 'Image')
        self.pth_train_lbl = os.path.join(self.pth_train, 'Label')
        self.pth_test_img = os.path.join(self.pth_test, 'Image')
        self.pth_test_lbl = os.path.join(self.pth_test, 'Label')
        self.pth_test_result = os.path.join(self.pth_test, 'Result')
        self.pth_ftrain_img = os.path.join(self.pth_ftrain, 'Image')
        self.pth_ftrain_lbl = os.path.join(self.pth_ftrain, 'Label')
        self.pth_ftest_img = os.path.join(self.pth_ftest, 'Image')
        self.pth_ftest_lbl = os.path.join(self.pth_ftest, 'Label')

        # Files
        self.pth_fshift = os.path.join(self.pth_ref, 'frame_shifts.xlsx')
        self.pth_index = os.path.join(self.pth_ref, 'blind_test_index_v2.xlsx')

        self.all_cycles = {
            "191105_Cycle01": ['IVER', 'IWR'],
            "191112_Cycle02": ['BIO', 'IVER'],
            "191127_Cycle03": ['BIO', 'IVER', 'IWR'],
            "191204_Cycle04": ['BIO', 'IVER', 'IWR'],
            "191208_Cycle05": ['BIO', 'IVER', 'IWR'],
            "191230_Cycle06": ['CONTROL', 'CONTROL', 'CONTROL'],
            "200111_Cycle07": ['CONTROL', 'CONTROL', 'CONTROL'],
            "200131_Cycle08": ['CONTROL', 'AG1', 'C59'],
            "200206_Cycle09": ['CONTROL', 'AG1', 'C59'],
            "200305_Cycle10": ['CONTROL', 'CONTROL', 'CONTROL'],
            "200318_Cycle11": ['CONTROL', 'AG1', 'C59'],
            "200323_Cycle12": ['CONTROL', 'AG1', 'C59']
        }

        self.drug_cycles = {
            "191105_Cycle01": ['IVER', 'IWR'],
            "191112_Cycle02": ['BIO', 'IVER'],
            "191127_Cycle03": ['BIO', 'IVER', 'IWR'],
            "191204_Cycle04": ['BIO', 'IVER', 'IWR'],
            "191208_Cycle05": ['BIO', 'IVER', 'IWR'],
            "200131_Cycle08": ['CONTROL', 'AG1', 'C59'],
            "200206_Cycle09": ['CONTROL', 'AG1', 'C59'],
            "200318_Cycle11": ['CONTROL', 'AG1', 'C59'],
            "200323_Cycle12": ['CONTROL', 'AG1', 'C59']
        }

        self.control_cycles = {
            "191230_Cycle06": ['CONTROL', 'CONTROL', 'CONTROL'],
            "200111_Cycle07": ['CONTROL', 'CONTROL', 'CONTROL'],
            "200305_Cycle10": ['CONTROL', 'CONTROL', 'CONTROL'],
        }

        self.drug_position = {
            'BIO': 1,
            'IVER': 2,
            'IWR': 3,
            'CONTROL': 1,
            'AG1': 2,
            'C59': 3
        }

        self.drug_dosage = {
            'AG1': 5,
            'C59': 30
        }

        self.drug_stage = {
            'AG1': 10,
            'C59': 20
        }

        self.well_filter = {
            'Pmin': 800,  # 1200
            'Pmax': 10000,
            'Amin': 30000,  # 80000
            'Amax': 300000,
            'Emin': 0.85,  # 0.9
            'Emax': 0.98,
            'Droi': (6800, 9200, 3),
            'BboxRange': (100, 850)
        }
