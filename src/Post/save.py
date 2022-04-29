import common.util as util


import os
import pandas as pd


class Save(object):
    def __init__(self, path: str, fp_version: str, fp_name: str):
        self.path = path
        self.fp_version = fp_version
        self.fp_name = fp_name

    def csv(self, data: pd.DataFrame, name: str):
        save_dir = os.path.join(self.path, 'opt', 'csv', self.fp_version)
        util.make_dir(path=save_dir)

        # Save the optimization result
        data.to_csv(os.path.join(save_dir, name + '_' + self.fp_name + '.csv'), index=False, encoding='cp949')

    def db(self, data):
        pass
