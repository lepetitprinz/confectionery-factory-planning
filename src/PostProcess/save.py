import common.util as util
from PostProcess import process


import os
import pandas as pd


class Csv(process):
    def __init__(self):
        super.__init__()

    def save(self, data: pd.DataFrame, name: str):
        save_dir = os.path.join(self.save_path, 'opt', 'csv', self.fp_version)
        util.make_dir(path=save_dir)

        # Save the optimization result
        data.to_csv(os.path.join(save_dir, name + '_' + self.fp_name + '.csv'), index=False, encoding='cp949')


class DB(process):
    def __init__(self):
        super.__init__()