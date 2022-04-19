import os
import pandas as pd
from typing import Dict


class DataLoad(object):
    def __init__(self, io, sql_conf, fp_version: str):
        """
        :param io: Pipeline step configuration
        :param sql_conf: SQL configuration
        """
        self.io = io
        self.sql_conf = sql_conf
        self.base_dir = os.path.join('..', '..')
        self.fp_version = fp_version

    def load_demand(self) -> pd.DataFrame:
        demand = self.io.get_df_from_db(sql=self.sql_conf.sql_demand(**{'fp_version': self.fp_version}))

        return demand

    def load_master(self) -> Dict[str, pd.DataFrame]:
        fp_version = {'fp_version': self.fp_version}

        info = {
            'item': self.io.get_df_from_db(sql=self.sql_conf.sql_item_master(**fp_version)),
            'res_grp': self.io.get_df_from_db(sql=self.sql_conf.sql_res_grp(**fp_version)),
            'res_grp_nm': self.io.get_df_from_db(sql=self.sql_conf.sql_res_grp_nm()),
            'item_res_duration': self.io.get_df_from_db(sql=self.sql_conf.sql_item_res_duration(**fp_version)),
            'item_res_avail_time': self.io.get_df_from_db(sql=self.sql_conf.sql_res_available_time()),
            'job_change': self.io.load_object(
                path=os.path.join(self.base_dir, 'data', 'job_change.csv'),
                data_type='csv',
            ),
        }

        return info

