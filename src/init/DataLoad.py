import common.config as config

import os
import pandas as pd
from typing import Dict


class DataLoad(object):
    key_item = config.key_item
    key_jc = config.key_jc

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
            self.key_item: self.io.get_df_from_db(sql=self.sql_conf.sql_item_master(**fp_version)),
            'res_grp': self.io.get_df_from_db(sql=self.sql_conf.sql_res_grp(**fp_version)),
            'res_grp_nm': self.io.get_df_from_db(sql=self.sql_conf.sql_res_grp_nm()),
            'item_res_duration': self.io.get_df_from_db(sql=self.sql_conf.sql_item_res_duration(**fp_version)),
            'item_res_avail_time': self.io.get_df_from_db(sql=self.sql_conf.sql_res_available_time(**fp_version)),
            self.key_jc: self.io.get_df_from_db(sql=self.sql_conf.sql_job_change(**fp_version)),
        }

        return info

