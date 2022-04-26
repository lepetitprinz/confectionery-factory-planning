import common.config as config

import os
import pandas as pd
from typing import Dict


class DataLoad(object):
    key_item = config.key_item
    key_jc = config.key_jc
    key_res_grp = config.key_res_grp
    key_res_grp_nm = config.key_res_grp_nm
    key_res_avail_time = config.key_res_avail_time
    key_item_res_duration = config.key_item_res_duration
    key_sim_prod_cstr = config.key_sim_prod_cstr

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
            self.key_res_grp: self.io.get_df_from_db(sql=self.sql_conf.sql_res_grp(**fp_version)),
            self.key_res_grp_nm: self.io.get_df_from_db(sql=self.sql_conf.sql_res_grp_nm()),
            self.key_item_res_duration: self.io.get_df_from_db(sql=self.sql_conf.sql_item_res_duration(**fp_version)),
            self.key_res_avail_time: self.io.get_df_from_db(sql=self.sql_conf.sql_res_available_time(**fp_version)),
            self.key_jc: self.io.get_df_from_db(sql=self.sql_conf.sql_job_change(**fp_version)),
            self.key_sim_prod_cstr: self.io.load_object(    # ToDo: Temp
                path=os.path.join('..', '..', 'data', 'K130_simul_prod_cnst_sample.csv'), data_type='csv')
        }

        return info

