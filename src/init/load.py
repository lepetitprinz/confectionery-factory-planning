import common.config as config

import os
import pandas as pd
from typing import Dict


class DataLoad(object):
    ############################################
    # Dictionary key configuration
    ############################################
    # Main
    key_dmd = config.key_dmd      # Demand
    key_res = config.key_res      # Resource
    key_cstr = config.key_cstr    # Constraint
    key_item = config.key_item    # Item

    # Resource
    key_res_grp = config.key_res_grp
    key_res_grp_nm = config.key_res_grp_nm
    key_item_res_duration = config.key_res_duration

    # Constraint
    key_jc = config.key_jc
    key_human_capa = config.key_human_capa
    key_human_usage = config.key_human_usage
    key_sim_prod_cstr = config.key_sim_prod_cstr
    key_res_avail_time = config.key_res_avail_time

    def __init__(self, io, query, fp_version: str, fp_seq: str):
        """
        :param io: Pipeline step configuration
        :param query: SQL configuration
        :param fp_version: Factory planning version
        """
        self.io = io
        self.query = query
        self.fp_seq = fp_seq
        self.fp_version = fp_version
        self.fp_vrsn_dict = {'fp_vrsn_id': fp_version, 'fp_vrsn_seq': fp_seq}

    def load_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        demand = self.load_demand()      # Load the demand dataset
        resource = self.load_resource()    # Load the master dataset
        constraint = self.load_cstr()    # Load the constraint dataset

        data = {
            self.key_dmd: demand,
            self.key_res: resource,
            self.key_cstr: constraint
        }

        return data

    def load_demand(self) -> pd.DataFrame:
        demand = self.io.load_from_db(sql=self.query.sql_demand(**self.fp_vrsn_dict))

        return demand

    def load_resource(self) -> Dict[str, pd.DataFrame]:
        resource = {
            # Item master
            self.key_item: self.io.load_from_db(sql=self.query.sql_item_master(**self.fp_vrsn_dict)),
            # Resource group
            self.key_res_grp: self.io.load_from_db(sql=self.query.sql_res_grp(**self.fp_vrsn_dict)),
            # Resource group name
            self.key_res_grp_nm: self.io.load_from_db(sql=self.query.sql_res_grp_nm()),
            # Item - resource duration
            self.key_item_res_duration: self.io.load_from_db(sql=self.query.sql_item_res_dur(**self.fp_vrsn_dict)),
        }

        return resource

    def load_cstr(self) -> Dict[str, pd.DataFrame]:
        constraint = {
            # Job change
            self.key_jc: self.io.load_from_db(sql=self.query.sql_job_change(**self.fp_vrsn_dict)),
            # Resource available time
            self.key_res_avail_time: self.io.load_from_db(sql=self.query.sql_res_avail_time(**self.fp_vrsn_dict)),
            # Human resource capacity
            # self.key_human_capa: self.io.load_from_db(sql=self.query.sql_res_human_capacity(**self.fp_vrsn_dict)),
            self.key_human_capa: self.io.load_object(    # ToDo: Temp dataset
                path=os.path.join('..', '..', 'data', 'human', 'human_capacity_temp.csv'), data_type='csv'),
            # Human resource usage
            self.key_human_usage: self.io.load_from_db(sql=self.query.sql_res_human_usage(**self.fp_vrsn_dict)),
            # Simultaneous production constraint
            self.key_sim_prod_cstr: self.io.load_from_db(sql=self.query.sql_sim_prod_cstr(**self.fp_vrsn_dict)),
        }

        return constraint
