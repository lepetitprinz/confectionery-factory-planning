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

    def __init__(self, io, query, version):
        """
        :param io: Pipeline step configuration
        :param query: SQL configuration
        :param version: Factory planning version & sequence
        """
        self.io = io
        self.query = query
        self.version = version
        self.fp_vrsn_date = {
            'fp_vrsn_id': version.fp_version,
            'fp_vrsn_seq': version.fp_seq,
            'yy': version.fp_version[3:7],
            'week': version.fp_version[7:10]
        }

    def load_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        demand = self.load_demand()        # Load the demand dataset
        resource = self.load_resource()    # Load the master dataset
        constraint = self.load_cstr()      # Load the constraint dataset

        data = {
            self.key_dmd: demand,
            self.key_res: resource,
            self.key_cstr: constraint
        }

        return data

    def load_demand(self) -> pd.DataFrame:
        # Demand dataset
        demand = self.io.load_from_db(sql=self.query.sql_demand(**self.fp_vrsn_date))

        return demand

    def load_resource(self) -> Dict[str, pd.DataFrame]:
        resource = {
            # Item master
            self.key_item: self.io.load_from_db(sql=self.query.sql_item_master(**self.fp_vrsn_date)),

            # Resource group master
            self.key_res_grp: self.io.load_from_db(sql=self.query.sql_res_grp(**self.fp_vrsn_date)),

            # Resource group name information
            self.key_res_grp_nm: self.io.load_from_db(sql=self.query.sql_res_grp_nm()),

            # Item - resource duration
            self.key_item_res_duration: self.io.load_from_db(sql=self.query.sql_item_res_dur(**self.fp_vrsn_date)),
        }

        return resource

    def load_cstr(self) -> Dict[str, pd.DataFrame]:
        constraint = {
            # Job change constraint
            self.key_jc: self.io.load_from_db(sql=self.query.sql_job_change(**self.fp_vrsn_date)),

            # Resource available time constraint
            self.key_res_avail_time: self.io.load_from_db(sql=self.query.sql_res_avail_time(**self.fp_vrsn_date)),

            # Human resource usage constraint
            self.key_human_usage: self.io.load_from_db(sql=self.query.sql_res_human_usage(**self.fp_vrsn_date)),

            # Human resource capacity constraint
            self.key_human_capa: self.io.load_from_db(sql=self.query.sql_res_human_capacity(**self.fp_vrsn_date)),

            # Simultaneous production constraint
            self.key_sim_prod_cstr: self.io.load_from_db(sql=self.query.sql_sim_prod_cstr(**self.fp_vrsn_date)),
        }

        return constraint
