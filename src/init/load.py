from common.name import Key, Demand, Item, Resource, Constraint


import pandas as pd
from typing import Dict


class DataLoad(object):
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

        # Name instance attribute
        self.key = Key()
        self.dmd = Demand()
        self.res = Resource()
        self.item = Item()
        self.cstr = Constraint()

    def load_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        demand = self.load_demand()        # Load the demand dataset
        resource = self.load_resource()    # Load the master dataset
        constraint = self.load_cstr()      # Load the constraint dataset
        route = self.load_route()          # Load the BOm route

        data = {
            self.key.dmd: demand,
            self.key.res: resource,
            self.key.cstr: constraint,
            self.key.route: route
        }

        return data

    def load_demand(self) -> pd.DataFrame:
        # Demand dataset
        demand = self.io.load_from_db(sql=self.query.sql_demand(**self.fp_vrsn_date))

        return demand

    def load_resource(self) -> Dict[str, pd.DataFrame]:
        resource = {
            # Item master
            self.key.item: self.io.load_from_db(sql=self.query.sql_item_master(**self.fp_vrsn_date)),

            # Resource group master
            self.key.res_grp: self.io.load_from_db(sql=self.query.sql_res_grp(**self.fp_vrsn_date)),

            # Resource group name information
            self.key.res_grp_nm: self.io.load_from_db(sql=self.query.sql_res_grp_nm()),

            # Item - resource duration
            self.key.res_duration: self.io.load_from_db(sql=self.query.sql_item_res_dur(**self.fp_vrsn_date)),
        }

        return resource

    def load_route(self):
        route = {
            # BOM route
            self.key.route: self.io.load_from_db(sql=self.query.sql_bom_route(**self.fp_vrsn_date))
        }

        return route

    def load_cstr(self) -> Dict[str, pd.DataFrame]:
        constraint = {
            # Job change constraint
            self.key.jc: self.io.load_from_db(sql=self.query.sql_job_change(**self.fp_vrsn_date)),

            # Resource available time constraint
            self.key.res_avail_time: self.io.load_from_db(sql=self.query.sql_res_avail_time(**self.fp_vrsn_date)),

            # Human resource usage constraint
            self.key.human_usage: self.io.load_from_db(sql=self.query.sql_res_human_usage(**self.fp_vrsn_date)),

            # Human resource capacity constraint
            self.key.human_capa: self.io.load_from_db(sql=self.query.sql_res_human_capacity(**self.fp_vrsn_date)),

            # Simultaneous production constraint
            self.key.sim_prod_cstr: self.io.load_from_db(sql=self.query.sql_sim_prod_cstr(**self.fp_vrsn_date)),
        }

        return constraint
