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
        self._io = io
        self._query = query
        self._fp_vrsn_date = {
            'fp_vrsn_id': version.fp_version,    # Factory planning version
            'fp_vrsn_seq': version.fp_seq,       # Factory planning sequence
            'yy': version.fp_version[3: 7],      # year
            'week': version.fp_version[7: 10]    # week
        }

        # Name instance attribute
        self._key = Key()
        self._dmd = Demand()
        self._res = Resource()
        self._item = Item()
        self._cstr = Constraint()

    def load(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        route = self._load_route()          # Load the BOM route
        demand = self._load_demand()        # Load the demand dataset
        resource = self._load_resource()    # Load the master dataset
        constraint = self._load_cstr()      # Load the constraint dataset

        data = {
            self._key.route: route,
            self._key.dmd: demand,
            self._key.res: resource,
            self._key.cstr: constraint,
        }

        return data

    # Load demand dataset
    def _load_demand(self) -> pd.DataFrame:
        # Demand dataset
        demand = self._io.load_from_db(sql=self._query.sql_demand(**self._fp_vrsn_date))

        return demand

    # Load resource dataset
    def _load_resource(self) -> Dict[str, pd.DataFrame]:
        resource = {
            # Item master
            self._key.item: self._io.load_from_db(sql=self._query.sql_item_master(**self._fp_vrsn_date)),

            # Resource group master
            self._key.res_grp: self._io.load_from_db(sql=self._query.sql_res_grp(**self._fp_vrsn_date)),

            # Resource group name information
            self._key.res_grp_nm: self._io.load_from_db(sql=self._query.sql_res_grp_nm()),

            # Item - resource duration
            self._key.res_duration: self._io.load_from_db(sql=self._query.sql_item_res_dur(**self._fp_vrsn_date)),
        }

        return resource

    # Load route dataset
    def _load_route(self) -> Dict[str, pd.DataFrame]:
        route = {
            # BOM route
            self._key.route: self._io.load_from_db(sql=self._query.sql_bom_route(**self._fp_vrsn_date))
        }

        return route

    # Load constraint dataset
    def _load_cstr(self) -> Dict[str, pd.DataFrame]:
        constraint = {
            # Job change constraint
            self._key.jc: self._io.load_from_db(sql=self._query.sql_job_change(**self._fp_vrsn_date)),

            # Resource available time constraint
            self._key.res_avail_time: self._io.load_from_db(sql=self._query.sql_res_avail_time(**self._fp_vrsn_date)),

            # Human resource usage constraint
            self._key.human_usage: self._io.load_from_db(sql=self._query.sql_res_human_usage(**self._fp_vrsn_date)),

            # Human resource capacity constraint
            self._key.human_capa: self._io.load_from_db(sql=self._query.sql_res_human_capacity(**self._fp_vrsn_date)),

            # Simultaneous production constraint
            self._key.sim_prod_cstr: self._io.load_from_db(sql=self._query.sql_sim_prod_cstr(**self._fp_vrsn_date)),

            # Mold capacity constraint
            self._key.mold_cstr: self._io.load_from_db(sql=self._query.sql_mold_capacity_temp(**self._fp_vrsn_date))
        }

        return constraint
