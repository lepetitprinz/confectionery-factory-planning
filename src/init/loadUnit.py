from common.name import Key, Demand, Item, Resource, Constraint

import os
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

        self.data_path = os.path.join('..', '..', 'data', 'unit')

        # Name instance attribute
        self._key = Key()
        self._dmd = Demand()
        self._res = Resource()
        self._item = Item()
        self._cstr = Constraint()

    def load(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        item = self._load_item()
        route = self._load_route()          # Load the BOM route
        demand = self._load_demand()        # Load the demand dataset
        resource = self._load_resource()    # Load the master dataset
        constraint = self._load_cstr()      # Load the constraint dataset

        data = {
            self._key.item: item,
            self._key.route: route,
            self._key.dmd: demand,
            self._key.res: resource,
            self._key.cstr: constraint,
        }

        return data

    def _load_item(self) -> pd.DataFrame:
        # Demand dataset
        item = self._load_df(path=os.path.join(self.data_path, 'item_mst.csv'))

        return item

    # Load demand dataset
    def _load_demand(self) -> pd.DataFrame:
        # Demand dataset
        demand = self._load_df(path=os.path.join(self.data_path, 'demand.csv'))

        return demand

    # Load resource dataset
    def _load_resource(self) -> Dict[str, pd.DataFrame]:
        resource = {
            # Item master
            self._key.item: self._load_df(path=os.path.join(self.data_path, 'item_mst.csv')),
            # Resource group master
            self._key.res_grp: self._load_df(path=os.path.join(self.data_path, 'res_grp.csv')),
            # Resource group name information
            self._key.res_grp_nm: self._load_df(path=os.path.join(self.data_path, 'res_grp_nm.csv')),
            # Item - resource duration
            self._key.res_duration: self._load_df(path=os.path.join(self.data_path, 'res_dur.csv')),
        }

        return resource

    # Load route dataset
    def _load_route(self) -> Dict[str, pd.DataFrame]:
        route = {
            # BOM route
            self._key.route: self._load_df(path=os.path.join(self.data_path, 'bom_route.csv'))
        }

        return route

    # Load constraint dataset
    def _load_cstr(self) -> Dict[str, pd.DataFrame]:
        constraint = {
            # Job change constraint
            self._key.jc: self._load_df(path=os.path.join(self.data_path, 'job_change.csv')),

            # Resource available time constraint
            self._key.res_avail_time: self._load_df(path=os.path.join(self.data_path, 'res_avail_time.csv')),

            # Human resource usage constraint
            # self._key.human_usage: self._load_df(os.path.join(self.data_path, 'res_human_usage.csv')),

            # Human resource capacity constraint
            # self._key.human_capa: self._load_df(os.path.join(self.data_path, 'res_human_capa.csv')),

            # Simultaneous production constraint
            # self._key.sim_prod_cstr: self._load_df(os.path.join(self.data_path, 'sim_prod_cstr.csv')),

            # Mold capacity constraint
            self._key.mold_cstr: self._load_df(path=os.path.join(self.data_path, 'mold_capa.csv')),
            # 'temp': self._io.load_from_db(sql=self._query.sql_item_weight())
        }

        return constraint

    @staticmethod
    def _load_df(path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        data.columns = [col.lower() for col in data.columns]

        return data
