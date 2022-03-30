import os

import pandas as pd


class DataLoadDev(object):
    def __init__(self, io, sql_conf):
        """
        :param io: Pipeline step configuration
        :param sql_conf: SQL configuration
        """
        self.io = io
        self.sql_conf = sql_conf
        self.base_dir = os.path.join('..', '..')

    def load_info(self) -> dict:
        info = {
            'bom_route': self.io.get_df_from_db(sql=self.sql_conf.sql_bom_route()),
            'operation': self.io.get_df_from_db(sql=self.sql_conf.sql_operation()),
            'resource': self.io.get_df_from_db(sql=self.sql_conf.sql_res_grp()),
            'res_grp_item': self.io.get_df_from_db(sql=self.sql_conf.sql_res_grp_item()),
        }

        return info

    def load_demand(self) -> pd.DataFrame:
        demand = self.io.get_df_from_db(sql=self.sql_conf.sql_demand())

        # Temp
        # demand = demand[['demand_id', 'item_cd', 'duedate', 'qty']].copy()

        # Change data type
        # demand['qty'] = demand['qty'].astype(int)

        return demand
