import os

import pandas as pd


class DataLoad(object):
    def __init__(self, io, sql_conf):
        """
        :param io: Pipeline step configuration
        :param sql_conf: SQL configuration
        """
        self.io = io
        self.sql_conf = sql_conf
        self.base_dir = os.path.join('..', '..')

    def load_master(self) -> dict:
        info = {
            'res_grp': self.io.get_df_from_db(sql=self.sql_conf.sql_res_grp()),
            'res_human': self.io.load_object(
                file_path=os.path.join(self.base_dir, 'data', 'res_people.csv'),
                data_type='csv'
            ),
            'res_human_map': self.io.load_object(
                file_path=os.path.join(self.base_dir, 'data', 'res_people_map.csv'),
                data_type='csv'
            ),
            'item_res_duration': self.io.get_df_from_db(sql=self.sql_conf.sql_item_res_duration()),
        }

        return info

    def load_demand(self) -> pd.DataFrame:
        demand = self.io.get_df_from_db(sql=self.sql_conf.sql_demand())

        # Temp
        # demand = demand[['demand_id', 'item_cd', 'duedate', 'qty']].copy()

        # Change data type
        # demand['qty'] = demand['qty'].astype(int)

        return demand
