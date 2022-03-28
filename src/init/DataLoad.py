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

    def load_mst_temp(self) -> dict:
        bom = self.io.load_object(file_path=os.path.join(self.base_dir, 'data', 'bom.csv'), data_type='csv')
        item = self.io.load_object(file_path=os.path.join(self.base_dir, 'data', 'item.csv'), data_type='csv')
        oper = self.io.load_object(file_path=os.path.join(self.base_dir, 'data', 'operation.csv'), data_type='csv')
        res = self.io.load_object(file_path=os.path.join(self.base_dir, 'data', 'wc.csv'), data_type='csv')

        # change upper case to lower
        bom.columns = [col.lower() for col in bom.columns]
        item.columns = [col.lower() for col in item.columns]
        oper.columns = [col.lower() for col in oper.columns]
        res.columns = [col.lower() for col in res.columns]

        # Rename columns
        bom = bom.rename(columns={'parent_item': 'to', 'child_item': 'from'})

        # Change data type
        bom['rate'] = bom['rate'].astype(int)
        oper['schd_time'] = bom['schd_time'].astype(int)

        mst = {
            'bom': bom,
            'item': item,
            'oper': oper,
            'res': res
        }

        return mst

    def load_demand_temp(self) -> pd.DataFrame:
        demand = self.io.load_object(file_path=os.path.join(self.base_dir, 'data', 'demand.csv'), data_type='csv')
        demand.columns = [col.lower() for col in demand.columns]

        return demand

    def load_mst(self) -> dict:
        bom = self.io.get_df_from_db(sql=self.sql_conf.sql_bom_mst())
        item = self.io.get_df_from_db(sql=self.sql_conf.sql_item_route())
        oper = self.io.get_df_from_db(sql=self.sql_conf.sql_operation())
        res = self.io.get_df_from_db(sql=self.sql_conf.sql_res_mst())

        # Filtering
        bom = bom[['parent_item', 'child_item', 'rate']]
        item = item[['item_cd', 'res_cd']]
        oper = oper[['item_cd', 'operation_no', 'wc_cd', 'schd_time', 'time_uom']]

        # Rename columns
        bom = bom.rename(columns={'parent_item': 'to', 'child_item': 'from'})

        mst = {
            'bom': bom,
            'item': item,
            'oper': oper,
            'res': res
        }

        return mst

    def load_demand(self) -> pd.DataFrame:
        demand = self.io.get_df_from_db(sql=self.sql_conf.sql_demand())

        # Temp
        demand = demand[['demand_id', 'item_cd', 'duedate', 'qty']].copy()

        # Change data type
        demand['qty'] = demand['qty'].astype(int)

        return demand
