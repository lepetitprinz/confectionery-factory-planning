import numpy as np
import pandas as pd


class Preprocessing(object):
    def __init__(self):
        self.due_date_col = 'due_date'

    def run(self, mst, demand):

        self.prep_mst(data=mst)

        self.prep_demand(data=demand)

    def prep_mst(self, data):
        item = self.prep_item(data=data['item'])
        oper = self.prep_oper(data=data['item'])
        res = self.prep_res(data=data['item'])

    def prep_demand(self, data: pd.DataFrame):
        data[self.due_date_col] = self.change_data_type(data=data['duedate'], data_type='str')
        data = self.calc_deadline(data=data)

    def prep_item(self, data: pd.DataFrame):
        # Rename columns
        data = data.rename(columns={
            'parent_item': 'to',
            'child_item': 'from',
        })

        # get unique item list
        item_list = list(set(data['from'].values + data['to'].values))


    @staticmethod
    def change_data_type(data, data_type: str):
        if data_type == 'str':
            data = data.astype(str)

        return data

    def calc_deadline(self, data):
        due_date_min = data[self.due_date_col].min()
        data['days'] = data[self.due_date_col] - due_date_min
        data['minutes'] = data['days'] / np.timedelta64(1, "m")

        return data