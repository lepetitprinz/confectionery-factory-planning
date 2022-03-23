import common.util as util

import numpy as np
import pandas as pd
import networkx as nx


class Preprocessing(object):
    def __init__(self):
        self.due_date_col = 'due_date'

        self.item_code_map = {}
        self.item_code_rev_map = {}
        self.oper_map = {}
        self.res_map = {}

    def run(self, mst, demand):

        mst_map = self.prep_mst(data=mst)

        dmd_qty = self.prep_demand(data=demand)

        bom_route = self.make_bom_route(bom=mst['bom'])

        return mst_map, dmd_qty, bom_route

    def prep_mst(self, data):
        self.prep_bom(data=data['bom'])
        oper_map = self.prep_oper(data=data['oper'])
        res_map = self.prep_res(data=data['res'])

        mst_map = {
            'oper': oper_map,
            'res': res_map
        }

        return mst_map

    def prep_demand(self, data: pd.DataFrame):
        data[self.due_date_col] = self.change_data_type(data=data['duedate'], data_type='str')
        data[self.due_date_col] = self.change_data_type(data=data[self.due_date_col], data_type='datetime')
        data = self.calc_deadline(data=data)

        data['code'] = [self.item_code_map[item] for item in data['item_cd']]
        dmd_qty = data.groupby("code")['qty'].sum()

        return dmd_qty

    def make_bom_route(self, bom: pd.DataFrame):
        _, bom_route = nx.DiGraph(), nx.DiGraph()

        for i in bom.index:
            _.add_edge(
                bom['from'][i],
                bom['to'][i],
                rate=bom['rate'][i]
            )
            bom_route.add_edge(
                self.item_code_map[bom['from'][i]],
                self.item_code_map[bom['to'][i]],
                rate=bom['rate'][i]
            )
        return bom_route

    def prep_bom(self, data: pd.DataFrame):
        # get unique item list
        bom_list = list(set(data['from'].values) | set(data['to'].values))

        # item vs code mapping
        self.item_code_map = {item: util.generate_alphabet_code(i, 10) for i, item in enumerate(bom_list)}
        self.item_code_rev_map = {util.generate_alphabet_code(i, 10): item for i, item in enumerate(bom_list)}

    def prep_oper(self, data: pd.DataFrame) -> dict:
        data['code'] = [self.item_code_map[item] for item in data['item_cd']]
        data['schd_time'] = np.ceil(data['schd_time'])

        # Todo: need to correct code
        oper_map = dict(list(map(lambda x: (x[0], x[1]), data.groupby('code'))))

        return oper_map

    def prep_res(self, data: pd.DataFrame) -> dict:
        res_map = {code: num for code, num in zip(data['wc_cd'], data['wc_num'])}

        return res_map

    @staticmethod
    def change_data_type(data, data_type: str):
        if data_type == 'str':
            data = data.astype(str)
        elif data_type == 'datetime':
            data = pd.to_datetime(data)

        return data

    def calc_deadline(self, data):
        due_date_min = data[self.due_date_col].min()
        data['days'] = data[self.due_date_col] - due_date_min
        data['minutes'] = data['days'] / np.timedelta64(1, "m")

        return data