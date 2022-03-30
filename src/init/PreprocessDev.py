import datetime
import common.util as util

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict


class PreprocessDev(object):
    # Column name setting
    col_plant = 'plant_cd'
    col_sku = 'item_cd'
    col_res_grp = 'res_grp_cd'
    col_res = 'res_cd'
    col_capacity = 'capacity'
    col_qty = 'qty'
    col_route_from = 'from_item'
    col_route_to = 'to_item'
    col_bom_lvl = 'bom_lvl'
    col_from_to_rate = 'rate'
    col_duration = 'duration'
    col_due_date = 'due_date'

    # Column usage setting
    use_col_dmd = ['dmd_id', 'item_cd', 'qty', 'due_date']
    use_col_res_grp = ['plant_cd', 'res_grp_cd', 'res_cd', 'capacity']

    def __init__(self):
        # configuration
        self.dmd_plant_list = []
        self.dmd_plant_item_map = {}
        self.item_code_map = {}
        self.item_code_rev_map = {}
        self.oper_map = {}
        self.res_map = {}

    def set_dmd_info(self, data: pd.DataFrame):
        # Get plant list of demand list
        dmd_plant_list = list(set(data[self.col_plant]))

        # Change data type
        data[self.col_sku] = data[self.col_sku].astype(str)
        data[self.col_res] = data[self.col_res].astype(str)
        data[self.col_due_date] = data[self.col_due_date].astype(str)
        data[self.col_due_date] = pd.to_datetime(data[self.col_due_date])
        data[self.col_qty] = np.ceil(data[self.col_qty]).astype(int)    # Ceiling quantity

        # Add columns
        data = self.calc_deadline(data=data)

        # group demand by each plant
        dmd_by_plant, dmd_plant_item_map = {}, {}
        for plant in dmd_plant_list:
            # Filter by each plant
            dmd = data[data[self.col_plant] == plant]

            # Convert form of demand data
            dmd_by_plant[plant] = self.convert_dmd_form(data=dmd)

            # all of demand item list by plant
            dmd_plant_item_map[plant] = list(set(dmd[self.col_sku]))

        self.dmd_plant_list = dmd_plant_list
        self.dmd_plant_item_map = dmd_plant_item_map

        return dmd_plant_list, dmd_by_plant

    def convert_dmd_form(self, data: pd.DataFrame) -> List[Tuple[Any]]:
        data_use = data[self.use_col_dmd].copy()
        data_tuple = [tuple(row) for row in data_use.to_numpy()]

        return data_tuple

    def set_res_grp(self, data) -> dict:
        # Choose columns used in model
        data = data[self.use_col_res_grp].copy()

        # Change data type
        data[self.col_res] = data[self.col_res].astype(str)
        data[self.col_capacity] = data[self.col_capacity].astype(int)

        # Choose plants of demand list
        data = data[data[self.col_plant].isin(self.dmd_plant_list)].copy()

        # Group resource by each plant
        res_grp_by_plant = {}
        for plant in self.dmd_plant_list:
            res_grp_df = data[data[self.col_plant] == plant]

            res_grp_to_res = {}
            res_grp_list = list(set(res_grp_df[self.col_res_grp].values))
            for res_grp in res_grp_list:
                res_grp_filtered = res_grp_df[res_grp_df[self.col_res_grp] == res_grp].copy()
                res_grp_to_res[res_grp] = [tuple(row) for row in
                                           res_grp_filtered[[self.col_res, self.col_capacity]].to_numpy()]

            res_grp_by_plant[plant] = res_grp_to_res

        return res_grp_by_plant

    def set_res_grp_item(self, data) -> dict:
        # Change data type
        data[self.col_res] = data[self.col_res].astype(str)
        data[self.col_sku] = data[self.col_sku].astype(str)

        # Choose plants of demand list
        data = data[data[self.col_plant].isin(self.dmd_plant_list)].copy()

        res_item_by_plant = {}
        for plant in self.dmd_plant_list:
            # Filter data contained in each plant
            res_item = data[data[self.col_plant] == plant]

            # Filter only items of each plant involved in demand
            res_item = res_item[res_item[self.col_sku].isin(self.dmd_plant_item_map[plant])]

            res_item_list = defaultdict(list)
            for res, item in zip(res_item[self.col_res], res_item[self.col_sku]):
                res_item_list[item].append(res)

            res_item_by_plant[plant] = res_item_list

        return res_item_by_plant

    def set_bom_route_info(self, data):
        # rename columns
        data = data.rename(columns={'child_item': self.col_route_from, 'parent_item': self.col_route_to})

        # Change data type
        data[self.col_route_from] = data[self.col_route_from].astype(str)
        data[self.col_route_to] = data[self.col_route_to].astype(str)
        data[self.col_bom_lvl] = data[self.col_bom_lvl].astype(int)
        data[self.col_from_to_rate] = np.ceil(data[self.col_from_to_rate])

        # Choose plants of demand list
        data = data[data[self.col_plant].isin(self.dmd_plant_list)].copy()

        # Group bom route by each plant
        bom_route_by_plant = {}
        for plant in self.dmd_plant_list:
            # Filter data contained in each plant
            bom_route = data[data[self.col_plant] == plant].copy()

            # Filter only items of each plant involved in demand
            bom_route = bom_route[bom_route[self.col_route_to].isin(self.dmd_plant_item_map[plant])]

            # Generate item <-> code map
            bom_map = self.generate_item_code_map(data=bom_route)

            # draw bom route graph
            route_graph = self.draw_graph(data=bom_route, data_map=bom_map)

            bom_route_by_plant[plant] = {
                'map': bom_map,
                'graph': route_graph
            }

        return bom_route_by_plant

    def set_oper_info(self, data):
        # rename columns
        data = data.rename(columns={'wc_cd': self.col_res, 'schd_time': self.col_duration})

        # Change data type
        data[self.col_sku] = data[self.col_sku].astype(str)
        data[self.col_res] = data[self.col_res].astype(str)
        data[self.col_duration] = np.ceil(data[self.col_duration])

        # Group bom route by each plant
        oper_by_plant = {}
        for plant in self.dmd_plant_list:
            # Filter data contained in each plant
            operation = data[data[self.col_plant] == plant].copy()

            # Filter only items of each plant involved in demand
            operation = operation[operation[self.col_sku].isin(self.dmd_plant_item_map[plant])]

            oper_by_plant[plant] = operation

        return oper_by_plant

    def generate_item_code_map(self, data):
        # get unique item list
        item_list = list(set(data[self.col_route_from].values) | set(data[self.col_route_to].values))
        item_list = sorted(item_list)

        item_to_code, code_to_item = {}, {}
        for i, item in enumerate(item_list):
            code = util.generate_alphabet_code(i, 10)
            item_to_code[item] = code
            code_to_item[code] = item

        return {'item_to_code': item_to_code, 'code_to_item': code_to_item}

    def draw_graph(self, data: pd.DataFrame, data_map):
        _, route_graph = nx.DiGraph(), nx.DiGraph()

        for route_from, route_to, rate in zip(
                data[self.col_route_from],
                data[self.col_route_to],
                data[self.col_from_to_rate]
        ):
            _.add_edge(route_from, route_to, rate=rate)
            route_graph.add_edge(
                data_map['item_to_code'][route_from],
                data_map['item_to_code'][route_to],
                rate=rate)

        return route_graph

    def prep_bom(self, data: pd.DataFrame):
        # get unique item list
        bom_list = list(set(data['from'].values) | set(data['to'].values))

        # item vs code mapping
        self.item_code_map = {item: util.generate_alphabet_code(i, 10) for i, item in enumerate(bom_list)}
        self.item_code_rev_map = {util.generate_alphabet_code(i, 10): item for i, item in enumerate(bom_list)}
        bom_map = {
            'item_code_map': self.item_code_map,
            'item_code_rev_map': self.item_code_rev_map
        }

        return bom_map

    def run(self, mst, demand):
        mst_map, operation = self.prep_mst(data=mst)

        demand, dmd_qty = self.prep_demand(data=demand)

        bom_route = self.make_bom_route(bom=mst['bom'])

        return mst_map, demand, dmd_qty, bom_route, operation

    def prep_mst(self, data):
        bom_map = self.prep_bom(data=data['bom'])
        operation, oper_map = self.prep_oper(data=data['oper'])
        res_map = self.prep_res(data=data['res'])

        mst_map = {
            'bom': bom_map,
            'oper': oper_map,
            'res': res_map
        }

        return mst_map, operation

    def prep_demand(self, data: pd.DataFrame):
        data[self.col_due_date] = self.change_data_type(data=data['duedate'], data_type='str')
        data[self.col_due_date] = self.change_data_type(data=data[self.col_due_date], data_type='datetime')
        demand = self.calc_deadline(data=data)

        demand['code'] = [self.item_code_map[item] for item in demand['item_cd']]
        dmd_qty = demand.groupby("code")['qty'].sum()

        return demand, dmd_qty

    def make_bom_route(self, bom: pd.DataFrame):
        _, bom_route = nx.DiGraph(), nx.DiGraph()

        for bom_from, bom_to, rate in zip(bom['from'], bom['to'], bom['rate']):
            _.add_edge(bom_from, bom_to, rate=rate)
            bom_route.add_edge(self.item_code_map[bom_from], self.item_code_map[bom_to], rate=rate)

        return bom_route

    def prep_bom(self, data: pd.DataFrame):
        # get unique item list
        bom_list = list(set(data['from'].values) | set(data['to'].values))

        # item vs code mapping
        self.item_code_map = {item: util.generate_alphabet_code(i, 10) for i, item in enumerate(bom_list)}
        self.item_code_rev_map = {util.generate_alphabet_code(i, 10): item for i, item in enumerate(bom_list)}
        bom_map = {
            'item_code_map': self.item_code_map,
            'item_code_rev_map': self.item_code_rev_map
        }

        return bom_map

    def prep_oper(self, data: pd.DataFrame) -> dict:
        # Filter
        data = data[data['item_cd'].isin(list(self.item_code_map))].copy()
        data['code'] = [self.item_code_map[item] for item in data['item_cd']]
        data['schd_time'] = np.ceil(data['schd_time'])

        # Todo: need to correct code
        oper_map = dict(list(map(lambda x: (x[0], x[1]), data.groupby('code'))))

        return data, oper_map

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
        # due_date_min = data[self.col_due_date].min()
        # data['days'] = data[self.due_date_col] - due_date_min
        data['days'] = data[self.col_due_date] - datetime.datetime.now()
        data['minutes'] = data['days'] / np.timedelta64(1, "m")

        return data
