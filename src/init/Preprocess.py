import datetime
import common.util as util

import numpy as np
import pandas as pd
import datetime as dt
# import networkx as nx
from typing import List, Tuple, Dict, Any


class Preprocess(object):
    # Time setting
    time_uom = 'sec'    # min / sec

    # Column name setting
    col_dmd = 'dmd_id'
    col_plant = 'plant_cd'
    col_sku = 'item_cd'
    col_res = 'res_cd'
    col_res_grp = 'res_grp_cd'
    col_res_map = 'res_map_cd'
    col_res_type = 'res_type_cd'
    col_capacity = 'capacity'
    col_capa_unit = 'capa_unit_cd'
    col_qty = 'qty'
    col_route_from = 'from_item'
    col_route_to = 'to_item'
    col_bom_lvl = 'bom_lvl'
    col_from_to_rate = 'rate'
    col_duration = 'duration'
    col_due_date = 'due_date'

    # Column usage setting
    use_col_dmd = ['dmd_id', 'item_cd', 'res_grp_cd', 'qty', 'due_date']
    use_col_res_grp = ['plant_cd', 'res_grp_cd', 'res_cd', 'capacity', 'capa_unit_cd','res_type_cd']
    use_col_res_people_map = ['plant_cd', 'res_grp_cd', 'res_cd', 'res_map_cd']
    use_col_item_res_duration = ['plant_cd', 'item_cd', 'res_grp_cd', 'duration']

    def __init__(self):
        # configuration
        self.plant_dmd_list = []
        self.plant_dmd_res_list = []
        self.plant_dmd_item = {}
        self.item_code_map = {}
        self.item_code_rev_map = {}
        self.oper_map = {}
        self.res_map = {}

    def calc_due_date(self, data):
        data[self.col_due_date] = data[self.col_due_date] * 24 * 60 * 60
        data[self.col_due_date] = data[self.col_due_date].astype(int)

        return data

    def set_dmd_info(self, data: pd.DataFrame) -> dict:
        # Get plant list of demand list
        plant_dmd_list = list(set(data[self.col_plant]))

        self.calc_due_date(data=data)
        # data = self.calc_deadline(data=data)

        # Change data type
        # data[self.col_due_date] = pd.to_datetime(data[self.col_due_date])
        data[self.col_sku] = data[self.col_sku].astype(str)
        data[self.col_res_grp] = data[self.col_res_grp].astype(str)
        data[self.col_qty] = np.ceil(data[self.col_qty]).astype(int)    # Ceiling quantity

        # Group demand by each plant
        plant_dmd, plant_dmd_item, plant_dmd_res, plant_dmd_due = {}, {}, {}, {}
        for plant in plant_dmd_list:
            # Filter data by each plant
            dmd = data[data[self.col_plant] == plant]

            # Convert form of demand dataset
            plant_dmd[plant] = self.convert_dmd_form(data=dmd)

            # Get resource group only contained in demand
            plant_dmd_res[plant] = list(set(dmd[self.col_res_grp].values))

            # All of demand item list by plant
            plant_dmd_item[plant] = list(set(dmd[self.col_sku]))

            # Set demand due date by plant
            plant_dmd_due[plant] = self.set_plant_dmd_due(data=dmd)

        self.plant_dmd_list = plant_dmd_list
        self.plant_dmd_item = plant_dmd_item
        self.plant_dmd_res_list = plant_dmd_res

        dmd_prep = {
            'plant_dmd_list': plant_dmd,
            'plant_dmd_item': plant_dmd_item,
            'plant_dmd_due': plant_dmd_due

        }

        return dmd_prep

    def set_res_info(self, data: dict) -> dict:
        plant_res_grp = self.set_res_grp(data=data['res_grp'])
        # plant_res_human = self.set_res_grp(data=data['res_human'])
        # plant_res_human_map = self.set_res_human_map(data=data['res_human_map'])
        plant_item_res_grp_duration = self.set_item_res_duration(data=data['item_res_duration'])

        res_prep = {
            'plant_res_grp': plant_res_grp,
            # 'plant_res_human': plant_res_human,
            # 'plant_res_human_map': plant_res_human_map,
            'plant_item_res_grp_duration': plant_item_res_grp_duration,
        }

        return res_prep

    def convert_dmd_form(self, data: pd.DataFrame) -> List[Tuple[Any]]:
        data_use = data[self.use_col_dmd].copy()
        data_tuple = [tuple(row) for row in data_use.to_numpy()]

        return data_tuple

    def set_plant_dmd_due(self, data: pd.DataFrame):
        temp = data[[self.col_dmd, self.col_sku, self.col_due_date]].copy()
        plant_dmd_due = {}
        for demand, group in temp.groupby(self.col_dmd):
            plant_dmd_due[demand] = group[[self.col_sku, self.col_due_date]]\
                .set_index(self.col_sku)\
                .to_dict()[self.col_due_date]

        return plant_dmd_due

    def set_res_grp(self, data) -> dict:
        # Rename columns
        data = data.rename(columns={'res_capa_val': self.col_capacity})

        # Choose columns used in model
        data = data[self.use_col_res_grp].copy()

        # Change data type
        data[self.col_res] = data[self.col_res].astype(str)
        data[self.col_res_grp] = data[self.col_res_grp].astype(str)
        data[self.col_capacity] = data[self.col_capacity].astype(int)

        # Choose plants of demand list
        data = data[data[self.col_plant].isin(self.plant_dmd_list)].copy()

        # Group resource by each plant
        res_grp_by_plant = {}
        for plant in self.plant_dmd_list:
            res_grp_df = data[data[self.col_plant] == plant]
            # Filter
            res_grp_df = res_grp_df[res_grp_df[self.col_res_grp].isin(self.plant_dmd_res_list[plant])]

            # Resource group -> (resource / capacity / resource type)
            res_grp_to_res = {}
            for res_grp, group in res_grp_df.groupby(self.col_res_grp):
                res_grp_to_res[res_grp] = [tuple(row) for row in group[
                    [self.col_res, self.col_capacity, self.col_capa_unit, self.col_res_type]].to_numpy()]

            res_grp_by_plant[plant] = res_grp_to_res

        return res_grp_by_plant

    def set_res_human_map(self, data: pd.DataFrame):
        # Choose columns used in model
        data = data[self.use_col_res_people_map].copy()

        # Change data type
        data[self.col_res] = data[self.col_res].astype(str)
        data[self.col_res_grp] = data[self.col_res_grp].astype(str)
        data[self.col_res_map] = data[self.col_res_map].astype(str)

        # Choose plants of demand list
        data = data[data[self.col_plant].isin(self.plant_dmd_list)].copy()

        # Group resource by each plant
        res_human_by_plant = {}
        for plant in self.plant_dmd_list:
            res_human_df = data[data[self.col_plant] == plant]

            res_to_human = {}
            for res_grp, group in res_human_df.groupby(self.col_res_grp):
                res_to_human[res_grp] = group\
                    .groupby(by=self.col_res_map)[self.col_res]\
                    .apply(list)\
                    .to_dict()

            # res_grp_list = list(set(res_human_df[self.col_res_grp].values))
            # for res_grp in res_grp_list:
            #     res_people_filtered = res_human_df[res_human_df[self.col_res_grp] == res_grp].copy()
            #     res_to_human[res_grp] = res_people_filtered\
            #         .groupby(by=self.col_res_map)[self.col_res]\
            #         .apply(list)\
            #         .to_dict()
            res_human_by_plant[plant] = res_to_human

        return res_human_by_plant

    def set_item_res_duration(self, data: pd.DataFrame):
        # Choose columns used in model
        data = data[self.use_col_item_res_duration].copy()

        # Change data type
        data[self.col_res_grp] = data[self.col_res_grp].astype(str)
        data[self.col_sku] = data[self.col_sku].astype(str)
        data[self.col_duration] = data[self.col_duration].astype(int)

        # Group bom route by each plant
        item_res_duration_by_plant = {}
        for plant in self.plant_dmd_list:
            # Filter data contained in each plant
            item_res_duration = data[data[self.col_plant] == plant].copy()

            # Filter items of each plant involved in demand
            item_res_duration = item_res_duration[item_res_duration[self.col_sku].isin(self.plant_dmd_item[plant])]

            # item -> resource group -> duration mapping
            item_res_grp_duration_map = {}
            for item, group in item_res_duration.groupby(self.col_sku):
                item_res_grp_duration_map[item] = group[[self.col_res_grp, self.col_duration]]\
                    .set_index(self.col_res_grp)\
                    .to_dict()[self.col_duration]

            item_res_duration_by_plant[plant] = item_res_grp_duration_map

        return item_res_duration_by_plant

    def calc_deadline(self, data):
        # due_date_min = data[self.col_due_date].min()
        # data['days'] = data[self.due_date_col] - due_date_min
        days = data[self.col_due_date] - datetime.datetime.now()   # ToDo : need to revise start day

        due_date = None
        if self.time_uom == 'min':
            due_date = np.round(days / np.timedelta64(1, 'm'), 0)
        elif self.time_uom == 'sec':
            due_date = np.round(days / np.timedelta64(1, 's'), 0)

        data[self.col_due_date] = due_date.astype(int)

        return data

    # def set_bom_route_info(self, data: pd.DataFrame):
    #     # rename columns
    #     data = data.rename(columns={'child_item': self.col_route_from, 'parent_item': self.col_route_to})
    #
    #     # Change data type
    #     data[self.col_route_from] = data[self.col_route_from].astype(str)
    #     data[self.col_route_to] = data[self.col_route_to].astype(str)
    #     data[self.col_bom_lvl] = data[self.col_bom_lvl].astype(int)
    #     data[self.col_from_to_rate] = np.ceil(data[self.col_from_to_rate])
    #
    #     # Choose plants of demand list
    #     data = data[data[self.col_plant].isin(self.plant_dmd_list)].copy()
    #
    #     # Group bom route by each plant
    #     bom_route_by_plant = {}
    #     for plant in self.plant_dmd_list:
    #         # Filter data contained in each plant
    #         bom_route = data[data[self.col_plant] == plant].copy()
    #
    #         # Filter only items of each plant involved in demand
    #         bom_route = bom_route[bom_route[self.col_route_to].isin(self.plant_dmd_item[plant])]
    #
    #         # Generate item <-> code map
    #         bom_map = self.generate_item_code_map(data=bom_route)
    #
    #         # draw bom route graph
    #         route_graph = self.draw_graph(data=bom_route, data_map=bom_map)
    #
    #         bom_route_by_plant[plant] = {
    #             'map': bom_map,
    #             'graph': route_graph
    #         }
    #
    #     return bom_route_by_plant

    # def make_bom_route(self, bom: pd.DataFrame):
    #     _, bom_route = nx.DiGraph(), nx.DiGraph()
    #
    #     for bom_from, bom_to, rate in zip(bom['from'], bom['to'], bom['rate']):
    #         _.add_edge(bom_from, bom_to, rate=rate)
    #         bom_route.add_edge(self.item_code_map[bom_from], self.item_code_map[bom_to], rate=rate)
    #
    #     return bom_route

    # def draw_graph(self, data: pd.DataFrame, data_map):
    #     _, route_graph = nx.DiGraph(), nx.DiGraph()
    #
    #     for route_from, route_to, rate in zip(
    #             data[self.col_route_from],
    #             data[self.col_route_to],
    #             data[self.col_from_to_rate]
    #     ):
    #         _.add_edge(route_from, route_to, rate=rate)
    #         route_graph.add_edge(
    #             data_map['item_to_code'][route_from],
    #             data_map['item_to_code'][route_to],
    #             rate=rate)
    #
    #     return route_graph