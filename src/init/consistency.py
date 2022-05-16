import common.config as config
from common.name import Key, Demand, Resource, Item

import os
import pandas as pd


class Consistency(object):
    def __init__(self, data: dict, path: str, verbose=False):
        self.log = []
        self.path = os.path.join(path, 'result', 'consistency')
        self.verbose = verbose

        # name instance attribute
        self.key = Key()
        self.dmd = Demand()
        self.res = Resource()
        self.item = Item()

        # Dataset
        self.demand = data[self.key.dmd]
        self.resource = data[self.key.res]
        self.constraint = data[self.key.cstr]

        self.plant_res_grp_res = {}
        self.plant_res_res_grp = {}
        self.apply_plant = config.apply_plant
        self.col_dmd = [self.res.plant, self.res.res_grp, self.item.sku]

    def run(self):
        # Make resource group to resource mapping
        self.make_plant_res_grp_res_map()

        #########################
        # Check resource dataset
        #########################
        # Check resource exist
        self.check_plant_res_existence()

        # Check that the resource exist on each demand
        self.check_dmd_res_existence()

        # Check that resource duration exist
        self.check_res_duration_existence()

    # Check if resource exists on each plant
    def check_plant_res_existence(self) -> None:
        # set resource dataset
        res_grp = self.resource[self.key.res_grp].copy()

        for plant in self.apply_plant:
            res_grp_by_plant = res_grp[res_grp[self.res.plant] == plant]
            if len(res_grp_by_plant) == 0:
                # Plant is removed from apply list
                self.apply_plant.remove(plant)

                # Error message
                msg = f"Warning: Resource dataset doesn't have Plant[{plant}] data."
                self.log.append(msg)

                if self.verbose:
                    print(msg)

    def check_dmd_res_existence(self) -> None:
        dmd = self.demand.copy()
        dmd = dmd[self.col_dmd].drop_duplicates()

        for plant, plant_df in dmd.groupby(by=self.res.plant):
            if plant in self.apply_plant:
                for res_grp, res_grp_df in plant_df.groupby(by=self.res.res_grp):
                    resource = self.plant_res_grp_res[plant].get(res_grp, None)
                    if resource is None:
                        # Error message
                        msg = f"Warning: Plant[{plant}] - Resource Group[{res_grp}] doesn't have resource data."
                        self.log.append(msg)

                        if self.verbose:
                            print(msg)

    def check_res_duration_existence(self):
        # set dataset
        dmd = self.demand.copy()
        dmd = dmd[self.col_dmd].drop_duplicates()

        res_dur_map = self.make_res_dur_map()
        for plant, plant_df in dmd.groupby(by=self.res.plant):
            for sku, sku_df in plant_df.groupby(by=self.item.sku):
                for res_grp in sku_df[self.res.res_grp]:
                    sku_to_res_list = res_dur_map[plant].get(sku, None)
                    if sku_to_res_list is None:
                        # Error message
                        msg = f"Warning: Plant[{plant}] - SKU[{sku}] doesn't have any resource duration data."
                        self.log.append(msg)

                        if self.verbose:
                            print(msg)
                    else:
                        res_list = sku_to_res_list.get(res_grp, None)
                        if res_list is None:
                            # Error message
                            msg = f"Warning: Plant[{plant}] - SKU[{sku}] - Resource Group[{res_grp}] doesn't have" \
                                  " any resource duration data."
                            self.log.append(msg)

    def make_res_dur_map(self):
        res_dur = self.resource[self.key.res_duration].copy()

        res_grp_list = []
        for plant, res in zip(res_dur[self.res.plant], res_dur[self.res.res]):
            res_to_res_grp = self.plant_res_res_grp.get(plant, None)
            if res_to_res_grp is not None:
                res_grp = res_to_res_grp.get(res, None)
                if res_grp is not None:
                    res_grp_list.append(res_grp)
                else:
                    res_grp_list.append('-')
            else:
                res_grp_list.append('-')

        res_dur[self.res.res_grp] = res_grp_list

        res_dur_map = {}
        for plant, plant_df in res_dur.groupby(by=self.res.plant):
            plant_sku = {}
            for sku, sku_df in plant_df.groupby(by=self.item.sku):
                sku_res_grp = {}
                for res_grp, res_grp_df in sku_df.groupby(by=self.res.res_grp):
                    for res in res_grp_df[self.res.res]:
                        if res_grp in sku_res_grp:
                            sku_res_grp[res_grp].append(res)
                        else:
                            sku_res_grp[res_grp] = [res]
                plant_sku[sku] = sku_res_grp
            res_dur_map[plant] = plant_sku

        return res_dur_map

    def make_plant_res_grp_res_map(self) -> None:
        data = self.resource[self.key.res_grp].copy()
        data = data[[self.res.plant, self.res.res_grp, self.res.res]].drop_duplicates()

        plant_res_grp_res = {}
        for plant, plant_df in data.groupby(by=self.res.plant):
            res_grp_res = {}
            for res_grp, res_grp_df in plant_df.groupby(by=self.res.res_grp):
                for res in res_grp_df[self.res.res]:
                    if res_grp in res_grp_res:
                        res_grp_res[res_grp].append(res)
                    else:
                        res_grp_res[res_grp] = [res]
            plant_res_grp_res[plant] = res_grp_res

        self.plant_res_grp_res = plant_res_grp_res

        # make resource to resource group map on each plant
        plant_res_res_grp = {}
        for plant, plant_dict in plant_res_grp_res.items():
            res_res_grp = {}
            for res_grp, res_list in plant_dict.items():
                for res in res_list:
                    res_res_grp[res] = res_grp
            plant_res_res_grp[plant] = res_res_grp

        self.plant_res_res_grp = plant_res_res_grp
