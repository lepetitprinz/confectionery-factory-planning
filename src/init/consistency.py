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
        self.apply_plant = config.apply_plant
        self.col_dmd = [self.res.plant, self.res.res_grp, self.item.sku]

    def run(self):
        self.make_plant_res_grp_res_map()

        # Resource
        # Check that resource exist
        self.check_plant_res_existence()

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
                msg = f"Warning: Resource dataset doesn't have plant [{plant}] information."
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
                        msg = f"Warning: Plant {plant} - Resource Group {res_grp} doesn't have resource information."
                        self.log.append(msg)

                        if self.verbose:
                            print(msg)

    def check_res_duration_existence(self):
        # set dataset
        dmd = self.demand.copy()
        dmd = dmd[self.col_dmd].drop_duplicates()

        res_dur = self.resource[self.key.res_duration].copy()

        for plant, plant_df in dmd.groupby(by=self.res.plant):
            for res_grp, res_grp_df in plant_df.groupby(by=self.res.res_grp):
                for sku in res_grp_df[self.item.sku]:
                    pass

    def make_plant_res_grp_res_map(self) -> None:
        data = self.resource[self.key.res_grp].copy()
        data = data[[self.res.plant, self.res.res_grp, self.res.res]].drop_duplicates()

        plant_res_grp_res = {}
        for plant, plant_df in data.groupby(by=self.res.plant):
            plant_res_grp = {}
            for res_grp, res_grp_df in plant_df.groupby(by=self.res.res_grp):
                res_grp_res = {}
                for res in res_grp_df[self.res.res]:
                    if res_grp in res_grp_res:
                        res_grp_res[res_grp].append(res)
                    else:
                        res_grp_res[res_grp] = [res]
                plant_res_grp[res_grp] = res_grp_res
            plant_res_grp_res[plant] = plant_res_grp

        self.plant_res_grp_res = plant_res_grp_res
