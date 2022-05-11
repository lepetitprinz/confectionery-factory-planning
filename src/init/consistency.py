import common.config as config

import os
import pandas as pd


class Consistency(object):
    ############################################
    # Apply plant list
    ############################################
    apply_plant = ['K110', 'K120', 'K130', 'K140', 'K170']

    ############################################
    # Dictionary key configuration
    ############################################
    key_dmd = config.key_dmd      # Demand
    key_res = config.key_res      # Resource
    key_item = config.key_item    # Item master
    key_cstr = config.key_cstr    # Constraint

    key_res_grp = config.key_res_grp          # Resource group code
    key_res_grp_nm = config.key_res_grp_nm    # Resource group name
    key_res_duration = config.key_res_duration    # Resource duration

    ############################################
    # Column name configuration
    ############################################
    # Demand
    col_dmd = config.col_dmd
    col_plant = config.col_plant
    col_qty = config.col_qty
    col_duration = config.col_duration
    col_due_date = config.col_due_date

    # Item
    col_sku = config.col_sku

    # Resource
    col_res = config.col_res
    col_res_grp = config.col_res_grp

    use_col_dmd = [col_plant, col_res_grp, col_sku]

    def __init__(self, data: dict, path: str, verbose=False):
        self.log = []
        self.path = os.path.join(path, 'result', 'consistency')
        self.verbose = verbose

        # Dataset
        self.demand = data[self.key_dmd]
        self.resource = data[self.key_res]
        self.constraint = data[self.key_cstr]

        self.plant_res_grp_res = {}

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
        res_grp = self.resource[self.key_res_grp].copy()

        for plant in self.apply_plant:
            res_grp_by_plant = res_grp[res_grp[self.col_plant] == plant]
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
        dmd = dmd[self.use_col_dmd].drop_duplicates()

        for plant, plant_df in dmd.groupby(by=self.col_plant):
            if plant in self.apply_plant:
                for res_grp, res_grp_df in plant_df.groupby(by=self.col_res_grp):
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
        dmd = dmd[self.use_col_dmd].drop_duplicates()

        res_dur = self.resource[self.key_res_duration].copy()

        for plant, plant_df in dmd.groupby(by=self.col_plant):
            for res_grp, res_grp_df in plant_df.groupby(by=self.col_res_grp):
                for sku in res_grp_df[self.col_sku]:
                    pass

    def make_plant_res_grp_res_map(self) -> None:
        data = self.resource[self.key_res_grp].copy()
        data = data[[self.col_plant, self.col_res_grp, self.col_res]].drop_duplicates()

        plant_res_grp_res = {}
        for plant, plant_df in data.groupby(by=self.col_plant):
            plant_res_grp = {}
            for res_grp, res_grp_df in plant_df.groupby(by=self.col_res_grp):
                res_grp_res = {}
                for res in res_grp_df[self.col_res]:
                    if res_grp in res_grp_res:
                        res_grp_res[res_grp].append(res)
                    else:
                        res_grp_res[res_grp] = [res]
                plant_res_grp[res_grp] = res_grp_res
            plant_res_grp_res[plant] = plant_res_grp

        self.plant_res_grp_res = plant_res_grp_res
