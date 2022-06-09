from common.name import Key, Demand, Item, Resource, Constraint, Post

import pandas as pd
from typing import Union, Tuple


class Mold(object):
    def __init__(self, plant, plant_start_time, mold_capa_cstr, item, route):
        self.plant = plant
        self.plant_start_time = plant_start_time
        self.mold_capa_cstr = mold_capa_cstr

        # Name instance attribute
        self._key = Key()
        self._dmd = Demand()
        self._res = Resource()
        self._item = Item()
        self._cstr = Constraint()

        # Dataset
        self.item_halb = item
        self.route = route

        # Column usage
        self._col_item = [self._item.sku, self._item.weight, self._item.weight_uom]
        self._weight_map = {'G': 1, 'KG': 1000, 'TON': 1000000}

    def apply(self, data: pd.DataFrame):
        self.preprocess()
        apply_dmd, non_apply_dmd = self.classify_cstr_apply(data=data)

        if len(apply_dmd) == 0:
            return data
        else:
            apply_dmd = self.change_data_format(data=apply_dmd)
            for res_grp, res_grp_df in apply_dmd.groupby(by=self._res.res_grp):
                pass

    def preprocess(self):
        # Preprocess the dataset
        route = self.prep_route()
        item = self.prep_item()

    def classify_cstr_apply(self, data) -> Tuple[pd.DataFrame, pd.DataFrame]:
        halb_data = data[data[self._item.sku].isin(self.item_halb[self._item.sku])].copy()
        halb_res_grp = halb_data[self._res.res_grp].unique()

        apply_dmd = data[data[self._res.res_grp].isin(halb_res_grp)]
        non_apply_dmd = data[~data[self._res.res_grp].isin(halb_res_grp)]

        return apply_dmd, non_apply_dmd

    def change_data_format(self, data: pd.DataFrame):
        data['']

    def prep_route(self):
        print("")

    def prep_item(self):
        item = self.item_halb[self._col_item]
        # item[self.item.weight] = [weight * self.weight_map[]]
