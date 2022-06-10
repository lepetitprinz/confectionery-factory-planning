from common.name import Key, Demand, Item, Resource, Constraint, Post

import datetime as dt
import pandas as pd
from typing import Union, Tuple


class Mold(object):
    def __init__(self, plant, plant_start_time, mold_capa_cstr, item, route):
        self.plant = plant
        self.plant_start_time = plant_start_time
        self.mold_capa_cstr = mold_capa_cstr

        # Name instance attribute
        self._key = Key()
        self._item = Item()
        self._dmd = Demand()
        self._res = Resource()
        self._cstr = Constraint()

        # Dataset
        self.route = route
        self.item_halb = item

        # Time
        self.days = 60
        self.day_second = 86400
        self.time_interval = []

        # Column usage
        self._weight_map = {'G': 1, 'KG': 1000, 'TON': 1000000}
        self._col_item = [self._item.sku, self._item.item_type, self._item.weight, self._item.weight_uom]

    def apply(self, data: pd.DataFrame):
        data = self.preprocess(data=data)
        apply_dmd, non_apply_dmd = self.classify_cstr_apply(data=data)

        if len(apply_dmd) == 0:
            return data
        else:
            apply_dmd = self.slice_time_for_each_day(data=apply_dmd)
            for res_grp, res_grp_df in apply_dmd.groupby(by=self._res.res_grp):
                pass

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Preprocess the dataset
        item = self.prep_item()
        # route = self.prep_route()
        self.make_daily_time_interval()

        # Add item information
        merged = pd.merge(data, item, on=[self._item.sku], how='left').fillna('-')

        return merged

    def make_daily_time_interval(self):
        self.time_interval = [(i, i * self.day_second, (i + 1) * self.day_second) for i in range(self.days)]

    def classify_cstr_apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        halb_data = data[data[self._item.sku].isin(self.item_halb[self._item.sku])].copy()
        halb_res_grp = halb_data[self._res.res_grp].unique()

        apply_dmd = data[data[self._res.res_grp].isin(halb_res_grp)]
        non_apply_dmd = data[~data[self._res.res_grp].isin(halb_res_grp)]

        return apply_dmd, non_apply_dmd

    def slice_time_for_each_day(self, data: pd.DataFrame) -> pd.DataFrame:
        fert = data[data[self._item.item_type] == 'FERT']
        halb = data[data[self._item.item_type] == 'HALB']

        # Todo: Temporal conversion
        halb[self._item.weight] = 10
        halb['day'] = 0

        # Slice timeline of half-item
        for row in halb.iterrows():
            stime = row['starttime']
            etime = row['endtime']
            for day, day_start, day_end in self.time_interval:
                if stime >= day_start:
                    if etime <= day_end:
                        row['day'] = day


        return data

    def change_data_format(self, data: pd.DataFrame) -> pd.DataFrame:
        data[''] = [self.plant_start_time + dt.timedelta(seconds=sec) for sec in data['']]

        return data

    def prep_route(self):
        print("")

    def prep_item(self):
        item = self.item_halb[self._col_item]
        # item[self.item.weight] = [weight * self.weight_map[]]
        return item
