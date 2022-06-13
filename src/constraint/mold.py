import common.util as util
from common.name import Key, Demand, Item, Resource, Constraint, Post

import numpy as np
import pandas as pd
from typing import Tuple, List


class Mold(object):
    def __init__(self, plant, plant_start_time, cstr, route, res_dur, mold_capa_cstr, half_item):
        self.plant = plant
        self.plant_start_time = plant_start_time
        self.daily_mold_capa = mold_capa_cstr

        # Name instance attribute
        self._key = Key()
        self._post = Post()
        self._item = Item()
        self._dmd = Demand()
        self._res = Resource()
        self._cstr = Constraint()

        # Dataset
        self.route = route
        self.cstr_mst = cstr
        self.res_to_capa = {}
        self.res_dur = res_dur
        self.half_item = half_item

        # Time instance attribute
        self.days = 60
        self.day_second = 86400
        self.time_interval = []
        self.work_day = 5  # Monday ~ Friday
        self.sec_of_day = 86400  # Seconds of 1 day
        self.time_multiple = 60  # Minute -> Seconds
        self.schedule_weeks = 104
        self.plant_start_hour = 0

        # Column usage
        self._weight_map = {'G': 1, 'KG': 1000, 'TON': 1000000}
        self._col_item = [self._item.sku, self._item.item_type, self._item.weight, self._item.weight_uom]

    def apply(self, data: pd.DataFrame):
        # Preprcess the dataset
        data = self.preprocess(data=data)

        # Classify constraints appliance
        apply_dmd, non_apply_dmd = self.classify_cstr_apply(data=data)

        if len(apply_dmd) == 0:
            return data
        else:
            # Slice timeline by each day
            apply_dmd = self.slice_timeline_by_each_day(data=apply_dmd)

            # Add production weight information
            apply_dmd = self.add_weight(data=apply_dmd)
            apply_result = pd.DataFrame()
            for res_grp, res_grp_df in apply_dmd.groupby(by=self._res.res_grp):
                apply_res_grp = self.check_daily_capa(data=res_grp_df, res_grp=res_grp)
                apply_result = pd.concat([apply_result, apply_res_grp], axis=0)

            return pd.concat([non_apply_dmd, apply_result], axis=0)

    def check_daily_capa(self, data: pd.DataFrame, res_grp: str):
        daily_mold_capa = self.daily_mold_capa[res_grp]
        days = np.sort(data[data['day'] != '-']['day'].unique())
        for day in days:
            day_data = data[data['day'] == day].copy()
            day_capa = sum(day_data['tot_weight'])
            if day_capa <= daily_mold_capa:
                continue
            else:
                data = self.correct_dayily_prod(data=data, res_grp=res_grp, day=day, day_data=day_data, capa=day_capa)

        return data

    def correct_dayily_prod(self, data, res_grp, day, day_data, capa):
        weight_diff = capa - self.daily_mold_capa[res_grp]

        # Decide what resource to move
        res_to_move = self.decide_resource_move(data=data, day_data=day_data)

        # correct resource timeline
        apply_data = data[data[self._res.res] == res_to_move]
        non_apply_data = data[data[self._res.res] != res_to_move]
        apply_data = self.correct_res_timeline(
            data=apply_data,
            day=day,
            res=res_to_move,
            weight_diff=weight_diff
        )
        data = pd.concat([non_apply_data, apply_data], axis=0)

        return data

    def correct_res_timeline(self, data, day, res, weight_diff):
        day_res = data[data['day'] == day].copy()

        # Move timeline
        moved_data = self.move_timeline(data=data, day_data=day_res, day=day, res=res, weight_diff=weight_diff)

        return moved_data

    def move_timeline(self, data, day_data, day, res, weight_diff):
        move_dur = self.conv_weight_to_duration(data=day_data, weight=weight_diff)

        # split day resource capacity
        day_res_bf, day_res_af = day_data.copy(), day_data.copy()
        day_res_bf[self._dmd.start_time] = day_res_bf[self._dmd.start_time] + move_dur
        day_res_af[self._dmd.start_time] = day_res_bf[self._dmd.end_time]
        day_res_af[self._dmd.end_time] = day_res_af[self._dmd.start_time] + move_dur
        day_res_af['day'] = self.add_day(day=day)

        day_res_bf['tot_weight'] = day_res_bf['tot_weight'] - weight_diff
        day_res_af['tot_weight'] = weight_diff

        # Move next timeline
        comp_time = day_data[self._dmd.end_time].values[0]
        move_res = data[data[self._dmd.start_time] >= comp_time].copy()
        non_move_res = data[data[self._dmd.start_time] < comp_time].copy()

        move_res[self._dmd.end_time] = move_res[self._dmd.end_time] + move_dur
        move_res[self._dmd.start_time] = move_res[self._dmd.start_time] + move_dur

        # Apply resource capacity
        move_res = self.apply_res_capa(data=move_res, res=res)

        moved_data = pd.concat([day_res_bf, day_res_af, non_move_res, move_res], axis=0)

        return moved_data

    def apply_res_capa(self, data: pd.DataFrame, res: str):
        applied_data = pd.DataFrame()

        time_start = 0
        res_capa_list = self.res_to_capa[res]
        data = data.sort_values(by=self._dmd.start_time)
        for idx, start_time, end_time in zip(data.index, data[self._dmd.start_time], data[self._dmd.end_time]):
            if time_start > start_time:
                time_gap = time_start - start_time
                start_time, end_time = time_start, end_time + time_gap
            for capa_start, capa_end in res_capa_list:
                dmd = data.loc[idx].copy()
                if start_time < capa_end:
                    if start_time < capa_start:
                        gap = capa_start - start_time
                        start_time, end_time = start_time + gap, end_time + gap

                    running_time = end_time - start_time

                    if running_time <= capa_end - start_time:
                        dmd[self._dmd.start_time] = start_time
                        dmd[self._dmd.end_time] = end_time
                        dmd[self._dmd.duration] = end_time - start_time
                        applied_data = applied_data.append(dmd)
                        time_start = end_time
                        break
                    else:
                        dmd[self._dmd.start_time] = start_time
                        dmd[self._dmd.end_time] = capa_end
                        dmd[self._dmd.duration] = capa_end - start_time
                        applied_data = applied_data.append(dmd)
                        start_time = capa_end

        applied_data = applied_data.reset_index(drop=True)

        return applied_data

    def conv_weight_to_duration(self, data: pd.Series, weight):
        prod_qty = weight / data[self._item.weight]
        duration = prod_qty * data[self._dmd.duration]

        return duration.values[0]

    @staticmethod
    def add_day(day):
        if (day + 1) % 7 == 5:
            day_add = day + 3
        elif (day + 1) % 7 == 6:
            day_add = day + 2
        else:
            day_add = day + 1

        return day_add

    def decide_resource_move(self, data, day_data):
        res_list = day_data[self._res.res].unique()
        min_end_df = data[data[self._res.res].isin(res_list)]\
            .groupby(by=self._res.res)[self._dmd.end_time]\
            .max()\
            .reset_index()
        latest_fin_res = min_end_df[
            min_end_df[self._dmd.end_time] == min_end_df[self._dmd.end_time].min()
            ][self._res.res][0]

        return latest_fin_res

    def add_weight(self, data):
        data['diff'] = data[self._dmd.end_time] - data[self._dmd.start_time]
        data[self._dmd.duration] = [    # Capa use rate
            self.res_dur[sku][res] for sku, res in zip(data[self._item.sku], data[self._res.res])
        ]

        # Calculate Production quantity
        data[self._dmd.prod_qty] = data['diff'] / data[self._dmd.duration]
        data[self._item.weight] = [
            int(weight * self._weight_map[uom]) for weight, uom in
            zip(data[self._item.weight], data[self._item.weight_uom])]
        data['tot_weight'] = data[self._item.weight] * np.floor(data[self._dmd.prod_qty])

        data = data.drop(columns=['diff', self._item.weight_uom])

        return data

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Preprocess the dataset
        item = self.prep_item()
        # route = self.prep_route()
        self.make_daily_time_interval()

        self.set_res_capacity(data=self.cstr_mst[self._key.res_avail_time])

        # Add item information
        merged = pd.merge(data, item, on=[self._item.sku], how='left').fillna('-')

        return merged

    def make_daily_time_interval(self) -> None:
        self.time_interval = [(i, i * self.day_second, (i + 1) * self.day_second) for i in range(self.days)
                              if i % 7 not in [5, 6]]

    def classify_cstr_apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        halb_data = data[data[self._item.sku].isin(self.half_item[self._item.sku])].copy()
        halb_res_grp = halb_data[self._res.res_grp].unique()

        apply_dmd = data[data[self._res.res_grp].isin(halb_res_grp)]
        non_apply_dmd = data[~data[self._res.res_grp].isin(halb_res_grp)]

        return apply_dmd, non_apply_dmd

    def slice_timeline_by_each_day(self, data: pd.DataFrame) -> pd.DataFrame:
        fert = data[data[self._item.item_type] == 'FERT'].copy()
        halb = data[data[self._item.item_type] == 'HALB'].copy()

        # Todo: Temporal conversion
        halb[self._item.weight] = 5
        halb['day'] = 0

        # Slice timeline of half-item
        splited_list = []
        for i, row in halb.iterrows():
            splited = self.slice_timeline(row, splited=[])
            splited_list.extend(splited)

        halb_splited = pd.DataFrame(splited_list)
        merged = pd.concat([fert, halb_splited], axis=0).fillna('-')

        return merged

    def slice_timeline(self, row: pd.Series, splited: list) -> List[pd.Series]:
        stime = row[self._dmd.start_time]
        etime = row[self._dmd.end_time]
        for day, day_start, day_end in self.time_interval:
            if stime >= day_end:
                continue
            else:
                if etime <= day_end:
                    row['day'] = day
                    splited.append(row)

                    return splited
                else:
                    # split times
                    row_bf = row.copy()
                    row_bf[self._dmd.end_time] = day_end
                    row_bf['day'] = day
                    splited.append(row_bf)

                    row_af = row.copy()
                    row_af[self._dmd.start_time] = day_end
                    splited = self.slice_timeline(row=row_af, splited=splited)

                    return splited

    def prep_item(self):
        item = self.half_item[self._col_item]
        item[self._item.weight] = item[self._item.weight].astype(float)
        item[self._item.weight] = np.ceil(item[self._item.weight])

        return item

    def set_res_capacity(self, data: pd.DataFrame):
        # Choose current plant
        data = data[data[self._res.plant] == self.plant]

        capa_col_list = []
        for i in range(self.work_day):
            capa_col = self._res.res_capa + str(i + 1)
            capa_col_list.append(capa_col)

        res_to_capa = {}
        for res, capa_df in data.groupby(by=self._res.res):
            days_capa = capa_df[capa_col_list].values.tolist()[0]

            days_capa_list = []
            start_time, end_time = (self.plant_start_hour, self.plant_start_hour)
            for i, time in enumerate(days_capa * self.schedule_weeks):
                start_time, end_time = util.calc_daily_avail_time(
                    day=i, time=int(time) * self.time_multiple, start_time=start_time, end_time=end_time
                )
                days_capa_list.append([start_time, end_time])
                if i % 5 == 4:  # skip saturday & sunday
                    start_time += self.sec_of_day * 3

            days_capa_list = self.connect_continuous_capa(data=days_capa_list)
            res_to_capa[res] = days_capa_list

        self.res_to_capa = res_to_capa

    @staticmethod
    def connect_continuous_capa(data: list):
        result = []
        idx = 0
        add_idx = 1
        while idx + add_idx < len(data) + 1:
            curr_capa = data[idx]
            if idx + add_idx < len(data):
                next_capa = data[idx + add_idx]
                if curr_capa[1] == next_capa[0]:
                    curr_capa[1] = next_capa[1]
                    add_idx += 1
                else:
                    result.append(curr_capa)
                    idx += add_idx
                    add_idx = 1
            else:
                result.append(curr_capa)
                break

        return result
