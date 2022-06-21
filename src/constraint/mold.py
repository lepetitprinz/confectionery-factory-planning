import common.util as util
import common.config as config
from common.name import Key, Demand, Item, Resource, Constraint, Post

import numpy as np
import pandas as pd
from typing import Tuple, List, Hashable


class Mold(object):
    def __init__(self, plant, plant_start_time, cstr, route, res_dur, mold_cstr, res_to_res_grp):
        # Name instance attribute
        self._key = Key()
        self._post = Post()
        self._item = Item()
        self._dmd = Demand()
        self._res = Resource()
        self._cstr = Constraint()

        self.plant = plant
        self.plant_start_time = plant_start_time
        self.mold_apply_res_grp = []

        # Dataset & hash map
        self.route = route
        self.cstr_mst = cstr
        self.res_dur = res_dur
        self.res_to_capa = {}
        self.res_to_res_grp = res_to_res_grp

        # Mold constraint instance attribute
        self.mold_res = mold_cstr[self._key.mold_res].get(plant, None)
        self.mold_capa = mold_cstr[self._key.mold_capa].get(plant, None)
        self.item_weight = mold_cstr[self._key.item_weight]

        # Time instance attribute
        self.work_day = config.work_day    # 6 days: Monday ~ Friday
        self.sec_of_day = config.sec_of_day    # Seconds of 1 day
        self.time_multiple = config.time_multiple    # Minute -> Seconds
        self.time_interval = []
        self.schedule_weeks = config.schedule_weeks
        self.plant_start_hour = config.plant_start_hour

        # Column usage
        self._weight_map = {'G': 0.001, 'KG': 1, 'TON': 1000}
        self._col_item = [self._item.sku, self._item.item_type, self._item.weight, self._item.weight_uom]

    def apply(self, data: pd.DataFrame):
        # Preprocess the dataset
        self.preprocess(data=data)

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

    def check_daily_capa(self, data: pd.DataFrame, res_grp: Hashable):
        daily_mold_capa = self.daily_mold_capa[res_grp]
        last_day = int(data[data['day'] != '-']['day'].max() + 28)
        for day in range(0, last_day + 1):
            day_data = data[(data['day'] == day) & (data[self._item.item_type] == 'HALB')].copy()
            if len(day_data) > 0:
                day_capa = sum(day_data['tot_weight'])
                if day_capa > daily_mold_capa:
                    data = self.correct_daily_prod(
                        data=data,
                        res_grp=res_grp,
                        day=day,
                        day_data=day_data,
                        capa=day_capa
                    )
                else:
                    data = self.add_surplus_prod(
                        data=data,
                        res_grp=res_grp,
                        day=day,
                        day_data=day_data,
                        capa=day_capa
                    )

        return data

    def add_surplus_prod(self, data, res_grp, day, day_data, capa) -> pd.DataFrame:
        pass

    def update_day(self, data: pd.DataFrame) -> pd.DataFrame:
        data['day'] = [self.check_day(time=stime) for stime in data[self._dmd.start_time]]

        return data

    def check_day(self, time):
        for i, stime, etime in self.time_interval:
            if stime <= time < etime:
                return i

    def correct_daily_prod(self, data: pd.DataFrame, res_grp: Hashable, day, day_data, capa: float) -> pd.DataFrame:
        weight_diff = capa - self.daily_mold_capa[res_grp]

        # Decide what resource to move
        res_to_move = self.decide_resource_move(data=data, day_data=day_data, weight_diff=weight_diff)

        # Correct the resource timeline
        for res, weight in res_to_move:
            apply_data = data[data[self._res.res] == res]
            non_apply_data = data[data[self._res.res] != res]
            apply_data = self.correct_res_timeline(
                data=apply_data,
                day=day,
                res=res,
                weight_diff=weight
            )
            data = pd.concat([non_apply_data, apply_data], axis=0)
            data = self.update_day(data=data)
            data = self.connect_continuous_dmd(data=data)

        return data

    # Correct the resource timeline
    def correct_res_timeline(self, data, day, res, weight_diff) -> pd.DataFrame:
        day_res = data[data['day'] == day].copy()

        # Move timeline
        moved_data = self.move_timeline(data=data, day_data=day_res, res=res, weight_diff=weight_diff)
        moved_data = moved_data.sort_values(by=self._dmd.start_time).reset_index(drop=True)

        return moved_data

    def move_timeline(self, data, day_data, res, weight_diff) -> pd.DataFrame:
        move_dur = self.conv_weight_to_duration(
            weight_unit=day_data[self._item.weight],
            weight=weight_diff,
            capa_use_rate=day_data['capa_use_rate']
        )

        # split day resource capacity
        day_res_bf, day_res_af = day_data.copy(), day_data.copy()
        day_res_bf[self._dmd.start_time] = day_res_bf[self._dmd.start_time] + move_dur
        day_res_af[self._dmd.start_time] = day_res_bf[self._dmd.end_time]
        day_res_af[self._dmd.end_time] = day_res_af[self._dmd.start_time] + move_dur
        # day_res_af['day'] = self.add_day(day=day)

        day_res_bf['tot_weight'] = day_res_bf['tot_weight'] - weight_diff
        day_res_af['tot_weight'] = weight_diff

        # Move next timeline
        move_res = data[data[self._dmd.start_time] >= day_data[self._dmd.end_time].values[0]].copy()\
            .reset_index(drop=True)
        non_move_res = data[data[self._dmd.end_time] <= day_data[self._dmd.start_time].values[0]].copy()\
            .reset_index(drop=True)

        move_res[self._dmd.end_time] = move_res[self._dmd.end_time] + move_dur
        move_res[self._dmd.start_time] = move_res[self._dmd.start_time] + move_dur

        # Apply resource capacity
        move_res = self.apply_res_capa(data=move_res, res=res)

        moved_data = pd.concat([day_res_bf, day_res_af, non_move_res, move_res], axis=0).reset_index(drop=True)
        moved_data[self._dmd.start_time] = moved_data[self._dmd.start_time].astype(int)
        moved_data[self._dmd.end_time] = moved_data[self._dmd.end_time].astype(int)

        return moved_data

    def apply_res_capa(self, data: pd.DataFrame, res: str) -> pd.DataFrame:
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
                        # dmd[self._dmd.duration] = end_time - start_time
                        applied_data = applied_data.append(dmd)
                        time_start = end_time
                        break
                    else:
                        dmd[self._dmd.start_time] = start_time
                        dmd[self._dmd.end_time] = capa_end
                        # dmd[self._dmd.duration] = capa_end - start_time
                        applied_data = applied_data.append(dmd)
                        start_time = capa_end

        applied_data = applied_data.reset_index(drop=True)

        return applied_data

    @staticmethod
    def conv_weight_to_duration(weight, weight_unit, capa_use_rate):
        duration = weight / weight_unit * capa_use_rate
        return duration.values[0]

    @staticmethod
    def conv_duration_to_weight(duration, weight_unit, capa_use_rate):
        return duration * weight_unit / capa_use_rate

    def decide_resource_move(self, data, day_data, weight_diff) -> list:
        # filter resource weight that is bigger than over-weight
        day_data_filter = day_data[day_data['tot_weight'] >= weight_diff].copy()

        res_move_list = []
        if len(day_data_filter) > 0:
            res_list = day_data_filter[self._res.res].unique()
            min_end_df = data[data[self._res.res].isin(res_list)]\
                .groupby(by=self._res.res)[self._dmd.end_time]\
                .max()\
                .reset_index()
            latest_fin_res = min_end_df[
                min_end_df[self._dmd.end_time] == min_end_df[self._dmd.end_time].min()
                ][self._res.res].values[0]
            res_move_list = [[latest_fin_res, weight_diff]]
        else:
            res_list = day_data[self._res.res].unique()
            res_order = data[data[self._res.res].isin(res_list)]\
                .groupby(by=self._res.res)[self._dmd.end_time]\
                .max()\
                .reset_index()\
                .sort_values(by=self._dmd.end_time)[self._res.res].values

            for res in res_order:
                res_day_data = day_data[day_data[self._res.res] == res]

                # Summation of weight on resource
                res_day_weight = res_day_data['tot_weight'].sum()
                res_day_data = res_day_data.sort_values(by=self._dmd.start_time)
                first_data = res_day_data.iloc[0]    # 

                # Convert duration to weight
                weight = self.conv_duration_to_weight(
                    duration=(first_data['day'] + 1) * self.sec_of_day - first_data[self._dmd.start_time],
                    weight_unit=first_data[self._item.weight],
                    capa_use_rate=first_data['capa_use_rate']
                )

                if res_day_weight >= weight_diff:
                    res_move_list.append([res, weight-res_day_weight + weight_diff])
                    break
                else:
                    res_move_list.append([res, weight])
                    weight_diff -= res_day_weight

        return res_move_list

    def move_timeline_backward(self, data):
        data = data.sort_values(by=self._dmd.start_time)
        first_data = data.iloc[0]
        other_data = data.iloc[1:]

    def add_weight(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self._dmd.duration] = data[self._dmd.end_time] - data[self._dmd.start_time]
        data['capa_use_rate'] = [    # Capa use rate
            self.res_dur[sku][res] for sku, res in zip(data[self._item.sku], data[self._res.res])
        ]

        # Calculate Production quantity
        data[self._dmd.prod_qty] = data[self._dmd.duration] / data['capa_use_rate']
        data['tot_weight'] = data['mold_weight'] * np.floor(data[self._dmd.prod_qty])

        return data

    def preprocess(self, data: pd.DataFrame) -> None:
        # Preprocess mold data
        self.prep_mold()

        # route = self.prep_route()
        self.make_daily_time_interval()

        self.set_res_capacity(data=self.cstr_mst[self._key.res_avail_time])

        # Add item information
        # merged = pd.merge(data, item, on=[self._item.sku], how='left').fillna('-')

        # return merged

    def prep_mold(self):
        self.mold_apply_res_grp = list(set([self.res_to_res_grp[res] for res in self.mold_res
                                            if res in self.res_to_res_grp]))

    def make_daily_time_interval(self) -> None:
        self.time_interval = [(i, i * self.sec_of_day, (i + 1) * self.sec_of_day) for i in range(self.schedule_weeks)]

    def classify_cstr_apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        flag, mold_res, weight = [], [], []
        for res, sku in zip(data[self._res.res], data[self._item.sku]):
            sku_exist = self.mold_res.get(res, None)
            if sku_exist is None:
                flag.append(False)
            else:
                mold_res_weight = sku_exist.get(sku, None)
                if mold_res_weight is None:
                    flag.append(False)
                else:
                    flag.append(True)
                    mold_res.append(mold_res_weight[0])
                    weight.append(mold_res_weight[1])
        data['apply_flag'] = flag

        apply_dmd = data[data['apply_flag']]
        apply_dmd['mold_res'] = mold_res
        apply_dmd['mold_weight'] = weight
        non_apply_dmd = data[~data['apply_flag']]

        apply_dmd = apply_dmd.drop(columns='apply_flag')
        non_apply_dmd = non_apply_dmd.drop(columns='apply_flag')

        return apply_dmd, non_apply_dmd

    def slice_timeline_by_each_day(self, data: pd.DataFrame) -> pd.DataFrame:
        # Slice timeline of half-item
        splited_list = []
        for i, row in data.iterrows():
            splited = self.slice_timeline(row, splitted=[])
            splited_list.extend(splited)

        data_splite = pd.DataFrame(splited_list)

        return data_splite

    def slice_timeline(self, row: pd.Series, splitted: list) -> List[pd.Series]:
        stime = row[self._dmd.start_time]
        etime = row[self._dmd.end_time]
        for day, day_start, day_end in self.time_interval:
            if stime >= day_end:
                continue
            else:
                if etime <= day_end:
                    row['day'] = day
                    splitted.append(row)

                    return splitted
                else:
                    # split times
                    row_bf = row.copy()
                    row_bf[self._dmd.end_time] = day_end
                    row_bf['day'] = day
                    splitted.append(row_bf)

                    row_af = row.copy()
                    row_af[self._dmd.start_time] = day_end
                    splitted = self.slice_timeline(row=row_af, splitted=splitted)

                    return splitted

    def set_res_capacity(self, data: pd.DataFrame) -> None:
        # Choose current plant
        data = data[data[self._res.plant] == self.plant]

        capa_col_list = []
        for i in range(self.work_day):
            for kind in ['d', 'n']:
                capa = self._res.res_capa + str(i + 1) + '_' + kind
                capa_col_list.append(capa)

        res_to_capa = {}
        for res, capa_df in data.groupby(by=self._res.res):
            days_capa = capa_df[capa_col_list].values.tolist()[0]
            days_capa = util.make_time_pair(data=days_capa)

            days_capa_list = []
            for i, (day_time, night_time) in enumerate(days_capa * self.schedule_weeks):
                start_time, end_time = util.calc_daily_avail_time(
                    day=i, day_time=int(day_time), night_time=int(night_time))
                if start_time != end_time:
                    days_capa_list.append([start_time, end_time])

            days_capa_list = self.connect_continuous_capa(data=days_capa_list)
            if len(days_capa_list) > 0:
                res_to_capa[res] = days_capa_list

        self.res_to_capa = res_to_capa

    @staticmethod
    def connect_continuous_capa(data: list) -> list:
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

    @staticmethod
    def connect_continuous_capa_weight(data: list) -> list:
        result = []
        idx = 0
        add_idx = 1
        while idx + add_idx < len(data) + 1:
            curr_capa = data[idx]
            if idx + add_idx < len(data):
                next_capa = data[idx + add_idx]
                if curr_capa[1] == next_capa[0]:
                    curr_capa[1] = next_capa[1]
                    curr_capa[2] += next_capa[2]
                    add_idx += 1
                else:
                    result.append(curr_capa)
                    idx += add_idx
                    add_idx = 1
            else:
                result.append(curr_capa)
                break

        return result

    def connect_continuous_dmd(self, data: pd.DataFrame):
        revised_data = pd.DataFrame()

        for dmd, dmd_df in data.groupby(by=self._dmd.dmd):
            for item, item_df in dmd_df.groupby(by=self._item.sku):
                for res_grp, res_grp_df in item_df.groupby(by=self._res.res_grp):
                    for res, res_df in res_grp_df.groupby(by=self._res.res):
                        for day, day_df in res_df.groupby(by='day'):
                            if len(day_df) > 1:
                                # Sort demand on start time
                                day_df = day_df.sort_values(by=self._dmd.start_time)
                                timeline_list = day_df[
                                    [self._dmd.start_time, self._dmd.end_time, 'tot_weight']
                                ].values.tolist()

                                # Connect the capacity if timeline is continuous
                                timeline_connected = self.connect_continuous_capa_weight(data=timeline_list)

                                # Remake demand dataframe using connected timeline
                                for stime, etime, weight in timeline_connected:
                                    common = day_df.iloc[0].copy()
                                    dmd_series = self.update_connected_timeline(
                                        common=common,
                                        dmd=dmd,
                                        item=item,
                                        res_grp=res_grp,
                                        res=res,
                                        start_time=stime,
                                        end_time=etime,
                                        weight=weight,
                                    )
                                    revised_data = revised_data.append(dmd_series)
                            else:
                                revised_data = revised_data.append(day_df)

        revised_data = revised_data.sort_values(by=self._dmd.dmd)
        revised_data = revised_data.reset_index(drop=True)

        return revised_data

    def update_connected_timeline(self, common: pd.Series, dmd, item, res_grp, res, start_time, end_time, weight):
        common[self._dmd.dmd] = dmd
        common[self._item.sku] = item
        common[self._res.res] = res
        common[self._res.res_grp] = res_grp
        common[self._dmd.start_time] = start_time
        common[self._dmd.end_time] = end_time
        common['tot_weight'] = weight

        return common
