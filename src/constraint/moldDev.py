import common.util as util
import common.config as config
from common.name import Key, Demand, Item, Resource, Constraint, Post

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Tuple, List


class Mold(object):
    def __init__(self, plant, data, res_dur, mold_cstr, res_to_res_grp):
        # Name instance attribute
        self._key = Key()
        self._item = Item()
        self._dmd = Demand()
        self._res = Resource()
        self._cstr = Constraint()
        self._post = Post()

        self._plant = plant
        self._mold_apply_res_grp = []

        # Dataset & hash map
        self._item_mst = data[self._key.item]
        self._cstr_mst = data[self._key.cstr]
        self._res_dur = res_dur
        self._res_to_capa = {}
        self._res_day_capa = {}
        self._res_to_res_grp = res_to_res_grp

        # Mold constraint instance attribute
        self.weight_conv_map = {'G': 0.001, 'KG': 1, 'TON': 1000}
        self.mold_res = mold_cstr[self._key.mold_res].get(plant, None)
        self.mold_capa = mold_cstr[self._key.mold_capa].get(plant, None)

        # Time instance attribute
        self.work_day = config.work_day  # 6 days: Monday ~ Friday
        self.sec_of_day = config.sec_of_day  # Seconds of 1 day
        self.half_sec_of_day = self.sec_of_day // 2
        self.day_multiple = 3
        self.time_interval = []
        self.time_multiple = config.time_multiple  # Minute -> Seconds
        self.schedule_days = config.schedule_days
        self.time_idx_map = {'D': 0, 'N': 1}

        # Column information
        self._col_item_mst = [self._item.sku, self._item.weight, self._item.weight_uom]

        self.prev_fix_dmd = None

    def apply(self, data: pd.DataFrame):
        # Preprocess the dataset
        data = self.preprocess(data=data)

        # Classify constraints appliance
        apply_dmd, non_apply_dmd = self.classify_cstr_apply(data=data)

        if len(apply_dmd) == 0:
            return data
        else:
            # Add production weight information
            apply_dmd = self.add_weight(data=apply_dmd)

            # Slice timeline by each day
            apply_dmd = self.update_timeline_by_day_and_time_index(data=apply_dmd)

            mold_res_day = apply_dmd.groupby(by=self._cstr.mold_res).max()['day'].reset_index()
            mold_res_day['day'] = mold_res_day['day'].values * self.day_multiple

            result = pd.DataFrame()
            for mold_res in mold_res_day[self._cstr.mold_res]:
                mold_df = apply_dmd[apply_dmd[self._cstr.mold_res] == mold_res].copy()
                for day in range(0, mold_res_day[mold_res_day[self._cstr.mold_res] == mold_res]['day'].values[0]):
                    for time_idx in ['D', 'N']:
                        if len(mold_df) > 0:
                            mold_df, fix_dmd = self.apply_time_move(
                                data=mold_df,
                                mold_res=mold_res,
                                day=day,
                                time_idx=time_idx
                            )
                            result = pd.concat([result, fix_dmd], axis=0)

            result = pd.concat([result, non_apply_dmd], axis=0).reset_index(drop=True)
            # result = self.connect_continuous_dmd(data=result)
            result = result.drop(
                columns=[self._dmd.prod_qty, self._dmd.duration, self._cstr.mold_res, self._item.weight,
                         self._post.time_idx, 'capa_use_rate', 'tot_weight', 'day', 'fixed']
            )

            return result

    def apply_time_move(self, data, mold_res, day, time_idx):
        if self.check_daily_capa_excess(data=data, mold_res=mold_res, day=day, time_idx=time_idx):
            # print(f'resource: {mold_res}, day: {day}')
            data, fix_dmd = self.get_fix_dmd_list(
                data=data, mold_res=mold_res, day=day, time_idx=time_idx
            )
            self.prev_fix_dmd = fix_dmd
        else:
            data, fix_dmd = self.add_extra_capa_on_dmd(
                data=data,
                mold_res=mold_res,
                day=day,
                time_idx=time_idx
            )

        return data, fix_dmd

    def add_extra_capa_on_dmd(self, data: pd.DataFrame, mold_res, day, time_idx) -> Tuple[pd.DataFrame, pd.DataFrame]:
        mold_capa = self.mold_capa[mold_res][day % 7][self.time_idx_map[time_idx]]

        prev_fix_dmd = self.prev_fix_dmd.copy()
        if len(prev_fix_dmd) == 1:
            choose_dmd = data[data[self._dmd.dmd] == prev_fix_dmd[self._dmd.dmd]].copy().squeeze()
        else:
            prev_fix_last = prev_fix_dmd[prev_fix_dmd[self._dmd.dmd].isin(data[self._dmd.dmd])].copy()
            prev_fix_last = prev_fix_last[
                prev_fix_dmd[self._dmd.end_time] == prev_fix_last[self._dmd.end_time].max()
                ].squeeze()
            choose_dmd = data[data[self._dmd.dmd] == prev_fix_last[self._dmd.dmd]].copy().squeeze()

        choose_dmd['tot_weight'] = mold_capa
        choose_dmd[self._dmd.prod_qty] = round(choose_dmd['tot_weight'] / choose_dmd[self._item.weight], 3)
        choose_dmd[self._dmd.duration] = int(choose_dmd[self._dmd.prod_qty] * choose_dmd['capa_use_rate'])
        choose_dmd[self._dmd.end_time] = choose_dmd[self._dmd.start_time] + choose_dmd[self._dmd.duration]

        data = data.drop(index=choose_dmd.name)

        return data, choose_dmd.to_frame().T

    def move_timeline(self, data, fix_af_dmd, day, time_idx) -> pd.DataFrame:
        fix_res_data = data[data[self._res.res] == fix_af_dmd[self._res.res]].copy()
        other_data = data[data[self._res.res] != fix_af_dmd[self._res.res]].copy()

        # resource demand containing fixed resource
        fix_res_move_time = fix_af_dmd[self._dmd.end_time] - fix_res_data[self._dmd.start_time].min()
        fix_res_data[self._dmd.start_time] += fix_res_move_time
        fix_res_data[self._dmd.end_time] += fix_res_move_time

        # Outer demand
        base_time = self.time_interval[day][1]
        if time_idx == 'D':
            base_time += self.half_sec_of_day
        else:
            base_time += self.sec_of_day
        other_move_data = pd.DataFrame()
        for res, res_df in other_data.groupby(self._res.res):
            other_res_move_time = base_time - res_df[self._dmd.start_time].min()
            res_df[self._dmd.start_time] += other_res_move_time
            res_df[self._dmd.end_time] += other_res_move_time
            other_move_data = pd.concat([other_move_data, res_df], axis=0)

        result = pd.concat([fix_res_data, other_move_data], axis=0)

        return result

    def get_fix_dmd_list(self, data: pd.DataFrame, mold_res: str, day: int, time_idx: str) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        # get current capacity
        mold_capa = deepcopy(self.mold_capa[mold_res][day % 7][self.time_idx_map[time_idx]])

        fix_dmd_df = pd.DataFrame()
        fix_af_dmd = None
        while mold_capa != 0:
            day_time_data = data[(data['day'] == day) & (data[self._post.time_idx] == time_idx)].copy()
            data, fix_bf_dmd, fix_af_dmd, mold_capa = self.decide_which_dmd_fix(
                data=data,
                day_time_data=day_time_data,
                mold_capa=mold_capa
            )
            fix_dmd_df = fix_dmd_df.append(fix_bf_dmd)

        moved_dmd = self.move_timeline(data=data, fix_af_dmd=fix_af_dmd, day=day, time_idx=time_idx)
        moved_dmd = moved_dmd.append(fix_af_dmd)
        moved_dmd = moved_dmd.reset_index(drop=True)
        moved_dmd = self.apply_res_capa_on_timeline(data=moved_dmd)

        # Connect continuous timeline of each demand
        moved_dmd = self.connect_continuous_dmd(data=moved_dmd)

        # update timeline by day & time index
        moved_dmd = self.update_timeline_by_day_and_time_index(data=moved_dmd)

        return moved_dmd, fix_dmd_df

    def decide_which_dmd_fix(self, data, day_time_data, mold_capa):
        if len(day_time_data[day_time_data['fixed'] == 1]) > 0:
            fix_tick = day_time_data[day_time_data['fixed'] == 1].squeeze()
        else:
            # Get earliest start time
            earliest_start_time = day_time_data[self._dmd.start_time].min()
            earliest_dmd = day_time_data[(day_time_data[self._dmd.start_time] == earliest_start_time)]

            if len(earliest_dmd) == 1:
                fix_tick = earliest_dmd.squeeze()
            else:
                fix_tick = self.get_shortest_dur_demand(data, earliest_dmd)

        data = data.drop(index=fix_tick.name)

        data, fix_bf_dmd, fix_af_dmd, mold_capa = self.compare_curr_capa(
            data=data,
            fix_bf_dmd=fix_tick,
            mold_capa=mold_capa
        )

        return data, fix_bf_dmd, fix_af_dmd, mold_capa

    def compare_curr_capa(self, data: pd.DataFrame, fix_bf_dmd: pd.Series, mold_capa):
        fix_af_dmd = pd.DataFrame()
        if fix_bf_dmd['tot_weight'] < mold_capa:
            fix_bf_dmd['fixed'] = True
            mold_capa -= fix_bf_dmd['tot_weight']
            # move_time = 0
        else:
            data, fix_bf_dmd, fix_af_dmd = self.split_over_capa_dmd(data=data, fix_tick=fix_bf_dmd, mold_capa=mold_capa)
            mold_capa = 0

        return data, fix_bf_dmd, fix_af_dmd, mold_capa

    def split_over_capa_dmd(self, data: pd.DataFrame, fix_tick: pd.Series, mold_capa) \
            -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        fix_rate = round(mold_capa / fix_tick['tot_weight'], 3)

        fix_bf_dmd = fix_tick.copy()
        fix_bf_dmd['tot_weight'] = mold_capa
        fix_bf_dmd[self._dmd.prod_qty] = fix_tick[self._dmd.prod_qty] * fix_rate
        fix_bf_dmd[self._dmd.duration] = int(fix_tick[self._dmd.duration] * fix_rate)
        fix_bf_dmd[self._dmd.end_time] = fix_bf_dmd[self._dmd.start_time] + fix_bf_dmd[self._dmd.duration]
        fix_bf_dmd['fixed'] = True

        fix_af_dmd = fix_tick.copy()
        fix_af_dmd['tot_weight'] = fix_tick['tot_weight'] - mold_capa
        fix_af_dmd[self._dmd.prod_qty] = fix_tick[self._dmd.prod_qty] * (1 - fix_rate)
        fix_af_dmd[self._dmd.duration] = int(fix_tick[self._dmd.duration] * (1 - fix_rate))

        standard_time = self.time_interval[fix_tick['day']][self.time_idx_map[fix_tick[self._post.time_idx]] + 1]
        if fix_tick[self._post.time_idx] == 'D':
            fix_af_dmd[self._dmd.start_time] = standard_time + self.half_sec_of_day
        else:
            fix_af_dmd[self._dmd.start_time] = standard_time
        fix_af_dmd[self._dmd.end_time] = int(fix_af_dmd[self._dmd.start_time] + fix_af_dmd[self._dmd.duration])
        # fix_af_dmd['day'] = fix_af_dmd['day'] + 1 if fix_af_dmd[self._post.time_idx] == 'N' else fix_af_dmd['day']
        # fix_af_dmd[self._post.time_idx] = 'D' if fix_af_dmd[self._post.time_idx] == 'N' else 'N'
        fix_af_dmd['fixed'] = True

        dmd_df = data[data[self._dmd.dmd] == fix_tick[self._dmd.dmd]]
        if len(dmd_df) > 0:
            for duration in dmd_df[self._dmd.duration]:
                fix_af_dmd[self._dmd.end_time] += duration
                fix_af_dmd[self._dmd.duration] += duration

        fix_af_dmd[self._dmd.prod_qty] = np.floor(fix_af_dmd[self._dmd.duration] / fix_af_dmd['capa_use_rate'])
        fix_af_dmd['tot_weight'] = fix_af_dmd[self._item.weight] * np.floor(fix_af_dmd[self._dmd.prod_qty])
        fix_af_dmd['tot_weight'] = fix_af_dmd['tot_weight'].astype(int)

        data = data.drop(index=dmd_df.index)

        return data, fix_bf_dmd, fix_af_dmd

    def get_shortest_dur_demand(self, data: pd.DataFrame, earliest_dmd: pd.DataFrame) -> pd.Series:
        dmd_list = earliest_dmd[self._dmd.dmd].values
        res_dmd = data[data[self._dmd.dmd].isin(dmd_list)].copy()

        dmd_dur_series = res_dmd.groupby(self._dmd.dmd).sum()[self._dmd.duration]
        shortest_dmd = dmd_dur_series.index[dmd_dur_series.argmin()]
        fix_tick = earliest_dmd[earliest_dmd[self._dmd.dmd] == shortest_dmd].squeeze()

        return fix_tick

    def check_daily_capa_excess(self, data: pd.DataFrame, mold_res: str, day: int, time_idx: str):
        data = self.update_timeline_by_day_and_time_index(data=data)
        sliced = data[(data['day'] == day) & (data[self._post.time_idx] == time_idx)].copy()
        flag = False
        if len(sliced) > 0:
            if int(sliced.sum()['tot_weight']) > self.mold_capa[mold_res][day % 7][self.time_idx_map[time_idx]]:
                flag = True

        return flag

    def apply_res_capa_on_timeline(self, data: pd.DataFrame) -> pd.DataFrame:
        applied_data = pd.DataFrame()

        for res, res_df in data.groupby(self._res.res):
            res_df = res_df.sort_values(by=self._dmd.start_time)
            res_capa_list = self._res_to_capa[res]
            time_start = 0

            for idx, start_time, end_time in zip(res_df.index, res_df[self._dmd.start_time], res_df[self._dmd.end_time]):
                if time_start > start_time:
                    time_gap = time_start - start_time
                    start_time, end_time = time_start, end_time + time_gap
                for capa_start, capa_end in res_capa_list:
                    dmd = res_df.loc[idx].copy()
                    if start_time < capa_end:
                        if start_time < capa_start:
                            gap = capa_start - start_time
                            # start_time += gap
                            start_time, end_time = start_time + gap, end_time + gap

                        running_time = end_time - start_time

                        if running_time <= capa_end - start_time:
                            if dmd[self._dmd.duration] != 0:
                                split_rate = running_time / dmd[self._dmd.duration]
                            else:
                                split_rate = 0
                            dmd[self._dmd.duration] = end_time - start_time
                            dmd[self._dmd.prod_qty] = round(dmd[self._dmd.prod_qty] * split_rate)
                            dmd['tot_weight'] = round(dmd['tot_weight'] * split_rate)

                            dmd[self._dmd.start_time] = start_time
                            dmd[self._dmd.end_time] = end_time
                            # dmd[self._dmd.duration] = end_time - start_time
                            applied_data = applied_data.append(dmd)
                            time_start = end_time
                            break
                        else:
                            dmd[self._dmd.start_time] = start_time
                            dmd[self._dmd.end_time] = capa_end
                            dmd[self._dmd.duration] = capa_end - start_time

                            split_rate = dmd[self._dmd.duration] / running_time
                            dmd[self._dmd.prod_qty] = round(dmd[self._dmd.prod_qty] * split_rate)
                            dmd['tot_weight'] = round(dmd['tot_weight'] * split_rate)

                            applied_data = applied_data.append(dmd)
                            start_time = capa_end

            applied_data = applied_data.reset_index(drop=True)

        return applied_data

    def add_weight(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self._dmd.duration] = data[self._dmd.end_time] - data[self._dmd.start_time]
        data['capa_use_rate'] = [  # Capa use rate
            self._res_dur[sku][res] for sku, res in zip(data[self._item.sku], data[self._res.res])
        ]

        # Calculate Production quantity
        data[self._dmd.prod_qty] = np.floor(data[self._dmd.duration] / data['capa_use_rate'])
        data[self._dmd.prod_qty] = np.where(data[self._dmd.dmd].str.contains('@'), 0, data[self._dmd.prod_qty])
        data['tot_weight'] = data[self._item.weight] * np.floor(data[self._dmd.prod_qty])
        # data['tot_weight'] = data['mold_weight'] * np.floor(data[self._dmd.prod_qty])
        data['tot_weight'] = data['tot_weight'].astype(int)

        # Change job change
        data[self._dmd.prod_qty] = np.where(data[self._dmd.dmd].str.contains('@'), 0, data[self._dmd.prod_qty])
        data = data.reset_index(drop=True)

        return data

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data['fixed'] = False

        # Preprocess item master
        item = self.prep_item()

        # Preprocess mold data
        self.prep_mold()

        # route = self.prep_route()
        self.make_daily_time_interval()

        self.set_res_capacity(data=self._cstr_mst[self._key.res_avail_time])

        data = self.add_item_info(data=data, item=item)

        return data

    def add_item_info(self, data: pd.DataFrame, item: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(data, item, on=self._item.sku, how='left')

        return merged

    def prep_item(self):
        # Filter columns
        item = self._item_mst[self._col_item_mst].copy()

        item[self._item.sku] = item[self._item.sku].astype(str)
        item[self._item.weight] = item[self._item.weight].astype(float)
        item[self._item.weight_uom] = item[self._item.weight_uom].fillna('G')

        item = item.dropna(subset=[self._item.weight])
        item[self._item.weight] = np.round([weight * self.weight_conv_map[uom] for weight, uom in zip(
            item[self._item.weight], item[self._item.weight_uom])], 4)

        item = item.drop(columns=[self._item.weight_uom])
        item = item.drop_duplicates()

        return item

    def prep_mold(self):
        self._mold_apply_res_grp = list(set([self._res_to_res_grp[res] for res in self.mold_res
                                             if res in self._res_to_res_grp]))

    def make_daily_time_interval(self) -> None:
        self.time_interval = [(i, i * self.sec_of_day, (i + 1) * self.sec_of_day) for i in range(self.schedule_days)]

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
                    mold_res.append(mold_res_weight)
        data['apply_flag'] = flag

        non_apply_dmd = data[~data['apply_flag']].copy()
        apply_dmd = data[data['apply_flag']].copy()
        apply_dmd[self._cstr.mold_res] = mold_res

        apply_dmd = apply_dmd.drop(columns='apply_flag')
        non_apply_dmd = non_apply_dmd.drop(columns='apply_flag')

        return apply_dmd, non_apply_dmd

    def update_timeline_by_day_and_time_index(self, data: pd.DataFrame) -> pd.DataFrame:
        # Slice timeline of half-item
        splited_list = []
        for i, row in data.iterrows():
            add_day = self.add_timeline_day(row, splitted=[])
            splited_list.extend(add_day)

        data_splite = pd.DataFrame(splited_list)
        data_splite = data_splite.reset_index(drop=True)

        return data_splite

    def add_timeline_day(self, row: pd.Series, splitted: list) -> List[pd.Series]:
        dmd_start = row[self._dmd.start_time]
        dmd_end = row[self._dmd.end_time]
        # for day, (day_start, day_end) in enumerate(self._res_to_capa[row[self._res.res]]):
        for day, day_start, day_end in self.time_interval:
            if dmd_start >= day_end:
                continue
            else:
                if dmd_end <= day_end:
                    row['day'] = day
                    rows = self.split_dmd_by_time_index(
                        data=row,
                        time=(dmd_start, dmd_end),
                        day_time=(day_start, day_end)
                    )
                    for row in rows:
                        splitted.append(row)

                    return splitted

    def split_dmd_by_time_index(self, data: pd.Series, time: tuple, day_time: tuple) -> List[pd.Series]:
        split_row = []
        dmd_start, dmd_end = time
        day_start, day_end = day_time

        if dmd_end <= day_start + self.half_sec_of_day:
            data[self._post.time_idx] = 'D'
            split_row.append(data)
        else:
            if dmd_start >= day_start + self.half_sec_of_day:
                data[self._post.time_idx] = 'N'
                split_row.append(data)
            else:
                split_bf = data.copy()
                split_bf[self._dmd.start_time] = dmd_start
                split_bf[self._dmd.end_time] = day_start + self.half_sec_of_day
                split_bf[self._dmd.duration] = split_bf[self._dmd.end_time] - split_bf[self._dmd.start_time]
                split_bf[self._post.time_idx] = 'D'

                split_rate = split_bf[self._dmd.duration] / (data[self._dmd.end_time] - data[self._dmd.start_time])
                split_bf[self._dmd.prod_qty] = int(data[self._dmd.prod_qty] * split_rate)
                split_bf['tot_weight'] = int(data['tot_weight'] * split_rate)
                split_row.append(split_bf)

                split_af = data.copy()
                split_af[self._dmd.start_time] = day_start + self.half_sec_of_day
                split_af[self._dmd.end_time] = dmd_end
                split_af[self._dmd.duration] = split_af[self._dmd.end_time] - split_af[self._dmd.start_time]
                split_af[self._post.time_idx] = 'N'

                split_af[self._dmd.prod_qty] = int(data[self._dmd.prod_qty] * (1 - split_rate))
                split_af['tot_weight'] = int(data['tot_weight'] * (1 - split_rate))
                split_row.append(split_af)

        return split_row

    def set_res_capacity(self, data: pd.DataFrame) -> None:
        # Choose current plant
        data = data[data[self._res.plant] == self._plant].copy()

        capa_col_list = []
        for i in range(self.work_day):
            for kind in ['d', 'n']:
                capa = self._res.res_capa + str(i + 1) + '_' + kind
                data[capa] = np.where(data[capa] > 720, 720, data[capa])
                capa_col_list.append(capa)

        # Filter not available capacity
        data = data[data[capa_col_list].sum(axis=1) != 0].copy()
        data = data[~data[self._res.res_grp].isna()].copy()

        res_to_capa, res_day_capa = {}, {}
        for res, capa_df in data.groupby(by=self._res.res):
            days_capa = capa_df[capa_col_list].values.tolist()[0]
            days_capa = util.make_time_pair(data=days_capa)

            days_capa_list = []
            days_capa_dict = {}
            for day, (day_time, night_time) in enumerate(days_capa * self.schedule_days):
                start_time, end_time = util.calc_daily_avail_time(
                    day=day,
                    day_time=int(day_time * self.time_multiple),
                    night_time=int(night_time * self.time_multiple),
                )
                if start_time != end_time:
                    days_capa_list.append([start_time, end_time])
                    days_capa_dict[day] = [start_time, end_time]

            days_capa_list = self.connect_continuous_capa(data=days_capa_list)
            if len(days_capa_list) > 0:
                res_to_capa[res] = days_capa_list
                res_day_capa[res] = days_capa_dict

        self._res_to_capa = res_to_capa
        self._res_day_capa = res_day_capa

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
                    curr_capa[3] += next_capa[3]
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
                        if len(res_df) > 1:
                            # Sort demand on start time
                            res_df = res_df.sort_values(by=self._dmd.start_time)
                            timeline_list = res_df[
                                [self._dmd.start_time, self._dmd.end_time, self._dmd.prod_qty, 'tot_weight']
                            ].values.tolist()

                            # Connect the capacity if timeline is continuous
                            timeline_connected = self.connect_continuous_capa_weight(data=timeline_list)

                            # Remake demand dataframe using connected timeline
                            for stime, etime, qty, weight in timeline_connected:
                                common = res_df.iloc[0].copy()
                                dmd_series = self.update_connected_timeline(
                                    common=common,
                                    dmd=dmd,
                                    item=item,
                                    res_grp=res_grp,
                                    res=res,
                                    start_time=stime,
                                    end_time=etime,
                                    qty=qty,
                                    weight=weight,
                                )
                                revised_data = revised_data.append(dmd_series)
                        else:
                            revised_data = revised_data.append(res_df)

        revised_data[self._dmd.duration] = revised_data[self._dmd.end_time] - revised_data[self._dmd.start_time]
        revised_data = revised_data.sort_values(by=self._dmd.dmd)
        revised_data = revised_data.reset_index(drop=True)

        return revised_data

    def update_connected_timeline(self, common: pd.Series, dmd, item, res_grp, res, start_time, end_time,
                                  qty, weight):
        common[self._dmd.dmd] = dmd
        common[self._item.sku] = item
        common[self._res.res] = res
        common[self._res.res_grp] = res_grp
        common[self._dmd.start_time] = start_time
        common[self._dmd.end_time] = end_time
        common[self._dmd.prod_qty] = qty
        common['tot_weight'] = weight

        return common
