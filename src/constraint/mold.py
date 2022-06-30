import common.util as util
import common.config as config
from common.name import Key, Demand, Item, Resource, Constraint

import numpy as np
import pandas as pd
from typing import Tuple, List


class Mold(object):
    def __init__(self, plant, data, res_dur, mold_cstr, res_to_res_grp):
        # Name instance attribute
        self._key = Key()
        self._item = Item()
        self._dmd = Demand()
        self._res = Resource()
        self._cstr = Constraint()

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
        self.work_day = config.work_day    # 6 days: Monday ~ Friday
        self.sec_of_day = config.sec_of_day    # Seconds of 1 day
        self.day_multiple = 3
        self.time_multiple = config.time_multiple    # Minute -> Seconds
        self.time_interval = []
        self.schedule_weeks = config.schedule_weeks

        # Column information
        self._col_item_mst = [self._item.sku, self._item.weight, self._item.weight_uom]

    def apply(self, data: pd.DataFrame):
        # Preprocess the dataset
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

            mold_res_day = apply_dmd.groupby(by=self._cstr.mold_res).max()['day'].reset_index()
            mold_res_day['day'] = mold_res_day['day'].values * self.day_multiple

            result = pd.DataFrame()
            for mold_res in mold_res_day[self._cstr.mold_res]:
                mold_df = apply_dmd[apply_dmd[self._cstr.mold_res] == mold_res].copy()
                for day in range(0, mold_res_day[mold_res_day[self._cstr.mold_res] == mold_res]['day'].values[0]):
                    mold_df = self.apply_time_move(data=mold_df, mold_res=mold_res, day=day)
                result = pd.concat([result, mold_df])

            result = self.update_day(data=result)
            result = pd.concat([result, non_apply_dmd], axis=0).reset_index(drop=True)
            result = result.drop(columns=[self._dmd.prod_qty, self._dmd.duration, self._cstr.mold_res,
                                          self._item.weight, 'capa_use_rate',  'tot_weight', 'day'])

            return result

    def apply_time_move(self, data, mold_res, day):
        while self.check_daily_capa_excess(data=data, mold_res=mold_res, day=day):
            # print(f'resource: {mold_res}, day: {day}')
            move_dmd, excess_capa = self.decide_which_dmd_move(
                data=data, mold_res=mold_res, day=day
            )
            data = self.split_and_merge_dmd(
                data=data,
                dmd=move_dmd,
                excess_capa=excess_capa,
                day=day
            )

        return data

    def split_and_merge_dmd(self, data, dmd, excess_capa, day):
        drop_idx = data[(data[self._dmd.dmd] == dmd[self._dmd.dmd])
                        & (data[self._dmd.start_time] == dmd[self._dmd.start_time])
                        & (data[self._dmd.end_time] == dmd[self._dmd.end_time])].index
        data = data.drop(index=drop_idx)
        dmd_bf, dmd_af, move_duration = self.split_dmd(data=dmd, excess_capa=excess_capa, day=day)
        revised_dmd = self.update_dmd_timeline(data=data, dmd=dmd_af, move_duration=move_duration)

        revised_dmd = revised_dmd.append(dmd_bf)
        # revised_dmd = revised_dmd.append(dmd_af)
        revised_dmd = revised_dmd.reset_index(drop=True)

        return revised_dmd

    def update_dmd_timeline(self, data, dmd: pd.Series, move_duration) -> pd.DataFrame:
        move_dmd, non_move_dmd = self.classify_dmd_move_or_not(data=data, dmd=dmd)

        time_moved_dmd = self.move_timeline(data=move_dmd, duration=move_duration)

        # Dev
        time_moved_dmd = time_moved_dmd.append(dmd).reset_index(drop=True)

        if len(time_moved_dmd) > 0:
            time_moved_dmd = self. apply_res_capa_on_timeline(data=time_moved_dmd, res=dmd[self._res.res])

            time_moved_dmd = self.update_day(data=time_moved_dmd)

            # Connect continuous timeline of each demand
            time_moved_dmd = self.connect_continuous_dmd(data=time_moved_dmd)

        revised_dmd = pd.concat([time_moved_dmd, non_move_dmd], axis=0)
        revised_dmd = revised_dmd.reset_index(drop=True)

        return revised_dmd

    def move_timeline(self, data, duration) -> pd.DataFrame:
        data[self._dmd.start_time] = data[self._dmd.start_time] + duration
        data[self._dmd.end_time] = data[self._dmd.end_time] + duration

        return data

    def classify_dmd_move_or_not(self, data, dmd: pd.Series):
        dmd_of_move_res = data[data[self._res.res] == dmd[self._res.res]].copy()
        move_dmd = dmd_of_move_res[dmd_of_move_res[self._dmd.start_time] >= dmd[self._dmd.start_time]]\
            .copy()\
            .reset_index(drop=True)
        non_move_dmd = dmd_of_move_res[dmd_of_move_res[self._dmd.start_time] < dmd[self._dmd.start_time]]\
            .copy()\
            .reset_index(drop=True)
        dmd_of_non_move_res = data[data[self._res.res] != dmd[self._res.res]].copy()

        non_move_dmd = pd.concat([dmd_of_non_move_res, non_move_dmd]).reset_index(drop=True)

        return move_dmd, non_move_dmd

    def split_dmd(self, data: pd.Series, excess_capa, day) -> Tuple[pd.Series, pd.Series, int]:
        bf_dmd, af_dmd = data.copy(), data.copy()

        # Demand (before)
        if data['tot_weight'] >= excess_capa:
            bf_capa = data['tot_weight'] - excess_capa
            bf_dmd['tot_weight'] = bf_capa
            bf_rate, af_rate = self.cala_split_rate(bf_capa=bf_capa, af_capa=excess_capa)
            bf_dmd[self._dmd.prod_qty] = round(data[self._dmd.prod_qty] * bf_rate)
            bf_dmd[self._dmd.duration] = round(data[self._dmd.duration] * bf_rate)
            bf_dmd[self._dmd.start_time] = bf_dmd[self._dmd.end_time] - bf_dmd[self._dmd.duration]
            bf_dmd['day'] = self.update_day(data=bf_dmd)

            # Demand (after)
            af_capa = excess_capa
            af_dmd['tot_weight'] = af_capa
            af_dmd[self._dmd.prod_qty] = round(data[self._dmd.prod_qty] * af_rate)
            af_dmd[self._dmd.duration] = round(data[self._dmd.duration] * af_rate)
            # af_dmd[self._dmd.start_time] = self.res_day_capa[af_dmd[self._res.res]][self.calc_next_day_bak(day)][0]
            af_dmd[self._dmd.start_time] = self.calc_next_day(res=af_dmd[self._res.res], day=bf_dmd['day'])
            af_dmd[self._dmd.end_time] = af_dmd[self._dmd.start_time] + af_dmd[self._dmd.duration]
            af_dmd['day'] = self.update_day(data=af_dmd)
            move_duration = af_dmd[self._dmd.end_time] - bf_dmd[self._dmd.end_time]
        else:
            bf_dmd = pd.DataFrame()
            # af_dmd[self._dmd.start_time] = self.res_day_capa[data[self._res.res]][self.calc_next_day_bak(day)][0]
            af_dmd[self._dmd.start_time] = self.calc_next_day(res=af_dmd[self._res.res], day=day)
            af_dmd[self._dmd.end_time] = af_dmd[self._dmd.start_time] + data[self._dmd.duration]
            af_dmd['day'] = self.update_day(data=af_dmd)
            move_duration = af_dmd[self._dmd.start_time] - data[self._dmd.start_time]

        return bf_dmd, af_dmd, move_duration

    @staticmethod
    def cala_split_rate(bf_capa: int, af_capa: int) -> Tuple[float, float]:
        bf_rate = bf_capa / (bf_capa + af_capa)

        return bf_rate, 1 - bf_rate

    def calc_next_day(self, res: str, day: int) -> int:
        res_day_capa_list = self._res_day_capa[res]
        if len(res_day_capa_list) == 7:
            next_day = day + 1
        elif len(res_day_capa_list) == 6:
            if day % 7 == 5:
                next_day = day + 2
            else:
                next_day = day + 1
        else:
            if day % 7 == 5:
                next_day = day + 2
            elif day % 7 == 4:
                next_day = day + 3
            else:
                next_day = day + 1

        return res_day_capa_list[next_day][0]

    def decide_which_dmd_move(self, data: pd.DataFrame, mold_res: str, day: int) -> Tuple[pd.Series, float]:
        res_data = data[(data[self._cstr.mold_res] == mold_res) & (data['day'] == day)].copy()

        excess_capa = int(res_data.sum()['tot_weight'] - sum(self.mold_capa[mold_res][day % 7]))
        weight_by_res = pd.merge(
            res_data,
            res_data[[self._res.res, self._dmd.end_time]].groupby(self._res.res).max().reset_index(),
            on=[self._res.res, self._dmd.end_time]
        )

        if len(weight_by_res[weight_by_res['tot_weight'] >= excess_capa]):
            weight_over_res = weight_by_res[weight_by_res['tot_weight'] >= excess_capa]
            weight_over_min_res = weight_over_res[weight_over_res['tot_weight'] == weight_over_res['tot_weight'].min()
                                                  ].iloc[0]
            move_dmd = res_data[(res_data[self._res.res] == weight_over_min_res[self._res.res])
                                & (res_data['tot_weight'] == weight_over_min_res['tot_weight'])].iloc[0]
        else:
            if len(res_data[res_data['tot_weight'] >= excess_capa]) > 0:
                weight_over_res = res_data[res_data['tot_weight'] >= excess_capa]
                move_dmd = weight_over_res[weight_over_res['tot_weight'] == weight_over_res['tot_weight'].min()].iloc[0]
            else:
                move_list = res_data[res_data[self._dmd.start_time] == res_data[self._dmd.start_time].min()]
                move_dmd = move_list[move_list['tot_weight'] == move_list['tot_weight'].max()].iloc[0]

        return move_dmd, excess_capa

    def check_daily_capa_excess(self, data: pd.DataFrame, mold_res: str, day: int):
        # res_data = data[(data[self._cstr.mold_res] == mold_res) & (data['day'] == day)].copy()
        data = self.update_day(data=data)
        day_data = data[data['day'] == day].copy()
        flag = False
        if len(day_data) > 0:
            if int(day_data.sum()['tot_weight']) > sum(self.mold_capa[mold_res][day % 7]):
                flag = True

        return flag

    def update_day(self, data) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            data['day'] = [self.check_day(time=stime) for stime in data[self._dmd.start_time]]
        elif isinstance(data, pd.Series):
            data = self.check_day(time=data[self._dmd.end_time])

        return data

    def check_day(self, time):
        for day, stime, etime in self.time_interval:
            if stime <= time < etime:
                return day

    def apply_res_capa_on_timeline(self, data: pd.DataFrame, res: str) -> pd.DataFrame:
        applied_data = pd.DataFrame()

        time_start = 0
        res_capa_list = self._res_to_capa[res]
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
                        dmd[self._dmd.duration] = end_time - start_time
                        if dmd[self._dmd.duration] != 0:
                            split_rate = running_time / dmd[self._dmd.duration]
                        else:
                            split_rate = 0
                        dmd[self._dmd.prod_qty] = round(dmd[self._dmd.prod_qty] * split_rate)
                        dmd['tot_weight'] = round(dmd['tot_weight'] * split_rate)

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

                        split_rate = dmd[self._dmd.duration] / running_time
                        dmd[self._dmd.prod_qty] = round(dmd[self._dmd.prod_qty] * split_rate)
                        dmd['tot_weight'] = round(dmd['tot_weight'] * split_rate)

                        applied_data = applied_data.append(dmd)
                        start_time = capa_end

        applied_data = applied_data.reset_index(drop=True)

        return applied_data

    def add_weight(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self._dmd.duration] = data[self._dmd.end_time] - data[self._dmd.start_time]
        data['capa_use_rate'] = [    # Capa use rate
            self._res_dur[sku][res] for sku, res in zip(data[self._item.sku], data[self._res.res])
        ]

        # Calculate Production quantity
        data[self._dmd.prod_qty] = data[self._dmd.duration] / data['capa_use_rate']
        data[self._dmd.prod_qty] = np.where(data[self._dmd.dmd].str.contains('@'), 0, data[self._dmd.prod_qty])
        data['tot_weight'] = data[self._item.weight] * np.floor(data[self._dmd.prod_qty])
        # data['tot_weight'] = data['mold_weight'] * np.floor(data[self._dmd.prod_qty])
        data['tot_weight'] = data['tot_weight'].astype(int)

        # Change job change
        data[self._dmd.prod_qty] = np.where(data[self._dmd.dmd].str.contains('@'), 0, data[self._dmd.prod_qty])
        data = data.reset_index(drop=True)

        return data

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
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
        self.time_interval = [(i, i * self.sec_of_day, (i + 1) * self.sec_of_day) for i in range(self.schedule_weeks)
                              if i % 7 != 6]

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

    def slice_timeline_by_each_day(self, data: pd.DataFrame) -> pd.DataFrame:
        # Slice timeline of half-item
        splited_list = []
        for i, row in data.iterrows():
            add_day = self.add_timeline_day(row, splitted=[])
            splited_list.extend(add_day)

        data_splite = pd.DataFrame(splited_list)

        return data_splite

    def add_timeline_day(self, row: pd.Series, splitted: list) -> List[pd.Series]:
        stime = row[self._dmd.start_time]
        etime = row[self._dmd.end_time]
        for day, (day_start, day_end) in enumerate(self._res_to_capa[row[self._res.res]]):
            if stime >= day_end:
                continue
            else:
                if etime <= day_end:
                    row['day'] = day
                    splitted.append(row)

                    return splitted

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
            for day, (day_time, night_time) in enumerate(days_capa * self.schedule_weeks):
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
                        for day, day_df in res_df.groupby(by='day'):
                            if len(day_df) > 1:
                                # Sort demand on start time
                                day_df = day_df.sort_values(by=self._dmd.start_time)
                                timeline_list = day_df[
                                    [self._dmd.start_time, self._dmd.end_time, self._dmd.prod_qty, 'tot_weight']
                                ].values.tolist()

                                # Connect the capacity if timeline is continuous
                                timeline_connected = self.connect_continuous_capa_weight(data=timeline_list)

                                # Remake demand dataframe using connected timeline
                                for stime, etime, qty, weight in timeline_connected:
                                    common = day_df.iloc[0].copy()
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
                                revised_data = revised_data.append(day_df)

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
