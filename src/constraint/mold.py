import common.util as util
import common.config as config
from common.name import Key, Demand, Item, Resource, Constraint, Post

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Tuple, List, Union


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
        self._res_sku_dur = {}
        self._res_to_capa = {}
        self._res_day_capa = {}
        self._res_to_res_grp = res_to_res_grp
        self._res_grp_to_res = {}

        # Mold constraint instance attribute
        self._mold_res = mold_cstr[self._key.mold_res].get(plant, None)
        self._mold_capa = mold_cstr[self._key.mold_capa].get(plant, None)
        self._new_sku_dmd = []
        self._sku_weight_map = {}
        self._weight_conv_map = {'G': 0.001, 'KG': 1, 'TON': 1000}
        self._sku_brand_pkg_map = {}

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
        self._col_item_weight = [self._item.sku, self._item.weight, self._item.weight_uom]
        self._col_item_info = [self._item.sku, self._item.brand, self._item.pkg]

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

            # Get last production day of each mold resource
            mold_res_day = apply_dmd.groupby(by=self._cstr.mold_res).max()['day'].reset_index()

            # Increase days since timeline would be moved backward
            mold_res_day['day'] = mold_res_day['day'].values * self.day_multiple
            result = pd.DataFrame()
            for mold_res in mold_res_day[self._cstr.mold_res]:
                mold_result = pd.DataFrame()
                self.prev_fix_dmd = None
                mold_df = apply_dmd[apply_dmd[self._cstr.mold_res] == mold_res].copy()
                mold_usable_days = self.calc_mold_usable_day(
                    mold_res=mold_res,
                    last_day=mold_res_day[mold_res_day[self._cstr.mold_res] == mold_res].squeeze()['day']
                )
                for day in mold_usable_days:
                    for time_idx in ['D', 'N']:
                        if (len(mold_df) > 0) & (len(mold_df[(mold_df['day'] == day)
                                                             & (mold_df[self._post.time_idx] == time_idx)]) > 0):
                            # Apply the constraint of mold resource
                            mold_df, fix_dmd = self.apply_mold_res_constraint(
                                data=mold_df,
                                mold_res=mold_res,
                                day=day,
                                time_idx=time_idx
                            )
                            mold_result = pd.concat([mold_result, fix_dmd], axis=0)
                            mold_result = mold_result.reset_index(drop=True)
                result = pd.concat([result, mold_result], axis=0)

            result = pd.concat([result, non_apply_dmd], axis=0).reset_index(drop=True)
            result = result.drop(
                columns=[self._dmd.prod_qty, self._dmd.duration, self._cstr.mold_res, self._item.weight,
                         self._post.time_idx, 'capa_use_rate', 'tot_weight', 'day', 'fixed']
            )

            return result

    def calc_mold_usable_day(self, mold_res, last_day):
        mold_avail_day = [i for i, capa in enumerate(self._mold_capa[mold_res]) if sum(capa) != 0]
        usable_day = [i for i in range(last_day) if i % 7 in mold_avail_day]

        return usable_day

    def apply_mold_res_constraint(self, data: pd.DataFrame, mold_res: str, day: int, time_idx: str) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Check if daily time index(D/N) capacity is over than mold capacity
        if self.check_daily_time_idx_capa_is_over(data=data, mold_res=mold_res, day=day, time_idx=time_idx):
            # Decide which demand is fixed
            data, fix_dmd = self.decide_fix_dmd_and_update_timeline(
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
        if len(data) > 0:
            data = self.update_timeline_by_day_and_time_index(data=data)

        return data, fix_dmd

    def choose_res_and_sku_to_make(self, data, day, time_idx, mold_res, mold_capa):
        # Get resource group of mold resource
        res_grp = data[self._res.res_grp].values[0]

        # Filter demand of day and time index (day/night)
        day_time_dmd = data[(data['day'] == day) & (data[self._post.time_idx] == time_idx)]

        # Get available resource on each demand
        dmd_avail_res = self.check_avail_dmd_list_of_each_sku(res_grp=res_grp, day_time_dmd=day_time_dmd)

        # Calculate total weight of demand (day / time index)
        day_time_idx_dmd_weight = day_time_dmd['tot_weight'].sum()

        # Calculate additional weight
        additional_weight = mold_capa - day_time_idx_dmd_weight

        if len(dmd_avail_res) > 0:
            over_weight_dmd, less_weight_dmd = self.check_additional_weight_on_avail_res(
                dmd_avail_res=dmd_avail_res,
                day_time_dmd=day_time_dmd,
                mold_capa=mold_capa,
                additional_weight=additional_weight
            )

            if len(over_weight_dmd) > 0:
                temp = []
                for dmd_id, candidates in over_weight_dmd.items():
                    for dmd, res, weight in candidates:
                        temp.append([dmd, res, weight])
                choose_dmd_list = [sorted(temp, key=lambda x: x[2], reverse=True)[0]]
                choose_dmd_list[0][2] = additional_weight
            else:
                choose_dmd_list = self.check_less_weight_possible_to_make(
                    data=less_weight_dmd,
                    additional_weight=additional_weight
                )

            choose_dmd_list = self.update_additional_dmd(dmd_list=choose_dmd_list)
            choose_dmd = pd.DataFrame(choose_dmd_list)
        else:
            choose_dmd = self.find_other_sku_to_make(
                res_grp=res_grp,
                day_time_dmd=day_time_dmd,
                day=day,
                time_idx=time_idx,
                mold_res=mold_res,
                additional_weight=additional_weight
            )

        return choose_dmd

    def find_other_sku_to_make(self, res_grp, day_time_dmd, day, time_idx, mold_res, additional_weight):
        # Get all of resource set in resource group
        all_res = set(self._res_grp_to_res[res_grp])

        # get resource list that used now
        day_time_dmd_res = set(day_time_dmd[self._res.res].copy())

        prod_avail_res = all_res - day_time_dmd_res

        avail_sku = []
        for res in prod_avail_res:
            sku = self._res_sku_dur.get(res, None)
            if sku is not None:
                avail_sku.append([res, sku])

        if len(avail_sku) == 0:
            print(f"Does not find any sku that can be made on available resource "
                  f"(Mold Resource: {mold_res} Day: {day} Time index: {time_idx}).")

        res_sku_max_weight = []
        for res, sku_list in avail_sku:
            for sku in sku_list:
                max_duration = self.calc_max_duration_on_res(res=res, day=day, time_idx=time_idx)
                if max_duration > 0:
                    max_prod_qty = round(max_duration / self._res_dur[sku][res], 3)
                    max_weight = int(max_prod_qty * self._sku_weight_map[sku])
                    res_sku_max_weight.append([res, sku, max_weight])

        res_sku_max_weight = sorted(res_sku_max_weight, key=lambda x: (-x[2], x[0], x[1]))

        choose_dmd = day_time_dmd.copy()
        avail_res = list(prod_avail_res)
        for res, sku, max_weight in res_sku_max_weight:
            if additional_weight > 0:
                if res in avail_res:
                    if max_weight > additional_weight:
                        new_dmd = self.make_new_sku_dmd(
                            res_grp=res_grp, res=res, sku=sku, day=day, time_idx=time_idx, mold_res=mold_res,
                            weight=additional_weight)
                        choose_dmd = choose_dmd.append(new_dmd, ignore_index=True)
                        break
                    else:
                        new_dmd = self.make_new_sku_dmd(
                            res_grp=res_grp, res=res, sku=sku, day=day, time_idx=time_idx, mold_res=mold_res,
                            weight=max_weight)
                        choose_dmd = choose_dmd.append(new_dmd, ignore_index=True)
                        avail_res.remove(res)
                        additional_weight -= max_weight

        return choose_dmd

    def make_new_sku_dmd(self, res_grp, res, sku, day, time_idx, mold_res, weight):
        new_dmd = pd.Series()
        new_dmd[self._res.res_grp] = res_grp
        new_dmd[self._res.res] = res
        new_dmd[self._item.sku] = sku
        new_dmd[self._item.pkg] = self._sku_brand_pkg_map[sku][self._item.pkg]
        new_dmd[self._item.brand] = self._sku_brand_pkg_map[sku][self._item.brand]
        new_dmd['day'] = day
        new_dmd['kind'] = 'demand'
        new_dmd['fixed'] = True
        new_dmd[self._cstr.mold_res] = mold_res
        new_dmd[self._post.time_idx] = time_idx
        new_dmd['weight'] = self._sku_weight_map[sku]
        new_dmd['tot_weight'] = weight
        new_dmd['capa_use_rate'] = self._res_dur[sku][res]
        new_dmd[self._dmd.prod_qty] = round(new_dmd['tot_weight'] / new_dmd['weight'], 3)
        new_dmd[self._dmd.duration] = int(new_dmd[self._dmd.prod_qty] * new_dmd['capa_use_rate'])
        new_dmd[self._dmd.start_time] = self._res_day_capa[res][day][0]
        new_dmd[self._dmd.end_time] = new_dmd[self._dmd.start_time] + new_dmd[self._dmd.duration]

        if len(self._new_sku_dmd) == 0:
            dmd_id = 'EP_0000001'
        else:
            dmd_id = 'EP_' + str(int(self._new_sku_dmd[-1][3:]) + 1).zfill(7)
        self._new_sku_dmd.append(dmd_id)
        new_dmd[self._dmd.dmd] = dmd_id

        return new_dmd

    def check_less_weight_possible_to_make(self, data, additional_weight) -> list:
        temp = {}
        for dmd_id, candidates in data.items():
            for dmd, res, weight in candidates:
                if res not in temp:
                    temp[res] = [[dmd, res, weight]]
                else:
                    temp[res].append([dmd, res, weight])

        res_max_weight_dmd_list = []
        for res, val in temp.items():
            res_max_weight_dmd = sorted(val, key=lambda x: x[2], reverse=True)[0]
            res_max_weight_dmd_list.append(res_max_weight_dmd)
        res_max_weight_dmd_list = sorted(res_max_weight_dmd_list, key=lambda x: x[2], reverse=True)

        choose_dmd_list = []
        for dmd, res, weight in res_max_weight_dmd_list:
            if weight < additional_weight:
                choose_dmd_list.append([dmd, res, weight])
                additional_weight -= weight
            else:
                choose_dmd_list.append([dmd, res, additional_weight])
                break

        # if additional_weight > 0:
        #     choose_dmd_list = []

        return choose_dmd_list

        # return choose_dmd, day_time_idx_dmd_weight
    def check_additional_weight_on_avail_res(self, dmd_avail_res, day_time_dmd, mold_capa, additional_weight):
        over_weight_dmd = {}
        less_weight_dmd = {}
        for dmd, avail_res in dmd_avail_res:
            for res in avail_res:
                max_weight = self.calc_dmd_res_max_weight(dmd=dmd, res=res)
                dmd_res_weight = (dmd, res, max_weight)
                if max_weight >= additional_weight:
                    if dmd[self._dmd.dmd] not in over_weight_dmd:
                        over_weight_dmd[dmd[self._dmd.dmd]] = [dmd_res_weight]
                    else:
                        over_weight_dmd[dmd[self._dmd.dmd]].append(dmd_res_weight)
                else:
                    if dmd[self._dmd.dmd] not in less_weight_dmd:
                        less_weight_dmd[dmd[self._dmd.dmd]] = [dmd_res_weight]
                    else:
                        less_weight_dmd[dmd[self._dmd.dmd]].append(dmd_res_weight)

        return over_weight_dmd, less_weight_dmd

    def calc_dmd_res_max_weight(self, dmd, res):
        capa_use_rate = self._res_dur[dmd[self._item.sku]][res]
        max_duration = self.calc_max_duration_on_res(res=res, day=dmd['day'], time_idx=dmd[self._post.time_idx])
        max_qty = round(max_duration / capa_use_rate, 3)
        max_weight = int(max_qty * dmd[self._item.weight])

        return max_weight

    def calc_max_duration_on_res(self, res, day, time_idx) -> int:
        start_time, end_time = self.time_interval[day][1:]
        start_capa, end_capa = self._res_day_capa[res][day]
        middle_time = start_time + self.half_sec_of_day

        max_duration = 0
        if time_idx == 'D':
            max_duration = int(middle_time - start_capa)
        elif time_idx == 'N':
            max_duration = int(end_capa - middle_time)

        return max_duration

    def check_avail_dmd_list_of_each_sku(self, res_grp, day_time_dmd):
        # Get all of resource set in resource group
        all_res = set(self._res_grp_to_res[res_grp])

        dmd_avail_res = []
        for i, dmd in day_time_dmd.iterrows():
            # get resource list that used now
            day_time_dmd_res = set(day_time_dmd[self._res.res].copy())

            # get available resource that currently not used in resource group now
            prod_avail_res = all_res - day_time_dmd_res

            # get available resource list of demand SKU based on item-resource duration
            dmd_sku_avail_res = set(self._res_dur[dmd[self._item.sku]].copy())

            # resource candidate that can produce additional SKU
            avail_res = list(dmd_sku_avail_res.intersection(prod_avail_res))

            if len(avail_res) > 0:
                dmd_avail_res.append([dmd, avail_res])

        return dmd_avail_res

    def add_extra_capa_on_dmd(self, data: pd.DataFrame, mold_res, day, time_idx) -> Tuple[pd.DataFrame, pd.DataFrame]:
        mold_capa = self._mold_capa[mold_res][day % 7][self.time_idx_map[time_idx]]

        # Case when previous fixed demand does not exist
        if self.prev_fix_dmd is None:
            choose_dmd = self.choose_res_and_sku_to_make(
                data=data, day=day, time_idx=time_idx, mold_res=mold_res, mold_capa=mold_capa
            )
        else:
            # Choose demand from previous fixed demand
            day_time_dmd = data[(data['day'] == day) & (data[self._post.time_idx] == time_idx)].copy()
            data = data.drop(index=day_time_dmd.index)
            choose_dmd = self.choose_dmd_from_prev_fix_dmd(
                data=day_time_dmd,
                mold_res=mold_res,
                mold_capa=mold_capa,
                day=day,
                time_idx=time_idx
            )
            if len(choose_dmd) == 0:
                print(f"Mold Resource {mold_res} cannot use all of capa in Day: {day} / Time-Index: {time_idx}.")

            # for choose_dmd in choose_dmd_list:
            #     data = data.drop(choose_dmd.name)
            #     data = data.reset_index(drop=True)

        return data, choose_dmd

    def choose_dmd_from_prev_fix_dmd(self, data, mold_res, mold_capa, day, time_idx) -> Union[pd.Series, pd.DataFrame]:
        # Case when previous fixed demand exist
        prev_fix_dmd = self.prev_fix_dmd.copy()

        choose_dmd = self.find_producible_res(
            data=data,
            prev_fix_dmd=prev_fix_dmd,
            res_grp=prev_fix_dmd[self._res.res_grp].iloc[0],
            day=day,
            time_idx=time_idx,
            mold_capa=mold_capa
        )
        data['fixed'] = True
        choose_dmd = pd.concat([choose_dmd, data], axis=0)

        #
        if choose_dmd['tot_weight'].sum() < mold_capa:
            print(f"Capa cannot be made from current resource status on Mold resource: "
                  f"{mold_res} Day: {day} Time index: {time_idx} {mold_capa - choose_dmd['tot_weight'].sum()}")

        return choose_dmd

    def find_producible_res(self, data, prev_fix_dmd, res_grp, day, time_idx, mold_capa) -> pd.DataFrame:
        # Get currently available resource
        all_res = set(self._res_grp_to_res[res_grp])
        use_res = set(data[self._res.res])
        avail_res = all_res - use_res

        # Get the producible SKU candidate
        sku_set = set(data[self._item.sku]) | set(prev_fix_dmd[self._item.sku])
        sku_res_list = [(sku, list(set(self._res_dur[sku].keys()) & avail_res)) for sku in list(sku_set)
                        if len(list(set(self._res_dur[sku].keys()) & avail_res)) > 0]

        sku_res_weight = []
        for sku, res_list in sku_res_list:
            for res in res_list:
                capa_use_rate = self._res_dur[sku][res]
                max_duration = self.calc_max_duration_on_res(res=res, day=day, time_idx=time_idx)
                if max_duration > 0:
                    max_qty = round(max_duration / capa_use_rate, 3)
                    max_weight = int(max_qty * self._sku_weight_map[sku])
                    sku_res_weight.append([sku, res, max_weight])

        sku_res_weight = sorted(sku_res_weight, key=lambda x: x[2], reverse=True)

        additional_weight = mold_capa - data['tot_weight'].sum()
        choose_res_sku = []
        avail_res_list = list(avail_res)
        for sku, res, weight in sku_res_weight:
            if additional_weight > 0:
                if res in avail_res_list:
                    if weight > additional_weight:
                        choose_res_sku.append([sku, res, additional_weight])
                        avail_res_list.remove(res)
                        break
                    else:
                        additional_weight -= weight
                        choose_res_sku.append([sku, res, weight])
                        avail_res_list.remove(res)
            else:
                break

        choose_dmd = self.get_dmd_of_choose_res_sku(
            data=pd.concat([data, prev_fix_dmd], axis=0),
            choose_res_sku=choose_res_sku,
            day=day,
            time_idx=time_idx
        )

        return choose_dmd

    def get_dmd_of_choose_res_sku(self, data, choose_res_sku: list, day, time_idx) -> pd.DataFrame:
        choose_dmd = []
        for sku, res, weight in choose_res_sku:
            dmd = data[data[self._item.sku] == sku].iloc[0].copy()
            dmd[self._res.res] = res
            dmd['day'] = day
            dmd[self._post.time_idx] = time_idx
            dmd['tot_weight'] = weight
            dmd['capa_use_rate'] = self._res_dur[sku][res]
            dmd[self._dmd.prod_qty] = round(weight / self._sku_weight_map[sku], 3)
            dmd[self._dmd.duration] = int(dmd[self._dmd.prod_qty] * dmd['capa_use_rate'])

            res_start_time = self._res_day_capa[res][day][0]
            dmd[self._dmd.start_time] = res_start_time + self.half_sec_of_day if time_idx == 'N' else res_start_time
            dmd[self._dmd.end_time] = dmd[self._dmd.start_time] + dmd[self._dmd.duration]

            choose_dmd.append(dmd)

        return pd.DataFrame(choose_dmd)

    def update_additional_dmd(self, dmd_list: list):
        result = []
        for dmd, res, weight in dmd_list:
            dmd[self._res.res] = res
            dmd['tot_weight'] = weight
            dmd[self._dmd.prod_qty] = round(weight / dmd[self._item.weight], 3)
            dmd[self._dmd.duration] = int(dmd[self._dmd.prod_qty] * dmd['capa_use_rate'])
            dmd[self._dmd.end_time] = dmd[self._dmd.start_time] + dmd[self._dmd.duration]
            result.append(dmd)

        return result

    def move_timeline(self, data, fix_dmd_df, fix_af_dmd, day, time_idx) -> pd.DataFrame:
        last_fix_res_data = data[data[self._res.res] == fix_af_dmd[self._res.res]].copy()
        other_fix_res = set(fix_dmd_df[self._res.res]) - set([fix_af_dmd[self._res.res]])
        other_fix_res_data = data[data[self._res.res].isin(other_fix_res)].copy()
        other_data = data[~data[self._res.res].isin(fix_dmd_df[self._res.res])].copy()

        # resource demand containing last fixed resource
        fix_res_move_time = fix_af_dmd[self._dmd.end_time] - last_fix_res_data[self._dmd.start_time].min()
        last_fix_res_data[self._dmd.start_time] += fix_res_move_time
        last_fix_res_data[self._dmd.end_time] += fix_res_move_time

        # Outer demand
        base_time = self.time_interval[day][1]
        if time_idx == 'D':
            base_time += self.half_sec_of_day
        else:
            base_time += self.sec_of_day

        # resource demand containing fixed resource (except last fixed resource)
        fix_res_move_time = base_time - other_fix_res_data[self._dmd.start_time].min()
        other_fix_res_data[self._dmd.start_time] += fix_res_move_time
        other_fix_res_data[self._dmd.end_time] += fix_res_move_time

        other_move_data = pd.DataFrame()
        for res, res_df in other_data.groupby(self._res.res):
            other_res_move_time = base_time - res_df[self._dmd.start_time].min()
            res_df[self._dmd.start_time] += other_res_move_time
            res_df[self._dmd.end_time] += other_res_move_time
            other_move_data = pd.concat([other_move_data, res_df], axis=0)

        result = pd.concat([last_fix_res_data, other_fix_res_data, other_move_data], axis=0)
        result = result.reset_index(drop=True)

        return result

    def decide_fix_dmd_and_update_timeline(self, data: pd.DataFrame, mold_res: str, day: int, time_idx: str) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        # get current capacity
        mold_capa = deepcopy(self._mold_capa[mold_res][day % 7][self.time_idx_map[time_idx]])

        fix_dmd_df = pd.DataFrame()
        fix_af_dmd = None
        while mold_capa > 0:
            day_time_data = data[(data['day'] == day) & (data[self._post.time_idx] == time_idx)].copy()
            data, fix_bf_dmd, fix_af_dmd, mold_capa = self.decide_which_dmd_fix(
                data=data,
                day_time_data=day_time_data,
                mold_capa=mold_capa
            )
            fix_dmd_df = fix_dmd_df.append(fix_bf_dmd)

        moved_dmd = self.move_timeline(
            data=data,
            fix_dmd_df=fix_dmd_df,
            fix_af_dmd=fix_af_dmd,
            day=day,
            time_idx=time_idx
        )
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
            fix_tick = day_time_data[day_time_data['fixed'] == 1].iloc[0]
            # fix_tick = day_time_data[day_time_data['fixed'] == 1].squeeze()
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
        fix_af_dmd[self._dmd.prod_qty] = round(fix_tick[self._dmd.prod_qty] * (1 - fix_rate), 3)
        fix_af_dmd[self._dmd.duration] = int(fix_tick[self._dmd.duration] * (1 - fix_rate))

        standard_time = self.time_interval[fix_tick['day']][self.time_idx_map[fix_tick[self._post.time_idx]] + 1]
        if fix_tick[self._post.time_idx] == 'D':
            fix_af_dmd[self._dmd.start_time] = standard_time + self.half_sec_of_day
        else:
            fix_af_dmd[self._dmd.start_time] = standard_time
        fix_af_dmd[self._dmd.end_time] = int(fix_af_dmd[self._dmd.start_time] + fix_af_dmd[self._dmd.duration])
        fix_af_dmd['fixed'] = True

        return data, fix_bf_dmd, fix_af_dmd

    def get_shortest_dur_demand(self, data: pd.DataFrame, earliest_dmd: pd.DataFrame) -> pd.Series:
        dmd_list = earliest_dmd[self._dmd.dmd].values
        res_dmd = data[data[self._dmd.dmd].isin(dmd_list)].copy()

        dmd_dur_series = res_dmd.groupby(self._dmd.dmd).sum()[self._dmd.duration]
        shortest_dmd = dmd_dur_series.index[dmd_dur_series.argmin()]
        fix_tick = earliest_dmd[earliest_dmd[self._dmd.dmd] == shortest_dmd].squeeze()

        return fix_tick

    def check_daily_time_idx_capa_is_over(self, data: pd.DataFrame, mold_res: str, day: int, time_idx: str) -> bool:
        # Update timeline by each day and time index
        data = self.update_timeline_by_day_and_time_index(data=data)
        sliced = data[(data['day'] == day) & (data[self._post.time_idx] == time_idx)].copy()
        flag = False
        if len(sliced) > 0:
            if int(sliced.sum()['tot_weight']) > self._mold_capa[mold_res][day % 7][self.time_idx_map[time_idx]]:
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
        data[self._dmd.prod_qty] = np.round(data[self._dmd.duration] / data['capa_use_rate'], 3)
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

        self.set_res_grp_to_res()
        self.set_res_sku_dur()

        # Preprocess item master
        item = self.prep_item()

        # Preprocess mold data
        self.prep_mold()

        # route = self.prep_route()
        self.make_daily_time_interval()

        self.set_res_capacity(data=self._cstr_mst[self._key.res_avail_time])

        data = self.add_item_info(data=data, item=item)

        return data

    def set_res_sku_dur(self):
        res_dur = deepcopy(self._res_dur)
        res_sku_dur = {}
        for sku, res_dur_map in res_dur.items():
            if sku[0] != '7':
                for res, duration in res_dur_map.items():
                    if res not in res_sku_dur:
                        res_sku_dur[res] = [sku]
                    else:
                        res_sku_dur[res].append(sku)

        self._res_sku_dur = res_sku_dur

    def set_res_grp_to_res(self):
        res_to_res_grp = self._res_to_res_grp.copy()
        res_grp_to_res = {}
        for res, res_grp in res_to_res_grp.items():
            if res_grp not in res_grp_to_res:
                res_grp_to_res[res_grp] = [res]
            else:
                res_grp_to_res[res_grp].append(res)

        self._res_grp_to_res = res_grp_to_res

    def add_item_info(self, data: pd.DataFrame, item: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(data, item, on=self._item.sku, how='left')

        return merged

    def prep_item(self):
        # Filter columns
        item = self._item_mst[self._col_item_weight].copy()

        item[self._item.sku] = item[self._item.sku].astype(str)
        item[self._item.weight] = item[self._item.weight].astype(float)
        item[self._item.weight_uom] = item[self._item.weight_uom].fillna('G')

        item = item.dropna(subset=[self._item.weight])
        item[self._item.weight] = np.round([weight * self._weight_conv_map[uom] for weight, uom in zip(
            item[self._item.weight], item[self._item.weight_uom])], 4)

        item = item.drop(columns=[self._item.weight_uom])
        item = item.drop_duplicates()

        self._sku_weight_map = {sku: weight for sku, weight in zip(item[self._item.sku], item[self._item.weight])}

        # Sku to brand / package
        item_info = self._item_mst[self._col_item_info].copy()
        item_info[self._item.sku] = item_info[self._item.sku].astype(str)
        item_info[self._item.pkg] = item_info[self._item.pkg].astype(str)

        self._sku_brand_pkg_map = {sku: {self._item.brand: brand, self._item.pkg: pkg} for sku, brand, pkg
                                   in zip(item_info[self._item.sku], item_info[self._item.brand],
                                          item_info[self._item.pkg])}

        return item

    def prep_mold(self):
        self._mold_apply_res_grp = list(set([self._res_to_res_grp[res] for res in self._mold_res
                                             if res in self._res_to_res_grp]))

    def make_daily_time_interval(self) -> None:
        self.time_interval = [(i, i * self.sec_of_day, (i + 1) * self.sec_of_day) for i in range(self.schedule_days)]

    def classify_cstr_apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        flag, mold_res, weight = [], [], []
        for res, sku in zip(data[self._res.res], data[self._item.sku]):
            sku_exist = self._mold_res.get(res, None)
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
            add_day = self.add_timeline_day(row)
            if add_day is not None:
                splited_list.extend(add_day)

        data_splite = pd.DataFrame(splited_list)
        data_splite = self.update_qty_weight(data=data_splite)
        data_splite = data_splite.reset_index(drop=True)

        return data_splite

    def add_timeline_day(self, row: pd.Series) -> List[pd.Series]:
        splitted = []
        dmd_start = row[self._dmd.start_time]
        dmd_end = row[self._dmd.end_time]
        for day, day_start, day_end in self.time_interval:
            if dmd_end <= day_start:
                break
            if dmd_start >= day_end:
                continue
            else:
                dmd = row.copy()
                dmd['day'] = day
                if dmd_end > day_end:
                    if dmd_start >= day_start:
                        time = (dmd_start, day_end)
                    else:
                        time = (day_start, day_end)
                    # dmd_list = self.split_dmd_by_time_index(
                    #     data=dmd,
                    #     time=time,
                    #     day_time=(day_start, day_end)
                    # )
                else:
                    if dmd_start >= day_start:
                        time = (dmd_start, dmd_end)
                    else:
                        time = (day_start, dmd_end)

                dmd_list = self.split_dmd_by_time_index(
                    data=dmd,
                    time=time,
                    day_time=(day_start, day_end)
                )

                for dmd_split in dmd_list:
                    splitted.append(dmd_split)

        return splitted

    def split_dmd_by_time_index(self, data: pd.Series, time: tuple, day_time: tuple) -> List[pd.Series]:
        split_row = []
        dmd_start, dmd_end = time
        day_start, day_end = day_time

        if dmd_end <= day_start + self.half_sec_of_day:
            split = data.copy()
            split[self._post.time_idx] = 'D'
            split[self._dmd.start_time] = dmd_start
            split[self._dmd.end_time] = dmd_end
            split[self._dmd.duration] = split[self._dmd.end_time] - split[self._dmd.start_time]

            if data[self._dmd.duration] > 0:
                split_rate = split[self._dmd.duration] / (data[self._dmd.duration])
                split[self._dmd.prod_qty] = round(data[self._dmd.prod_qty] * split_rate, 3)
                split['tot_weight'] = int(data['tot_weight'] * split_rate)

                split_row.append(split)
        else:
            if dmd_start >= day_start + self.half_sec_of_day:
                data[self._post.time_idx] = 'N'
                data[self._dmd.start_time] = dmd_start
                data[self._dmd.end_time] = dmd_end
                data[self._dmd.duration] = data[self._dmd.end_time] - data[self._dmd.start_time]
                split_row.append(data)
            else:
                split_bf = data.copy()
                split_bf[self._dmd.start_time] = dmd_start
                split_bf[self._dmd.end_time] = day_start + self.half_sec_of_day
                split_bf[self._dmd.duration] = split_bf[self._dmd.end_time] - split_bf[self._dmd.start_time]
                split_bf[self._post.time_idx] = 'D'

                split_bf_rate = split_bf[self._dmd.duration] / (data[self._dmd.duration])
                split_bf[self._dmd.prod_qty] = round((data[self._dmd.prod_qty] * split_bf_rate), 3)
                split_bf['tot_weight'] = int(data['tot_weight'] * split_bf_rate)
                split_row.append(split_bf)

                split_af = data.copy()
                split_af[self._dmd.start_time] = day_start + self.half_sec_of_day
                split_af[self._dmd.end_time] = dmd_end
                split_af[self._dmd.duration] = split_af[self._dmd.end_time] - split_af[self._dmd.start_time]
                split_af[self._post.time_idx] = 'N'

                split_af_rate = split_af[self._dmd.duration] / (data[self._dmd.duration])
                split_af[self._dmd.prod_qty] = round(data[self._dmd.prod_qty] * split_af_rate, 3)
                split_af['tot_weight'] = int(data['tot_weight'] * split_af_rate)
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
        revised_data = self.update_qty_weight(data=revised_data)
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

    def update_qty_weight(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self._dmd.prod_qty] = np.round(data[self._dmd.duration] / data['capa_use_rate'], 3)
        data[self._dmd.prod_qty] = np.where(data[self._dmd.dmd].str.contains('@'), 0, data[self._dmd.prod_qty])
        data['tot_weight'] = data[self._dmd.prod_qty] * data[self._item.weight]
        data['tot_weight'] = data['tot_weight'].astype(int)

        return data
