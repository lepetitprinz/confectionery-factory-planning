import common.util as util
import common.config as config
from common.name import Key, Demand, Item, Resource

import numpy as np
import pandas as pd
import datetime as dt
from typing import Tuple


class Necessary(object):
    def __init__(
        self,
        plant: str,
        plant_start_time: dt.datetime,
        org_data: dict,
        demand: pd.DataFrame,
        sim_prod_cstr: dict,
    ):
        # name instance attribute
        self.key = Key()
        self.dmd = Demand()
        self.item = Item()
        self.res = Resource()
        self.col_item = [self.item.sku, self.item.brand, self.item.pkg]

        self.plant = plant
        self.plant_start_time = plant_start_time

        # Dataset
        self.org_data = org_data
        self.item_mst = org_data[self.key.item]
        self.cstr_mst = org_data[self.key.cstr]
        self.dmd_schedule = demand
        self.sim_prod_cstr = sim_prod_cstr

        # data hash map
        self.res_to_capa = {}
        self.res_grp_to_res_map = {}
        self.res_to_res_grp_map = {}
        self.brand_pkg_sku_map = {}

        # Time instance attribute
        self.work_day = config.work_day    # Monday ~ Friday
        self.sec_of_day = 86400    # Seconds of 1 day
        self.time_multiple = config.time_multiple    # Minute -> Seconds
        self.schedule_weeks = config.schedule_weeks
        self.plant_start_hour = config.plant_start_hour

        self.log = []

    def apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        # Data Preprocessing
        data = self.preprocess(data=data)

        # Resource to capacity map
        self.set_res_capacity(data=self.cstr_mst[self.key.res_avail_time])

        # Classify demand if possible to product demand simultaneously
        apply_dmd, non_apply_dmd = self._classify_sim_prod_possible_dmd(data=data)

        # Find and make simultaneously producible product
        all_dmd = self._find_and_make_sim_product(apply_dmd_df=apply_dmd, all_dmd=data)

        # log
        log = sorted(list(set(self.log)))

        return all_dmd, log

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        res = self.prep_res()
        item = self._prep_item()
        data = self._add_item_information(data=data, item=item)

        # Mapping data
        self._set_res_grp_res_map(res=res)
        self._set_brand_pkg_sku_map()

        data['fixed'] = False

        return data

    def prep_res(self) -> pd.DataFrame:
        # Resource master
        res_mst = self.org_data[self.key.res][self.key.res_grp]
        res_mst[self.res.res] = res_mst[self.res.res]
        res_mst[self.res.res_grp] = res_mst[self.res.res_grp]

        res_mst = res_mst[[self.res.plant, self.res.res_grp, self.res.res]].drop_duplicates()

        res_mst = res_mst[res_mst[self.res.plant] == self.plant]

        return res_mst

    def _set_res_grp_res_map(self, res: pd.DataFrame) -> None:
        # Resource duration
        res_duration = self.org_data[self.key.res][self.key.res_duration]

        res_duration[self.res.res] = res_duration[self.res.res].astype(str)
        res_duration[self.item.sku] = res_duration[self.item.sku].astype(str)

        res_duration = pd.merge(res_duration, res, how='inner', on=self.res.res)

        res_to_res_grp_map = {}
        res_grp_to_res_map = {}
        for res_grp, res_grp_df in res_duration.groupby(by=self.res.res_grp):
            for res in res_grp_df[self.res.res]:
                res_to_res_grp_map[res] = res_grp
                if res_grp in res_grp_to_res_map:
                    if res not in res_grp_to_res_map[res_grp]:
                        res_grp_to_res_map[res_grp].append(res)
                else:
                    res_grp_to_res_map[res_grp] = [res]

        self.res_grp_to_res_map = res_grp_to_res_map
        self.res_to_res_grp_map = res_to_res_grp_map

    def _set_brand_pkg_sku_map(self) -> None:
        item = self.item_mst.copy()
        item = item[self.col_item]

        brand_pkg_sku_map = {}
        for brand, brand_df in item.groupby(by=self.item.brand):
            pkg_sku = {}
            for pkg, pkg_df in brand_df.groupby(by=self.item.pkg):
                for sku in pkg_df[self.item.sku]:
                    if pkg in pkg_sku:
                        pkg_sku[pkg].append(sku)
                    else:
                        pkg_sku[pkg] = [sku]
            brand_pkg_sku_map[brand] = pkg_sku

        self.brand_pkg_sku_map = brand_pkg_sku_map

    def _prep_item(self) -> pd.DataFrame:
        item = self.item_mst.copy()
        item = item[self.col_item]
        item = item.drop_duplicates()

        return item

    def _add_item_information(self, data: pd.DataFrame, item: pd.DataFrame) -> pd.DataFrame:
        if self.item.pkg in data.columns:
            merged = pd.merge(data, item, on=[self.item.sku, self.item.pkg], how='left')
        else:
            merged = pd.merge(data, item, on=[self.item.sku], how='left')
        merged = merged.fillna('-')

        return merged

    def _classify_sim_prod_possible_dmd(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        apply_idx = []
        for idx, res_grp, brand, pkg in zip(
                data.index, data[self.res.res_grp], data[self.item.brand], data[self.item.pkg]
        ):
            in_res_grp = self.sim_prod_cstr.get(res_grp, None)
            if in_res_grp is not None:
                in_brand = in_res_grp.get(brand, None)
                if in_brand is not None:
                    pkg = in_brand.get(pkg, None)
                    if pkg is not None:
                        apply_idx.append(idx)

        apply_dmd = data.loc[apply_idx]
        non_apply_dmd = data.loc[set(data.index) - set(apply_idx)]

        return apply_dmd, non_apply_dmd

    def _find_and_make_sim_product(self, apply_dmd_df: pd.DataFrame, all_dmd: pd.DataFrame) -> pd.DataFrame:
        # Search on each demand
        for idx, dmd in apply_dmd_df.iterrows():
            # Available package of simultaneously making
            sim_pkg = self.sim_prod_cstr[dmd[self.res.res_grp]][dmd[self.item.brand]][dmd[self.item.pkg]]

            # Select SKU to be made
            sku = self.select_sku_to_be_made(brand=dmd[self.item.brand], pkg=sim_pkg)

            # Search SKU if it is in other demand
            if sku is not None:
                sku_in_other_dmd = self.search_sku_in_other_dmd(sku=sku, dmd=dmd, apply_dmd=apply_dmd_df)
                if len(sku_in_other_dmd) > 0:
                    all_dmd = self.rearrange_sku_in_timeline(
                        dmd=dmd,
                        all_dmd=all_dmd,
                        cand_dmd=sku_in_other_dmd
                    )
                else:
                    all_dmd = self._arrange_sku_in_timeline(dmd=dmd, all_dmd=all_dmd, sku=sku)

        return all_dmd

    def _arrange_sku_in_timeline(self, dmd, all_dmd: pd.DataFrame, sku: str):
        # resource candidates (Resource list excluding apply demand resource)
        res_candidates = list(set(self.res_grp_to_res_map[dmd[self.res.res_grp]]) - {dmd[self.res.res]})

        if len(res_candidates) > 0:
            # apply
            dmd['fixed'] = True

            # Filter demand on available resource list
            avail_dmd = all_dmd[all_dmd[self.res.res].isin(res_candidates)]

            # Check that timeline is available on each resource
            flag_by_res = []
            for res, res_df in avail_dmd.groupby(by=self.res.res):
                flag = self._check_res_timeline_is_available(res_dmd=res_df, new_dmd=dmd)
                flag_by_res.append((res_df, flag))

            # Resource
            res_dmd, flag = self._decide_which_res_to_use(data=flag_by_res)

            # Update
            confirm_res = res_dmd[self.res.res].unique()[0]
            new_dmd = self._update_new_dmd(dmd=dmd, res=confirm_res, sku=sku)

            # Write log
            self.write_new_dmd_log(data=new_dmd)

            #
            all_dmd = self._arrange_sku(new_dmd=new_dmd, all_dmd=all_dmd, confirm_res=confirm_res, flag=flag)

        return all_dmd

    def _update_new_dmd(self, dmd: pd.Series, res, sku) -> pd.Series:
        new_dmd = dmd.copy()
        new_dmd[self.dmd.dmd] = 'SP' + new_dmd[self.dmd.dmd][2:]
        new_dmd[self.res.res] = res
        new_dmd[self.item.sku] = sku
        new_dmd[self.dmd.duration] = new_dmd[self.dmd.end_time] - new_dmd[self.dmd.start_time]
        new_dmd['fixed'] = True

        return new_dmd

    def _arrange_sku(self, new_dmd, all_dmd, confirm_res, flag):
        if flag:
            all_dmd = all_dmd.append(new_dmd)
        else:
            # Split all of demand by confirmed resource
            res_dmd = all_dmd[all_dmd[self.res.res] == confirm_res]
            non_res_dmd = all_dmd[all_dmd[self.res.res] != confirm_res]

            # split demand containing resource
            apply_start, apply_end = new_dmd[self.dmd.start_time], new_dmd[self.dmd.end_time]

            # split demand by needs to move or not
            move_dmd = res_dmd[res_dmd[self.dmd.end_time] > apply_start].copy()
            non_move_dmd = res_dmd[res_dmd[self.dmd.end_time] <= apply_start].copy()

            # Move timeline
            move_dmd = self._move_with_fixed_or_not(data=move_dmd, apply_end=apply_end)

            # Split
            move_dmd = self._apply_res_capa_on_timeline(data=move_dmd, res=confirm_res)
            # move_dmd = move_dmd.append(new_dmd).sort_values(by=self.dmd.start_time)

            res_dmd = pd.concat([non_move_dmd, move_dmd], axis=0, ignore_index=True).reset_index(drop=True)
            all_dmd = pd.concat([non_res_dmd, res_dmd], axis=0, ignore_index=True).reset_index(drop=True)
            all_dmd = all_dmd.append(new_dmd)

        return all_dmd

    def _move_with_fixed_or_not(self, data: pd.DataFrame, apply_end):
        data[self.dmd.duration] = data[self.dmd.end_time] - data[self.dmd.start_time]

        applied_df = pd.DataFrame()
        data = data.sort_values(by=self.dmd.start_time)
        fixed_data = data[data['fixed'] == 1]
        non_fixed_data = data[data['fixed'] == 0]
        time_start = apply_end
        impossible_period = []
        applied_df = applied_df.append(fixed_data)
        for impossible_start, impossible_end in zip(fixed_data[self.dmd.start_time], fixed_data[self.dmd.end_time]):
            impossible_period.append([impossible_start, impossible_end])

        if len(impossible_period) == 0:
            applied_df = data.reset_index(drop=True)
            return applied_df

        for idx, start_time, end_time in zip(
                non_fixed_data.index, non_fixed_data[self.dmd.start_time], non_fixed_data[self.dmd.end_time]
        ):
            dmd = non_fixed_data.loc[idx].copy()
            for i, (fixed_start, fixed_end) in enumerate(impossible_period):
                res_duration = end_time - start_time
                if start_time < time_start:
                    time_gap = time_start - start_time
                    start_time = time_start
                    end_time += time_gap

                if end_time <= fixed_start:
                    dmd[self.dmd.start_time] = start_time
                    dmd[self.dmd.end_time] = end_time
                    dmd[self.dmd.duration] = res_duration
                    applied_df = applied_df.append(dmd)
                    time_start = end_time
                    break
                else:
                    if start_time <= fixed_start:
                        dmd[self.dmd.start_time] = fixed_end
                        dmd[self.dmd.end_time] = fixed_end + res_duration
                        dmd[self.dmd.duration] = res_duration
                        applied_df = applied_df.append(dmd)
                        time_start = fixed_end + res_duration
                        break

                if i == len(impossible_period) - 1:
                    dmd[self.dmd.start_time] = start_time
                    dmd[self.dmd.end_time] = end_time
                    dmd[self.dmd.duration] = end_time - start_time
                    applied_df = applied_df.append(dmd)
                    time_start = end_time

        applied_df = applied_df.sort_values(by=self.dmd.start_time)
        applied_df = applied_df.reset_index(drop=True)

        return applied_df

    def _decide_which_res_to_use(self, data: list) -> Tuple[pd.DataFrame, bool]:
        available_cnt = sum([flag for _, flag in data])

        res_to_use = None
        if available_cnt == 0:
            end_time_list = []
            for i, (res_df, flag) in enumerate(data):
                end_time_list.append([i, res_df[self.dmd.end_time].max()])

            # Get minimum end time resource
            end_time_min = sorted(end_time_list, key=lambda x: x[1])[0]
            res_to_use = data[end_time_min[0]]

        elif available_cnt == 1:
            for res_df, flag in data:
                if flag:
                    res_to_use = (res_df, flag)

        else:
            tot_dur_list = []
            for i, (res_df, flag) in enumerate(data):
                total_duration = np.sum(res_df[self.dmd.end_time] - res_df[self.dmd.start_time])
                tot_dur_list.append([i, total_duration])

            # Get the resource of minimum total duration
            min_tot_dur = sorted(tot_dur_list, key=lambda x: x[1])[0]
            res_to_use = data[min_tot_dur[0]]

        return res_to_use

    def _check_res_timeline_is_available(self, res_dmd: pd.DataFrame, new_dmd: pd.DataFrame) -> bool:
        # sort current resource by timeline
        res_dmd = res_dmd.sort_values(by=self.dmd.start_time)

        # Make the timeline of current resource
        res_timeline = [[start, end] for start, end in zip(res_dmd[self.dmd.start_time], res_dmd[self.dmd.end_time])]

        # Connect timeline if time is continuous
        res_timeline = self.connect_continuous_capa(data=res_timeline)

        # check timeline is available
        add_timeline = [new_dmd[self.dmd.start_time], new_dmd[self.dmd.start_time]]
        flag = self.check_timeline_availability(res_timeline=res_timeline, add_timeline=add_timeline)

        return flag

    @staticmethod
    def check_timeline_availability(res_timeline, add_timeline):
        flag = True
        apply_start, apply_end = add_timeline[0], add_timeline[1]

        for start_time, end_time in res_timeline:
            if start_time >= apply_start:
                if start_time <= apply_end:
                    flag = False
            else:
                if apply_start <= end_time:
                    flag = False

        return flag

    def select_sku_to_be_made(self, brand, pkg) -> str:
        sku = None
        pkg_sku_list = self.brand_pkg_sku_map.get(brand, None)
        if pkg_sku_list is not None:
            sku_list = pkg_sku_list.get(pkg, None)
            if sku_list is not None:
                if len(sku_list) > 1:
                    sku = self.compare_curr_stock(sku_list=sku_list)
                elif len(sku_list) == 1:
                    sku = sku_list[0]

        return sku

    @staticmethod
    def compare_curr_stock(sku_list: list):
        # Todo: need to revise
        return sku_list[0]

    def search_sku_in_other_dmd(self, sku: str, dmd, apply_dmd):
        # resource candidates (Resource list excluding apply demand resource)
        res_candidates = list(set(self.res_grp_to_res_map[dmd[self.res.res_grp]]) - {dmd[self.res.res]})

        # Filter resources that can move timeline
        cand_dmd = apply_dmd[apply_dmd[self.res.res].isin(res_candidates)].copy()

        exist_df = cand_dmd[cand_dmd[self.item.sku] == sku]

        return exist_df

    def rearrange_sku_in_timeline(self, dmd, all_dmd, cand_dmd) -> pd.DataFrame:
        confirm_res = self.decide_rearrange_resource(data=cand_dmd)

        cand_dmd = cand_dmd[cand_dmd[self.res.res] == confirm_res].copy()

        flag = self.check_sku_is_movable(dmd=dmd, cand_dmd=cand_dmd)

        return all_dmd

    def decide_rearrange_resource(self, data: pd.DataFrame):
        res_list = list(data[self.res.res].unique())

        if len(res_list) == 1:
            confirm_res = res_list[0]
        else:
            res_end_time_list = []
            for res, res_df in data.groupby(by=self.res.res):
                res_end_time = res_df[self.dmd.end_time].max()
                res_end_time_list.append((res, res_end_time))

            confirm_res = sorted(res_end_time_list, key=lambda x: x[1])[0][0]

        return confirm_res

    def check_sku_is_movable(self, dmd: pd.Series, cand_dmd: pd.DataFrame):
        dmd_start_time, dmd_end_time = dmd[self.dmd.start_time], dmd[self.dmd.end_time]

        # Sort candidate demands bu demand start time
        cand_dmd = cand_dmd.sort_values(by=self.dmd.start_time)

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

    def set_res_capacity(self, data: pd.DataFrame):
        # Choose current plant
        data = data[data[self.res.plant] == self.plant]

        capa_col_list = []
        for i in range(self.work_day):
            for kind in ['d', 'n']:
                capa = self.res.res_capa + str(i + 1) + '_' + kind
                capa_col_list.append(capa)

        res_to_capa = {}
        for res, capa_df in data.groupby(by=self.res.res):
            days_capa = capa_df[capa_col_list].values.tolist()[0]
            days_capa = util.make_time_pair(data=days_capa)

            days_capa_list = []
            for day, (day_time, night_time) in enumerate(days_capa * self.schedule_weeks):
                start_time, end_time = util.calc_daily_avail_time(
                    day=day,
                    day_time=int(day_time * self.time_multiple),
                    night_time=int(night_time * self.time_multiple),
                )
                if start_time != end_time:
                    days_capa_list.append([start_time, end_time])

            days_capa_list = self.connect_continuous_capa(data=days_capa_list)
            res_to_capa[res] = days_capa_list

        self.res_to_capa = res_to_capa

    def _apply_res_capa_on_timeline(self, data, res):
        applied_data = pd.DataFrame()
        data = data.sort_values(by=self.dmd.start_time)
        res_capa_list = self.res_to_capa[res]
        time_start = 0
        for idx, start_time, end_time in zip(data.index, data[self.dmd.start_time], data[self.dmd.end_time]):
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
                        dmd[self.dmd.start_time] = start_time
                        dmd[self.dmd.end_time] = end_time
                        dmd[self.dmd.duration] = end_time - start_time
                        applied_data = applied_data.append(dmd)
                        time_start = end_time
                        break
                    else:
                        dmd[self.dmd.start_time] = start_time
                        dmd[self.dmd.end_time] = capa_end
                        dmd[self.dmd.duration] = capa_end - start_time
                        applied_data = applied_data.append(dmd)
                        start_time = capa_end

        applied_data = applied_data.reset_index(drop=True)

        return applied_data

    def write_new_dmd_log(self, data: pd.Series):
        log = f'Plant[{self.plant}] - Demand[{data[self.dmd.dmd]}] - Resource Group[{data[self.res.res_grp]}] -' \
              f'Resource[{data[self.res.res]}] - SKU[{data[self.item.sku]}] : Simultaneous production is added.'

        self.log.append(log)
