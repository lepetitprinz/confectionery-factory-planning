import common.util as util
from common.name import Key, Demand, Item, Resource, Constraint

import numpy as np
import pandas as pd
import datetime as dt
from typing import Hashable, Dict, Union, Tuple


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
        self.sim_prod_cstr = sim_prod_cstr

        # Dataset
        self.org_data = org_data
        self.item_mst = org_data[self.key.res][self.key.item]
        self.cstr_mst = org_data[self.key.cstr]
        self.dmd_schedule = demand

        self.res_to_capa = {}
        self.res_grp_to_res_map = {}
        self.res_to_res_grp_map = {}
        self.brand_pkg_sku_map = {}

        self.work_day = 5          # Monday ~ Friday
        self.sec_of_day = 86400    # Seconds of 1 day
        self.time_multiple = 60    # Minute -> Seconds
        self.schedule_weeks = 17
        self.plant_start_hour = 0

    def apply(self, data: pd.DataFrame):
        # Data Preprocessing
        data = self.preprocess(data=data)

        # Resource to capacity map
        self.set_res_capacity(data=self.cstr_mst[self.key.res_avail_time])

        # Classify demand if possible to product demand simultaneously
        apply_dmd, non_apply_dmd = self.classify_sim_prod_possible_dmd(data=data)

        #
        self.find_and_make_sim_product(apply_dmd_df=apply_dmd, all_dmd=data)

    def preprocess(self, data: pd.DataFrame):
        res = self.prep_res()
        item = self.prep_item()
        data = self.add_item_information(data=data, item=item)

        # Mapping data
        self.set_res_grp_res_map(res=res)
        self.set_brand_pkg_sku_map()

        data['fixed'] = False

        return data

    def prep_res(self):
        # Resource master
        res_mst = self.org_data[self.key.res][self.key.res_grp]
        res_mst[self.res.res] = res_mst[self.res.res]
        res_mst[self.res.res_grp] = res_mst[self.res.res_grp]

        res_mst = res_mst[[self.res.plant, self.res.res_grp, self.res.res]].drop_duplicates()

        res_mst = res_mst[res_mst[self.res.plant] == self.plant]

        return res_mst

    def set_res_grp_res_map(self, res: pd.DataFrame):
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

    def set_brand_pkg_sku_map(self):
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

    def prep_item(self) -> pd.DataFrame:
        item = self.item_mst.copy()
        item = item[self.col_item]
        item = item.drop_duplicates()

        return item

    def add_item_information(self, data: pd.DataFrame, item: pd.DataFrame):
        merged = pd.merge(data, item, on=self.item.sku, how='left')
        merged = merged.fillna('-')

        return merged

    def classify_sim_prod_possible_dmd(self, data: pd.DataFrame):
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

    def find_and_make_sim_product(self, apply_dmd_df: pd.DataFrame, all_dmd: pd.DataFrame):
        # Search on each demand
        # for idx, res_grp, res, brand, pkg in zip(
        #         dmd.index, dmd[self.res.res_grp], dmd[self.res.res],
        #         dmd[self.item.brand], dmd[self.item.pkg]
        # ):
        for idx, dmd in apply_dmd_df.iterrows():
            # Available package of simultaneously making
            sim_pkg = self.sim_prod_cstr[dmd[self.res.res_grp]][dmd[self.item.brand]][dmd[self.item.pkg]]

            # Select SKU to be made
            sku = self.select_sku_to_be_made(brand=dmd[self.item.brand], pkg=sim_pkg)

            # Search SKU if it is in other demand
            if sku is not None:
                sku_in_other_dmd = self.search_sku_in_other_dmd(
                    data=apply_dmd_df,
                    res_grp=dmd[self.res.res_grp],
                    sku=sku
                )
                if len(sku_in_other_dmd) > 0:
                    self.rearrange_sku_in_timeline(dmd=dmd, all_dmd=all_dmd, sku=sku)
                else:
                    all_dmd = self.arrange_sku_in_timeline(dmd=dmd, all_dmd=all_dmd, sku=sku)

    def arrange_sku_in_timeline(self, dmd, all_dmd: pd.DataFrame, sku: str):
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
                flag = self.check_res_timeline_is_available(res_dmd=res_df, new_dmd=dmd)
                flag_by_res.append((res_df, flag))

            # Resource
            res_dmd, flag = self.decide_which_res_to_use(data=flag_by_res)

            # Update
            confirm_res = res_dmd[self.res.res].unique()[0]
            new_dmd = self.update_new_dmd(dmd=dmd, res=confirm_res, sku=sku)

            #
            new_res_dmd = self.arrange_sku(new_dmd=new_dmd, all_dmd=all_dmd, confirm_res=confirm_res, flag=flag)
            print("")

        return all_dmd

    def update_new_dmd(self, dmd: pd.Series, res, sku) -> pd.Series:
        new_dmd = dmd.copy()
        new_dmd[self.dmd.dmd] = 'SP' + new_dmd[self.dmd.dmd][2:]
        new_dmd[self.res.res] = res
        new_dmd[self.item.sku] = sku
        new_dmd['fixed'] = True

        return new_dmd

    def arrange_sku(self, new_dmd, all_dmd, confirm_res, flag):
        if flag:
            pass
        else:
            # Split all of demand by confirmed resource
            res_dmd = all_dmd[all_dmd[self.res.res] == confirm_res]
            non_res_dmd = all_dmd[all_dmd[self.res.res] != confirm_res]

            # split demand containing resource
            apply_start, apply_end = new_dmd[self.dmd.start_time], new_dmd[self.dmd.end_time]

            # split demand by needs to move or not
            move_dmd = res_dmd[res_dmd[self.dmd.end_time] > apply_start]
            non_move_dmd = res_dmd[res_dmd[self.dmd.end_time] <= apply_start]

            time_gap = apply_end - move_dmd[self.dmd.start_time].min()

            # Move timeline
            move_dmd[self.dmd.start_time] = move_dmd[self.dmd.start_time] + time_gap
            move_dmd[self.dmd.end_time] = move_dmd[self.dmd.end_time] + time_gap

            # Split
            move_dmd = self.apply_res_capa_on_timeline(data=move_dmd)
            # move_dmd = move_dmd.append(new_dmd).sort_values(by=self.dmd.start_time)

            res_dmd = pd.concat([non_move_dmd, move_dmd], axis=0, ignore_index=True).reset_index(drop=True)
            all_dmd = pd.concat([non_res_dmd, res_dmd], axis=0, ignore_index=True).reset_index(drop=True)
            all_dmd = all_dmd.append(new_dmd)

            return all_dmd

    def decide_which_res_to_use(self, data: list) -> Tuple[pd.DataFrame, bool]:
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

    def check_res_timeline_is_available(self, res_dmd: pd.DataFrame, new_dmd: pd.DataFrame) -> bool:
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

    def search_sku_in_other_dmd(self, data, res_grp, sku):
        exist_df = data[(data[self.res.res_grp] == res_grp) & (data[self.item.sku] == sku)]

        return exist_df

    def rearrange_sku_in_timeline(self, dmd, all_dmd, sku):
        pass

    @staticmethod
    def connect_continuous_capa(data: list):
        result = []
        idx = 0
        add_idx = 1
        curr_capa = []
        while idx + add_idx < len(data):
            curr_capa = data[idx]
            next_capa = data[idx + add_idx]
            if curr_capa[1] == next_capa[0]:
                curr_capa[1] = next_capa[1]
                add_idx += 1
            else:
                result.append(curr_capa)
                idx += add_idx
                add_idx = 1

        result.append(curr_capa)

        return result

    def set_res_capacity(self, data: pd.DataFrame):
        # Choose current plant
        data = data[data[self.res.plant] == self.plant]

        capa_col_list = []
        for i in range(self.work_day):
            capa_col = self.res.res_capa + str(i + 1)
            capa_col_list.append(capa_col)

        res_to_capa = {}
        for res, capa_df in data.groupby(by=self.res.res):
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

    def apply_res_capa_on_timeline(self, data):
        applied_data = pd.DataFrame()

        for res, grp in data.groupby(self.res.res):
            grp = grp.sort_values(by=self.dmd.start_time)
            temp = grp.copy()
            res_capa_list = self.res_to_capa[res]
            time_gap = 0
            for idx, start_time, end_time in zip(grp.index, grp[self.dmd.start_time], grp[self.dmd.end_time]):
                start_time += time_gap
                end_time += time_gap
                for i, (capa_start, capa_end) in enumerate(res_capa_list):
                    if start_time < capa_start:
                        temp[self.dmd.start_time] = temp[self.dmd.start_time] + capa_start - start_time
                        temp[self.dmd.end_time] = temp[self.dmd.end_time] + capa_start - start_time
                    else:
                        if start_time > capa_end:
                            continue
                        else:
                            if end_time < capa_end:
                                applied_data = applied_data.append(temp.loc[idx])
                                break
                            else:
                                # demand (before)
                                dmd_bf = temp.loc[idx].copy()
                                time_gap = time_gap + dmd_bf[self.dmd.end_time] - capa_end
                                dmd_bf[self.dmd.end_time] = capa_end
                                dmd_bf[self.dmd.duration] = dmd_bf[self.dmd.end_time] - dmd_bf[self.dmd.start_time]
                                applied_data = applied_data.append(dmd_bf)

                                # demand (after)
                                dmd_af = temp.loc[idx].copy()
                                dmd_af[self.dmd.start_time] = res_capa_list[i+1][0]
                                dmd_af[self.dmd.end_time] = dmd_af[self.dmd.start_time] + int(time_gap)
                                dmd_af[self.dmd.duration] = dmd_af[self.dmd.end_time] - dmd_af[self.dmd.start_time]
                                applied_data = applied_data.append(dmd_af)
                                break

        applied_data = applied_data.reset_index(drop=True)

        return applied_data