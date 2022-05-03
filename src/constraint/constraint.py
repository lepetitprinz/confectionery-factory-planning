import common.config as config

import numpy as np
import pandas as pd
import datetime as dt
from typing import Hashable, Dict, Union


class Human(object):
    key_human_capa = config.key_human_capa      # Human capacity
    key_human_usage = config.key_human_usage    # Human usage
    key_res_avail_time = config.key_res_avail_time

    col_dmd = config.col_dmd
    col_sku = config.col_sku
    col_pkg = config.col_pkg
    col_item = config.col_item
    col_plant = config.col_plant
    col_res = config.col_res
    col_res_grp = config.col_res_grp
    col_due_date = config.col_due_date
    col_res_capa = config.col_res_capa

    col_duration = config.col_duration
    col_end_time = config.col_end_time
    col_start_time = config.col_start_time

    use_col_item = [col_sku, col_item, col_pkg]
    use_col_dmd = [col_dmd, col_res_grp, col_sku, col_due_date]
    use_col_human_capa = [col_plant, 'floor', 'm_val', 'w_val']

    def __init__(
            self,
            plant: str,
            plant_start_time: dt.datetime,
            cstr: dict,
            item: pd.DataFrame,
            demand: pd.DataFrame,
            ):
        self.plant = plant
        self.plant_start_time = plant_start_time
        self.cstr = cstr
        self.item = item
        self.demand = demand

        self.floor_capa = {}
        self.work_day = 5
        self.prod_time_comp_standard = 'min'
        self.res_to_capa = {}

    def apply(self, data):
        # preprocess demand
        dmd_by_floor, non_apply_dmd = self.preprocess(data=data)

        # Resource to capacity map
        self.set_res_capacity(data=self.cstr[self.key_res_avail_time])

        result = pd.DataFrame()
        for floor, schedule_dmd in dmd_by_floor.items():
            curr_time = 0
            curr_capa = self.floor_capa[floor]
            fixed_schedule = pd.DataFrame()
            curr_prod_dmd = pd.DataFrame()
            while len(schedule_dmd) > 0:
                # Search
                cand_dmd = self.search_dmd_candidate(data=schedule_dmd)
                flag = self.check_prod_availability(curr_capa=curr_capa, data=cand_dmd)

                if flag:
                    # Confirm demand production
                    curr_capa, schedule_dmd, fixed_schedule, curr_prod_dmd = self.confirm_dmd_prod(
                        curr_capa=curr_capa,
                        cand_dmd=cand_dmd,
                        schedule_dmd=schedule_dmd,
                        fixed_schedule=fixed_schedule,
                        curr_prod_dmd=curr_prod_dmd,
                    )
                else:
                    # Finish the latest demand
                    curr_capa, curr_time, curr_prod_dmd = self.finish_latest_dmd(
                        curr_capa=curr_capa,
                        curr_time=curr_time,
                        curr_prod_dmd=curr_prod_dmd,
                    )
                    # Update timeline of remaining demands
                    self.update_remain_dmd_timeline(
                        schedule_dmd=schedule_dmd,
                        curr_time=curr_time,
                    )
            result = pd.concat([result, fixed_schedule])

        return result

    def set_res_capacity(self, data: pd.DataFrame):
        # Choose current plant
        data = data[data[self.col_plant] == self.plant]

        capa_col_list = []
        for i in range(self.work_day):
            capa_col = self.col_res_capa + str(i + 1)
            capa_col_list.append(capa_col)

        res_to_capa = {}
        for res, capa_df in data.groupby(by=self.col_res):
            res_to_capa[res] = capa_df[capa_col_list].values.tolist()[0]

        self.res_to_capa = res_to_capa

    def update_remain_dmd_timeline(self, schedule_dmd, curr_time):
        move_dmd = schedule_dmd[schedule_dmd[self.col_start_time] < curr_time].copy()
        non_move_dmd = schedule_dmd[schedule_dmd[self.col_start_time] >= curr_time].copy()

        move_dmd[self.col_end_time] = move_dmd[self.col_end_time] - move_dmd[self.col_start_time] + curr_time
        move_dmd[self.col_start_time] = curr_time

        move_dmd = self.apply_res_capa_on_timeline(data=move_dmd)
        revised_dmd = pd.concat([move_dmd, non_move_dmd], axis=0)

        return revised_dmd

    def apply_res_capa_on_timeline(self, data):
        return data

    def finish_latest_dmd(self, curr_capa, curr_time, curr_prod_dmd):
        latest_finish_dmd = curr_prod_dmd[curr_prod_dmd[self.col_end_time] == curr_prod_dmd[self.col_end_time].min()]

        # Update current time    # Todo: need to check the logic
        curr_time = latest_finish_dmd[self.col_end_time].values[0]

        # Update current human capacity
        finished_dmd_capa = latest_finish_dmd[['m_val', 'w_val']].values.sum(axis=0)
        curr_capa = curr_capa + finished_dmd_capa

        # Update current producing demand
        curr_prod_dmd = curr_prod_dmd.drop(labels=latest_finish_dmd.index)

        return curr_capa, curr_time, curr_prod_dmd

    def confirm_dmd_prod(self, curr_capa, cand_dmd: Union[pd.DataFrame, pd.Series], schedule_dmd: pd.DataFrame,
                         fixed_schedule: pd.DataFrame, curr_prod_dmd: pd.DataFrame):
        # dmd = cand_dmd[cand_dmd['kind'] == 'demand']
        dmd_usage = [cand_dmd['m_val'], cand_dmd['w_val']]

        # update current capacity
        curr_capa = curr_capa - dmd_usage

        # add production information
        fixed_schedule = fixed_schedule.append(cand_dmd)

        # Remove on candidates
        schedule_dmd = schedule_dmd[schedule_dmd[self.col_dmd] != cand_dmd[self.col_dmd]]

        # Update current production demand
        curr_prod_dmd = curr_prod_dmd.append(cand_dmd)

        return curr_capa, schedule_dmd, fixed_schedule, curr_prod_dmd

    def search_dmd_candidate(self, data: pd.DataFrame) -> pd.Series:
        min_start_dmd = data[data[self.col_start_time] == data[self.col_start_time].min()]

        if len(min_start_dmd[min_start_dmd['kind'] == 'demand']) == 1:
            return min_start_dmd
        else:
            min_start_dmd = min_start_dmd[min_start_dmd[self.col_due_date] == min_start_dmd[self.col_due_date].min()]

            if len(min_start_dmd) == 1:
                return min_start_dmd
            else:
                min_start_dmd = min_start_dmd.sort_values(by=self.col_duration, ascending=True)
                if self.prod_time_comp_standard == 'min':
                    min_start_dmd = min_start_dmd.iloc[0, :]
                else:
                    min_start_dmd = min_start_dmd.iloc[-1, :]

                return min_start_dmd

    def check_prod_availability(self, curr_capa, data: Union[pd.DataFrame, pd.Series]):
        flag = False
        # dmd = data[data['kind'] == 'demand']
        dmd_usage = [data['m_val'], data['w_val']]
        diff = curr_capa - dmd_usage
        if sum(diff < 0) == 0:
            flag = True

        return flag

    def preprocess(self, data: pd.DataFrame):
        # Preprocess the dataset
        item = self.prep_item()
        usage = self.prep_usage()
        data = self.prep_result(data=data)
        self.prep_capacity()

        apply_dmd, non_apply_dmd = self.separate_dmd_on_capa_existence(
            item=item,
            usage=usage,
            result=data
        )

        dmd_by_floor = self.set_dmd_list_by_floor(data=apply_dmd)

        return dmd_by_floor, non_apply_dmd

    def separate_dmd_on_capa_existence(self, item: pd.DataFrame, usage: pd.DataFrame, result: pd.DataFrame):
        # add due date information
        dmd = pd.merge(result, item, on=self.col_sku)

        # add item information (item / pkg)
        dmd = pd.merge(result, item, on=self.col_sku)

        # separate demands based on item trait
        apply_dmd = dmd[~dmd[self.col_pkg].isnull()]
        non_apply_dmd = dmd[dmd[self.col_pkg].isnull()]

        # add human usage information
        apply_dmd = pd.merge(apply_dmd, usage, how='left', on=[self.col_res_grp, self.col_item, self.col_pkg])

        # separate demands based on resource (internal/external)
        non_apply_dmd = pd.concat([non_apply_dmd, apply_dmd[apply_dmd['floor'].isnull()]])
        apply_dmd = apply_dmd[~apply_dmd['floor'].isnull()]

        drop_col = [self.col_plant, self.col_item, self.col_pkg]
        non_apply_dmd = non_apply_dmd.drop(columns=drop_col)
        apply_dmd = apply_dmd.drop(columns=drop_col)

        return apply_dmd, non_apply_dmd

    def prep_item(self):
        item = self.item
        item = item[self.use_col_item]
        item = item.drop_duplicates()

        return item

    def prep_capacity(self) -> None:
        capacity = self.cstr[self.key_human_capa]

        capacity = capacity[self.use_col_human_capa]
        capacity = capacity[capacity[self.col_plant] == self.plant]

        floor_capa = {}
        for floor, m_val, w_val in zip(capacity['floor'], capacity['m_val'], capacity['w_val']):
            floor_capa[floor] = np.array([m_val, w_val])

        self.floor_capa = floor_capa

    def prep_usage(self):
        usage = self.cstr[self.key_human_usage]

        # Temp
        usage.columns = [col.lower() for col in usage.columns]

        usage = usage[usage[self.col_plant] == self.plant]

        # Change data type
        usage[self.col_pkg] = usage[self.col_pkg].astype(str)
        usage[self.col_res_grp] = usage[self.col_res_grp].astype(str)

        # Temp
        usage[self.col_pkg] = [val.zfill(5) for val in usage[self.col_pkg].values]

        return usage

    def prep_result(self, data: pd.DataFrame) -> pd.DataFrame:
        demand = self.demand[[self.col_dmd, self.col_due_date]]

        data = pd.merge(data, demand, how='left', on=self.col_dmd)
        data[self.col_due_date] = data[self.col_due_date].fillna(0)

        return data

    def set_dmd_list_by_floor(self, data: pd.DataFrame) -> Dict[Hashable, pd.DataFrame]:
        data['duration'] = data[self.col_end_time] - data[self.col_start_time]

        dmd_by_floor = {}
        for floor, floor_df in data.groupby(by='floor'):
            dmd_by_floor[floor] = floor_df

        return dmd_by_floor

    def get_prod_start_time(self):
        pass

    def determine_prod_dmd(self):
        pass