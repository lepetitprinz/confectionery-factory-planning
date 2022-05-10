import common.config as config
import common.util as util

import numpy as np
import pandas as pd
import datetime as dt
from typing import Hashable, Dict, Union, Tuple


class Human(object):
    key_human_capa = config.key_human_capa      # Human capacity
    key_human_usage = config.key_human_usage    # Human usage
    key_res_avail_time = config.key_res_avail_time

    # Column: Item
    col_pkg = config.col_pkg
    col_sku = config.col_sku

    # Column: Demand
    col_dmd = config.col_dmd
    col_item = config.col_item
    col_plant = config.col_plant
    col_start_time = config.col_start_time
    col_end_time = config.col_end_time
    col_duration = config.col_duration

    # Column: Resource
    col_res = config.col_res
    col_res_grp = config.col_res_grp
    col_due_date = config.col_due_date
    col_res_capa = config.col_res_capa

    # Column: Constraint
    col_m_val = config.col_m_val
    col_w_val = config.col_w_val
    col_floor = config.col_floor

    use_col_item = [col_sku, col_item, col_pkg]
    use_col_dmd = [col_dmd, col_res_grp, col_sku, col_due_date]
    use_col_human_capa = [col_plant, col_floor, col_m_val, col_w_val]

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
        self.res_to_capa = {}
        self.prod_time_comp_standard = 'min'

        # Time instance attribute
        self.work_day = 5          # Monday ~ Friday
        self.sec_of_day = 86400    # Seconds of 1 day
        self.time_multiple = 60    # Minute -> Seconds
        self.schedule_weeks = 17
        self.plant_start_hour = 0

    def apply(self, data):
        # preprocess demand
        capa_apply_dmd, non_apply_dmd = self.preprocess(data=data)

        # Resource to capacity map
        self.set_res_capacity(data=self.cstr[self.key_res_avail_time])

        result = pd.DataFrame()
        for floor, schedule_dmd in capa_apply_dmd.items():
            # Initialization
            curr_capa = self.floor_capa[floor]
            confirmed_schedule = pd.DataFrame()    # confirmed schedule
            curr_prod_dmd = pd.DataFrame()     # Current producing demand

            while len(schedule_dmd) > 0:
                # Search demand candidate
                candidate_dmd = self.search_candidate_dmd(data=schedule_dmd, curr_prod_dmd=curr_prod_dmd)

                # Check that capacity is available
                if candidate_dmd is not None:
                    flag = self.check_available_capa(curr_capa=curr_capa, data=candidate_dmd)
                else:
                    flag = False

                if flag:
                    # Confirm demand production
                    curr_capa, schedule_dmd, confirmed_schedule, curr_prod_dmd = self.confirm_dmd_prod(
                        curr_capa=curr_capa,
                        candidate_dmd=candidate_dmd,
                        schedule_dmd=schedule_dmd,
                        confirmed_schedule=confirmed_schedule,
                        curr_prod_dmd=curr_prod_dmd,
                    )
                else:
                    # Finish the latest demand
                    if len(curr_prod_dmd) > 0:
                        curr_capa, curr_time, curr_prod_dmd = self.finish_latest_dmd(
                            curr_capa=curr_capa,
                            curr_prod_dmd=curr_prod_dmd,
                        )
                        # Update timeline of remaining demands
                        schedule_dmd = self.update_remain_dmd_timeline(
                            schedule_dmd=schedule_dmd,
                            curr_time=curr_time,
                        )
            result = pd.concat([result, confirmed_schedule])

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

    def update_remain_dmd_timeline(self, schedule_dmd: pd.DataFrame, curr_time: int) -> pd.DataFrame:
        # Classify timeline movement
        move_dmd, non_move_dmd = self.classify_dmd_move_or_not(data=schedule_dmd, time=curr_time)

        # Move timeline by each resource
        time_moved_dmd = self.move_timeline(data=move_dmd, time=curr_time)

        if len(time_moved_dmd) > 0:
            time_moved_dmd = self.apply_res_capa_on_timeline(data=time_moved_dmd)

        revised_dmd = pd.concat([time_moved_dmd, non_move_dmd], axis=0)
        revised_dmd = revised_dmd.reset_index(drop=True)

        return revised_dmd

    # Classify the demand if timeline will be moved or not
    def classify_dmd_move_or_not(self, data: pd.DataFrame, time: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        moving_res_list = data[data[self.col_start_time] < time][self.col_res].unique()
        move_dmd = data[data[self.col_res].isin(moving_res_list)].copy()
        non_move_dmd = data[~data[self.col_res].isin(moving_res_list)].copy()

        return move_dmd, non_move_dmd

    # Move timeline by each resource
    def move_timeline(self, data: pd.DataFrame, time: int) -> pd.DataFrame:
        time_moved_dmd = pd.DataFrame()
        for res, res_df in data.groupby(by=self.col_res):
            time_gap = max(0, time - res_df[self.col_start_time].min())
            res_df[self.col_end_time] = res_df[self.col_end_time] + int(time_gap)
            res_df[self.col_start_time] = res_df[self.col_start_time] + int(time_gap)
            time_moved_dmd = time_moved_dmd.append(res_df)

        return time_moved_dmd

    def apply_res_capa_on_timeline(self, data):
        applied_data = pd.DataFrame()

        for res, grp in data.groupby(self.col_res):
            grp = grp.sort_values(by=self.col_start_time)
            temp = grp.copy()
            res_capa_list = self.res_to_capa[res]
            time_gap = 0
            for idx, start_time, end_time in zip(grp.index, grp[self.col_start_time], grp[self.col_end_time]):
                start_time += time_gap
                end_time += time_gap
                for i, (capa_start, capa_end) in enumerate(res_capa_list):
                    if start_time < capa_start:
                        temp[self.col_start_time] = temp[self.col_start_time] + capa_start - start_time
                        temp[self.col_end_time] = temp[self.col_end_time] + capa_start - start_time
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
                                time_gap = time_gap + dmd_bf[self.col_end_time] - capa_end
                                dmd_bf[self.col_end_time] = capa_end
                                dmd_bf[self.col_duration] = dmd_bf[self.col_end_time] - dmd_bf[self.col_start_time]
                                applied_data = applied_data.append(dmd_bf)

                                # demand (after)
                                dmd_af = temp.loc[idx].copy()
                                dmd_af[self.col_start_time] = res_capa_list[i+1][0]
                                dmd_af[self.col_end_time] = dmd_af[self.col_start_time] + int(time_gap)
                                dmd_af[self.col_duration] = dmd_af[self.col_end_time] - dmd_af[self.col_start_time]
                                applied_data = applied_data.append(dmd_af)
                                break

        applied_data = applied_data.reset_index(drop=True)

        return applied_data

    def finish_latest_dmd(self, curr_capa, curr_prod_dmd: pd.DataFrame):
        latest_finish_dmd = curr_prod_dmd[curr_prod_dmd[self.col_end_time] == curr_prod_dmd[self.col_end_time].min()]

        # Update current time    # Todo: need to check the logic
        curr_time = latest_finish_dmd[self.col_end_time].values[0]

        # Update human capacity
        finished_dmd_capa = latest_finish_dmd[[self.col_m_val, self.col_w_val]].values.sum(axis=0)
        curr_capa = curr_capa + finished_dmd_capa

        # Drop the demand that is finished
        curr_prod_dmd = curr_prod_dmd.drop(labels=latest_finish_dmd.index)

        return curr_capa, curr_time, curr_prod_dmd

    def confirm_dmd_prod(self, curr_capa, candidate_dmd: pd.Series, schedule_dmd: pd.DataFrame,
                         confirmed_schedule: pd.DataFrame, curr_prod_dmd: pd.DataFrame):
        # Get demand usage
        dmd_usage = [candidate_dmd[self.col_m_val], candidate_dmd[self.col_w_val]]

        # update current capacity
        curr_capa = curr_capa - dmd_usage

        # add production information
        confirmed_schedule = confirmed_schedule.append(candidate_dmd)

        # Remove confirmed demand from candidates
        schedule_dmd = schedule_dmd.drop(labels=candidate_dmd.name)

        # Update current production demand
        curr_prod_dmd = curr_prod_dmd.append(candidate_dmd)

        return curr_capa, schedule_dmd, confirmed_schedule, curr_prod_dmd

    def search_candidate_dmd(self, data: pd.DataFrame, curr_prod_dmd: pd.DataFrame) -> Union[pd.Series, None]:
        # Filter resources that are used now
        if len(curr_prod_dmd) > 0:
            data = self.exclude_current_use_resource(data=data, curr_prod_dmd=curr_prod_dmd)

            # if all of resource are used then none of demand can be made
            if len(data) == 0:
                return None

        # Search the first start demand on product start time
        fst_start_dmd = data[data[self.col_start_time] == data[self.col_start_time].min()]

        if len(fst_start_dmd) > 1:
            # Search the first start demand on product due date
            fst_start_dmd = fst_start_dmd[fst_start_dmd[self.col_due_date] == fst_start_dmd[self.col_due_date].min()]

            if len(fst_start_dmd) > 1:
                # Search the first start demand on product duration
                fst_start_dmd = fst_start_dmd.sort_values(by=self.col_duration, ascending=True)
                fst_start_dmd = self.first_start_by_duration(data=fst_start_dmd)

        # Convert dataframe to series if result is dataframe
        if isinstance(fst_start_dmd, pd.DataFrame):
            fst_start_dmd = fst_start_dmd.squeeze()

        return fst_start_dmd

    # Exclude resources that are currently used
    def exclude_current_use_resource(self, data, curr_prod_dmd) -> pd.DataFrame:
        using_resource = curr_prod_dmd[self.col_res].unique()

        # Filter resources that are used now
        data = data[~data[self.col_res].isin(using_resource)].copy()

        return data

    def first_start_by_duration(self, data: pd.DataFrame):
        if self.prod_time_comp_standard == 'min':
            data = data.iloc[0]
        elif self.prod_time_comp_standard == 'max':
            data = data.iloc[-1]

        return data

    def check_available_capa(self, data: Union[pd.DataFrame, pd.Series], curr_capa):
        flag = False

        dmd_usage = None
        if isinstance(data, pd.DataFrame):
            dmd_usage = data[[self.col_m_val, self.col_w_val]].values[0]
        elif isinstance(data, pd.Series):
            dmd_usage = [data[self.col_m_val], data[self.col_w_val]]

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

        capa_apply_dmd, non_apply_dmd = self.separate_dmd_on_capa_existence(
            item=item,
            usage=usage,
            result=data
        )

        capa_apply_dmd = self.set_dmd_list_by_floor(data=capa_apply_dmd)

        return capa_apply_dmd, non_apply_dmd

    def separate_dmd_on_capa_existence(self, item: pd.DataFrame, usage: pd.DataFrame, result: pd.DataFrame):
        # add item information (item / pkg)
        dmd = pd.merge(result, item, on=self.col_sku)

        # separate demands based on item trait
        apply_dmd = dmd[~dmd[self.col_pkg].isnull()]
        non_apply_dmd = dmd[dmd[self.col_pkg].isnull()]

        # add human usage information
        apply_dmd = pd.merge(apply_dmd, usage, how='left', on=[self.col_res_grp, self.col_item, self.col_pkg])

        # separate demands based on resource (internal/external)
        non_apply_dmd = pd.concat([non_apply_dmd, apply_dmd[apply_dmd[self.col_floor].isnull()]])
        apply_dmd = apply_dmd[~apply_dmd[self.col_floor].isnull()]

        drop_col = [self.col_plant, self.col_item, self.col_pkg]
        non_apply_dmd = non_apply_dmd.drop(columns=drop_col)
        apply_dmd = apply_dmd.drop(columns=drop_col)

        return apply_dmd, non_apply_dmd

    def prep_item(self) -> pd.DataFrame:
        item = self.item
        item = item[self.use_col_item]
        item = item.drop_duplicates()

        return item

    # Preprocess the human usage dataset
    def prep_capacity(self) -> None:
        capacity = self.cstr[self.key_human_capa]

        capacity = capacity[self.use_col_human_capa]
        capacity = capacity[capacity[self.col_plant] == self.plant]

        floor_capa = {}
        for floor, m_val, w_val in zip(capacity[self.col_floor], capacity[self.col_m_val], capacity[self.col_w_val]):
            floor_capa[floor] = np.array([m_val, w_val])

        self.floor_capa = floor_capa

    # Preprocess the human usage dataset
    def prep_usage(self) -> pd.DataFrame:
        usage = self.cstr[self.key_human_usage]

        # Temp
        usage.columns = [col.lower() for col in usage.columns]

        usage = usage[usage[self.col_plant] == self.plant].copy()

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
        data[self.col_duration] = data[self.col_end_time] - data[self.col_start_time]

        dmd_by_floor = {}
        for floor, floor_df in data.groupby(by=self.col_floor):
            dmd_by_floor[floor] = floor_df

        return dmd_by_floor
