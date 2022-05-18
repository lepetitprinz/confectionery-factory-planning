import common.util as util
from common.name import Key, Demand, Item, Resource, Constraint

import numpy as np
import pandas as pd
import datetime as dt
from typing import Hashable, Dict, Union, Tuple


class Human(object):
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
        self.cstr_mst = cstr
        self.item_mst = item
        self.dmd_mst = demand

        # Name instance attribute
        self.key = Key()
        self.dmd = Demand()
        self.res = Resource()
        self.item = Item()
        self.cstr = Constraint()

        # Column usage
        self.col_item = [self.item.sku, self.item.item, self.item.pkg]
        self.col_dmd = [self.dmd.dmd, self.res.res_grp, self.item.sku, self.dmd.due_date]
        self.col_human_capa = [self.res.plant, self.cstr.floor, self.cstr.m_capa, self.cstr.w_capa]

        self.floor_capa = {}
        self.res_to_capa = {}
        self.prod_time_comp_standard = 'min'

        # Time instance attribute
        self.work_day = 5          # Monday ~ Friday
        self.sec_of_day = 86400    # Seconds of 1 day
        self.time_multiple = 60    # Minute -> Seconds
        self.schedule_weeks = 100
        self.plant_start_hour = 0

    def apply(self, data):
        # preprocess demand
        capa_apply_dmd, non_apply_dmd = self.preprocess(data=data)

        # Resource to capacity map
        self.set_res_capacity(data=self.cstr_mst[self.key.res_avail_time])

        result = pd.DataFrame()
        for floor, schedule_dmd in capa_apply_dmd.items():
            print("---------------------------")
            print(f"Applying Floor: {floor}")
            print("---------------------------")

            # Initialization
            curr_capa = self.floor_capa[floor]
            confirmed_schedule = pd.DataFrame()    # confirmed schedule
            curr_prod_dmd = pd.DataFrame()     # Current producing demand

            while len(schedule_dmd) > 0:
                # print(f'Remaining Schedule: {len(schedule_dmd)}')
                # print(f'Current Capacity: {curr_capa}')

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

        result = pd.concat([result, non_apply_dmd])

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

            # connect
            time_moved_dmd = self.connect_continuous_dmd(data=time_moved_dmd)

        revised_dmd = pd.concat([time_moved_dmd, non_move_dmd], axis=0)
        revised_dmd = revised_dmd.reset_index(drop=True)

        return revised_dmd

    def connect_continuous_dmd(self, data: pd.DataFrame):
        revised_data = pd.DataFrame()

        for dmd, dmd_df in data.groupby(by=self.dmd.dmd):
            if len(dmd_df) > 1:
                # Sort demand on start time
                dmd_df = dmd_df.sort_values(by=self.dmd.start_time)
                timeline_list = dmd_df[[self.dmd.start_time, self.dmd.end_time]].values.tolist()

                # Connect the capacity if timeline is continuous
                timeline_connected = self.connect_continuous_capa(data=timeline_list)

                # Remake demand dataframe using connected timeline

                for stime, etime in timeline_connected:
                    dmd_copy = dmd_df.iloc[0].copy()
                    dmd_series = self.update_connected_timeline(data=dmd_copy, start_time=stime, end_time=etime)
                    revised_data = revised_data.append(dmd_series)
            else:
                revised_data = revised_data.append(dmd_df)

        revised_data = revised_data.sort_values(by=self.dmd.dmd)
        revised_data = revised_data.reset_index(drop=True)

        return revised_data

    def update_connected_timeline(self, data: pd.Series, start_time, end_time):
        data[self.dmd.start_time] = start_time
        data[self.dmd.end_time] = end_time
        data[self.dmd.duration] = end_time - start_time

        return data

    # Classify the demand if timeline will be moved or not
    def classify_dmd_move_or_not(self, data: pd.DataFrame, time: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        moving_res_list = data[data[self.dmd.start_time] < time][self.res.res].unique()
        move_dmd = data[data[self.res.res].isin(moving_res_list)].copy()
        non_move_dmd = data[~data[self.res.res].isin(moving_res_list)].copy()

        return move_dmd, non_move_dmd

    # Move timeline by each resource
    def move_timeline(self, data: pd.DataFrame, time: int) -> pd.DataFrame:
        time_moved_dmd = pd.DataFrame()
        for res, res_df in data.groupby(by=self.res.res):
            time_gap = max(0, time - res_df[self.dmd.start_time].min())
            res_df[self.dmd.end_time] = res_df[self.dmd.end_time] + int(time_gap)
            res_df[self.dmd.start_time] = res_df[self.dmd.start_time] + int(time_gap)
            time_moved_dmd = time_moved_dmd.append(res_df)

        return time_moved_dmd

    def apply_res_capa_on_timeline(self, data):
        applied_data = pd.DataFrame()

        # Dataset group by resource
        for res, grp in data.groupby(self.res.res):
            grp = grp.sort_values(by=self.dmd.start_time)
            res_capa_list = self.res_to_capa[res]
            time_start = 0
            for idx, start_time, end_time in zip(grp.index, grp[self.dmd.start_time], grp[self.dmd.end_time]):
                if time_start > start_time:
                    time_gap = time_start - start_time
                    start_time, end_time = time_start, end_time + time_gap
                for capa_start, capa_end in res_capa_list:
                    dmd = grp.loc[idx].copy()
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

    def apply_res_capa_on_timeline_bak(self, data):
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

    def finish_latest_dmd(self, curr_capa, curr_prod_dmd: pd.DataFrame):
        latest_finish_dmd = curr_prod_dmd[curr_prod_dmd[self.dmd.end_time] == curr_prod_dmd[self.dmd.end_time].min()]

        # Update current time    # Todo: need to check the logic
        curr_time = latest_finish_dmd[self.dmd.end_time].values[0]

        # Update human capacity
        finished_dmd_capa = latest_finish_dmd[[self.cstr.m_capa, self.cstr.w_capa]].values.sum(axis=0)
        curr_capa = curr_capa + finished_dmd_capa

        # Drop the demand that is finished
        curr_prod_dmd = curr_prod_dmd.drop(labels=latest_finish_dmd.index)
        curr_prod_dmd = curr_prod_dmd.reset_index(drop=True)

        return curr_capa, curr_time, curr_prod_dmd

    def confirm_dmd_prod(self, curr_capa, candidate_dmd: pd.Series, schedule_dmd: pd.DataFrame,
                         confirmed_schedule: pd.DataFrame, curr_prod_dmd: pd.DataFrame):
        # Get demand usage
        dmd_usage = [candidate_dmd[self.cstr.m_capa], candidate_dmd[self.cstr.w_capa]]

        # update current capacity
        curr_capa = curr_capa - dmd_usage

        # add production information
        confirmed_schedule = confirmed_schedule.append(candidate_dmd)

        # Remove confirmed demand from candidates
        schedule_dmd = schedule_dmd.drop(labels=candidate_dmd.name)

        # Update current production demand
        curr_prod_dmd = curr_prod_dmd.append(candidate_dmd)
        curr_prod_dmd = curr_prod_dmd.reset_index(drop=True)

        return curr_capa, schedule_dmd, confirmed_schedule, curr_prod_dmd

    def search_candidate_dmd(self, data: pd.DataFrame, curr_prod_dmd: pd.DataFrame) -> Union[pd.Series, None]:
        # Filter resources that are used now
        if len(curr_prod_dmd) > 0:
            data = self.exclude_current_use_resource(data=data, curr_prod_dmd=curr_prod_dmd)

            # if all of resource are used then none of demand can be made
            if len(data) == 0:
                return None

        # Search the first start demand on product start time
        fst_start_dmd = data[data[self.dmd.start_time] == data[self.dmd.start_time].min()]

        if len(fst_start_dmd) > 1:
            # Search the first start demand on product due date
            fst_start_dmd = fst_start_dmd[fst_start_dmd[self.dmd.due_date] == fst_start_dmd[self.dmd.due_date].min()]

            if len(fst_start_dmd) > 1:
                # Search the first start demand on product duration
                fst_start_dmd = fst_start_dmd.sort_values(by=self.dmd.duration, ascending=True)
                fst_start_dmd = self.first_start_by_duration(data=fst_start_dmd)

        # Convert dataframe to series if result is dataframe
        if isinstance(fst_start_dmd, pd.DataFrame):
            fst_start_dmd = fst_start_dmd.squeeze()

        return fst_start_dmd

    # Exclude resources that are currently used
    def exclude_current_use_resource(self, data, curr_prod_dmd) -> pd.DataFrame:
        using_resource = curr_prod_dmd[self.res.res].unique()

        # Filter resources that are used now
        data = data[~data[self.res.res].isin(using_resource)].copy()

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
            dmd_usage = data[[self.cstr.m_capa, self.cstr.w_capa]].values[0]
        elif isinstance(data, pd.Series):
            dmd_usage = [data[self.cstr.m_capa], data[self.cstr.w_capa]]

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
        dmd = pd.merge(result, item, on=self.item.sku)

        # separate demands based on item trait
        apply_dmd = dmd[~dmd[self.item.pkg].isnull()]
        non_apply_dmd = dmd[dmd[self.item.pkg].isnull()]

        # add human usage information
        apply_dmd = pd.merge(apply_dmd, usage, how='left', on=[self.res.res_grp, self.item.item, self.item.pkg])

        # separate demands based on resource (internal/external)
        non_apply_dmd = pd.concat([non_apply_dmd, apply_dmd[apply_dmd[self.cstr.floor].isnull()]])
        apply_dmd = apply_dmd[~apply_dmd[self.cstr.floor].isnull()]

        drop_col = [self.res.plant, self.item.item, self.item.pkg]
        non_apply_dmd = non_apply_dmd.drop(columns=drop_col)
        apply_dmd = apply_dmd.drop(columns=drop_col)

        return apply_dmd, non_apply_dmd

    def prep_item(self) -> pd.DataFrame:
        item = self.item_mst
        item = item[self.col_item]
        item = item.drop_duplicates()

        return item

    # Preprocess the human usage dataset
    def prep_capacity(self) -> None:
        capacity = self.cstr_mst[self.key.human_capa]

        capacity = capacity[self.col_human_capa]
        capacity = capacity[capacity[self.res.plant] == self.plant]

        floor_capa = {}
        for floor, m_val, w_val in zip(
                capacity[self.cstr.floor], capacity[self.cstr.m_capa], capacity[self.cstr.w_capa]
        ):
            floor_capa[floor] = np.array([m_val, w_val])

        self.floor_capa = floor_capa

    # Preprocess the human usage dataset
    def prep_usage(self) -> pd.DataFrame:
        usage = self.cstr_mst[self.key.human_usage]

        # Temp
        usage.columns = [col.lower() for col in usage.columns]

        usage = usage[usage[self.res.plant] == self.plant].copy()

        # Change data type
        usage[self.item.pkg] = usage[self.item.pkg].astype(str)
        usage[self.res.res_grp] = usage[self.res.res_grp].astype(str)

        # Temp
        usage[self.item.pkg] = [val.zfill(5) for val in usage[self.item.pkg].values]

        return usage

    def prep_result(self, data: pd.DataFrame) -> pd.DataFrame:
        demand = self.dmd_mst[[self.dmd.dmd, self.dmd.due_date]]

        data = pd.merge(data, demand, how='left', on=self.dmd.dmd)
        data[self.dmd.due_date] = data[self.dmd.due_date].fillna(0)

        return data

    def set_dmd_list_by_floor(self, data: pd.DataFrame) -> Dict[Hashable, pd.DataFrame]:
        data[self.dmd.duration] = data[self.dmd.end_time] - data[self.dmd.start_time]

        dmd_by_floor = {}
        for floor, floor_df in data.groupby(by=self.cstr.floor):
            dmd_by_floor[floor] = floor_df

        return dmd_by_floor