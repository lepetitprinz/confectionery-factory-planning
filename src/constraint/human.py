import common.util as util
import common.config as config
from common.name import Key, Demand, Item, Resource, Constraint, Post

import numpy as np
import pandas as pd
import datetime as dt
from copy import deepcopy
from typing import Hashable, Dict, Union, Tuple


class Human(object):
    def __init__(
            self,
            plant: str,
            plant_start_time: dt.datetime,
            cstr: dict,
            item: pd.DataFrame,
            demand: pd.DataFrame,
            calendar: pd.DataFrame,
            res_to_res_grp: dict
    ):
        self.plant = plant
        self.plant_start_time = plant_start_time
        self.cstr_mst = cstr
        self.item_mst = item
        self.dmd_mst = demand
        self.calendar = calendar

        # Name instance attribute
        self.key = Key()
        self.dmd = Demand()
        self.res = Resource()
        self.item = Item()
        self.cstr = Constraint()
        self.post = Post()

        # Column usage
        self.col_item = [self.item.sku, self.item.item, self.item.pkg]
        self.col_dmd = [self.dmd.dmd, self.res.res_grp, self.item.sku, self.dmd.due_date]
        self.col_human_capa = [self.res.plant, self.cstr.floor, 'week', self.cstr.m_capa, self.cstr.w_capa]

        # Data hash map
        self.floor_capa = {}
        self.res_to_capa = {}
        self.res_grp_to_floor = {}
        self.res_to_res_grp = res_to_res_grp
        self.prod_time_comp_standard = 'min'

        # Time instance attribute
        self.work_day = config.work_day
        self.sec_of_day = config.sec_of_day   # Seconds of 1 day
        self.time_multiple = config.time_multiple    # Minute -> Seconds
        self.schedule_weeks = config.schedule_weeks
        self.plant_start_hour = config.plant_start_hour
        self.yymmdd_to_week = {}

        # save the timeline change log
        self.log = []
        self.capa_profile = {}
        self.capa_profile_dtl = []

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
            curr_time = 0    # Start time
            curr_week = self.yymmdd_to_week[self.plant_start_time.strftime('%Y%m%d')]   # Start week
            curr_capa = deepcopy(self.floor_capa[floor][curr_week])    # Initial human capacity
            curr_prod_dmd = pd.DataFrame()  # Current producing demand
            confirmed_schedule = pd.DataFrame()  # confirmed schedule

            while len(schedule_dmd) > 0:
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
                        floor=floor,
                        curr_capa=curr_capa,
                        candidate_dmd=candidate_dmd,
                        schedule_dmd=schedule_dmd,
                        confirmed_schedule=confirmed_schedule,
                        curr_prod_dmd=curr_prod_dmd,
                    )
                else:
                    # Finish the latest demand
                    if len(curr_prod_dmd) > 0:
                        curr_capa, curr_time, curr_week, curr_prod_dmd = self.finish_latest_dmd(
                            floor=floor,
                            curr_capa=curr_capa,
                            curr_time=curr_time,
                            curr_week=curr_week,
                            curr_prod_dmd=curr_prod_dmd,
                        )
                        # Update timeline of remaining demands
                        schedule_dmd = self.update_remain_dmd_timeline(
                            schedule_dmd=schedule_dmd,
                            curr_time=curr_time,
                        )
            result = pd.concat([result, confirmed_schedule])

        result = pd.concat([result, non_apply_dmd])

        # log
        log = sorted(list(set(self.log)))

        # Capacity profile
        capa_profile = self.set_capa_profile()
        capa_profile_dn = self.make_capa_profile_day_night(data=capa_profile)
        capa_profile_dtl = self.set_capa_profile_dtl()  # Capa profile detail

        return result, log, capa_profile_dn, capa_profile_dtl

    def set_res_capacity(self, data: pd.DataFrame):
        # Choose current plant
        data = data[data[self.res.plant] == self.plant].copy()

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

    def update_remain_dmd_timeline(self, schedule_dmd: pd.DataFrame, curr_time: int) -> pd.DataFrame:
        # Classify timeline movement
        move_dmd, non_move_dmd = self.classify_dmd_move_or_not(data=schedule_dmd, time=curr_time)

        # Move timeline by each resource
        time_moved_dmd = self.move_timeline(data=move_dmd, time=curr_time)

        if len(time_moved_dmd) > 0:
            # Write timeline changing log
            self.write_timeline_chg_log(data=time_moved_dmd)

            # apply resource capacity on timeline
            time_moved_dmd = self.apply_res_capa_on_timeline(data=time_moved_dmd)

            # Connect continuous timeline of each demand
            time_moved_dmd = self.connect_continuous_dmd(data=time_moved_dmd)

        revised_dmd = pd.concat([time_moved_dmd, non_move_dmd], axis=0)
        revised_dmd = revised_dmd.reset_index(drop=True)

        return revised_dmd

    def connect_continuous_dmd(self, data: pd.DataFrame):
        revised_data = pd.DataFrame()

        for dmd, dmd_df in data.groupby(by=self.dmd.dmd):
            for item, item_df in dmd_df.groupby(by=self.item.sku):
                for res_grp, res_grp_df in item_df.groupby(by=self.res.res_grp):
                    for res, res_df in res_grp_df.groupby(by=self.res.res):
                        if len(res_df) > 1:
                            # Sort demand on start time
                            res_df = res_df.sort_values(by=self.dmd.start_time)
                            timeline_list = res_df[[self.dmd.start_time, self.dmd.end_time]].values.tolist()

                            # Connect the capacity if timeline is continuous
                            timeline_connected = self.connect_continuous_capa(data=timeline_list)

                            # Remake demand dataframe using connected timeline
                            for stime, etime in timeline_connected:
                                common = res_df.iloc[0].copy()
                                dmd_series = self.update_connected_timeline(
                                    common=common,
                                    dmd=dmd,
                                    item=item,
                                    res_grp=res_grp,
                                    res=res,
                                    start_time=stime,
                                    end_time=etime
                                )
                                revised_data = revised_data.append(dmd_series)
                        else:
                            revised_data = revised_data.append(res_df)

        revised_data = revised_data.sort_values(by=self.dmd.dmd)
        revised_data = revised_data.reset_index(drop=True)

        return revised_data

    def connect_continuous_dmd_bak(self, data: pd.DataFrame):
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
                    dmd_series = self.update_connected_timeline_bak(data=dmd_copy, start_time=stime, end_time=etime)
                    revised_data = revised_data.append(dmd_series)
            else:
                revised_data = revised_data.append(dmd_df)

        revised_data = revised_data.sort_values(by=self.dmd.dmd)
        revised_data = revised_data.reset_index(drop=True)

        return revised_data

    def update_connected_timeline_bak(self, data: pd.Series, start_time, end_time):
        data[self.dmd.start_time] = start_time
        data[self.dmd.end_time] = end_time
        data[self.dmd.duration] = end_time - start_time

        return data

    def update_connected_timeline(self, common: pd.Series, dmd, item, res_grp, res, start_time, end_time):
        common[self.dmd.dmd] = dmd
        common[self.item.sku] = item
        common[self.res.res_grp] = res_grp
        common[self.res.res] = res
        common[self.dmd.start_time] = start_time
        common[self.dmd.end_time] = end_time
        common[self.dmd.duration] = end_time - start_time

        return common

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

    def finish_latest_dmd(self, floor: Hashable, curr_capa: Tuple[int, int], curr_time, curr_week: str,
                          curr_prod_dmd: pd.DataFrame):
        latest_finish_dmd = curr_prod_dmd[curr_prod_dmd[self.dmd.end_time] == curr_prod_dmd[self.dmd.end_time].min()]

        # Update capacity profile
        if floor in self.capa_profile:
            self.capa_profile[floor].append([curr_time] + list(curr_capa))
        else:
            self.capa_profile[floor] = [[curr_time] + list(curr_capa)]

        # Update current time    # Todo: need to check the logic
        curr_time = latest_finish_dmd[self.dmd.end_time].values[0]

        # Check weekly change
        update_week = self.yymmdd_to_week[
            dt.datetime.strftime(self.plant_start_time + dt.timedelta(seconds=curr_time), '%Y%m%d')
        ]
        if update_week != curr_week:
            curr_capa = self.update_capa_on_next_week(
                floor=floor,
                curr_week=curr_week,
                update_week=update_week,
                curr_capa=curr_capa
            )
            # Update week information (next week)
            curr_week = update_week

        # Update human capacity
        finished_dmd_capa = latest_finish_dmd[[self.cstr.m_capa, self.cstr.w_capa]].values.sum(axis=0)
        curr_capa = curr_capa + finished_dmd_capa

        # Drop the demand that is finished
        curr_prod_dmd = curr_prod_dmd.drop(labels=latest_finish_dmd.index)
        curr_prod_dmd = curr_prod_dmd.reset_index(drop=True)

        return curr_capa, curr_time, curr_week, curr_prod_dmd

    def update_capa_on_next_week(self, floor, curr_week, update_week, curr_capa):
        curr_week_capa = self.floor_capa[floor].get(
            curr_week,
            self.floor_capa[floor][list(self.floor_capa[floor])[-1]]
        )
        next_week_capa = self.floor_capa[floor].get(
            update_week,
            self.floor_capa[floor][list(self.floor_capa[floor])[-1]]
        )
        capa_diff = next_week_capa - curr_week_capa
        curr_capa += capa_diff

        return curr_capa

    def confirm_dmd_prod(self, floor, curr_capa, candidate_dmd: pd.Series,
                         schedule_dmd: pd.DataFrame, confirmed_schedule: pd.DataFrame, curr_prod_dmd: pd.DataFrame):
        # Get demand usage
        dmd_usage = [candidate_dmd[self.cstr.m_capa], candidate_dmd[self.cstr.w_capa]]

        # Update current capacity
        curr_capa = curr_capa - dmd_usage

        # Update capacity profile
        profile = [floor, candidate_dmd[self.res.res_grp], candidate_dmd[self.res.res], candidate_dmd[self.item.sku],
                   candidate_dmd['starttime'], candidate_dmd['endtime'], candidate_dmd[self.cstr.m_capa],
                   candidate_dmd[self.cstr.w_capa], candidate_dmd[self.dmd.duration]]
        self.capa_profile_dtl.append(profile)

        # Add production information
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
        self.prep_calendar()
        self.prep_capacity()

        apply_dmd, non_apply_dmd = self.add_item_capa_info(
            data=data, item=item, usage=usage
        )

        apply_dmd = self.set_dmd_list_by_floor(data=apply_dmd)

        return apply_dmd, non_apply_dmd

    def prep_item(self) -> pd.DataFrame:
        item = self.item_mst
        item = item[self.col_item]
        item = item.drop_duplicates()

        return item

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

        res_grp_floor = usage[[self.res.res_grp, self.cstr.floor]].drop_duplicates()

        self.res_grp_to_floor = {
            res_grp: floor for res_grp, floor in zip(res_grp_floor[self.res.res_grp], res_grp_floor[self.cstr.floor])
        }

        return usage

    def prep_result(self, data: pd.DataFrame) -> pd.DataFrame:
        demand = self.dmd_mst[[self.dmd.dmd, self.dmd.due_date]]

        data = pd.merge(data, demand, how='left', on=self.dmd.dmd)
        data[self.dmd.due_date] = data[self.dmd.due_date].fillna(0)

        return data

    # Preprocess the human usage dataset
    def prep_capacity(self) -> None:
        capacity = self.cstr_mst[self.key.human_capa]

        capacity = capacity[self.col_human_capa]
        capacity = capacity[capacity[self.res.plant] == self.plant]

        floor_capa = {}
        for floor, floor_df in capacity.groupby(by=self.cstr.floor):
            week_to_capa = {}
            for week, m_val, w_val in zip(
                floor_df['week'], floor_df[self.cstr.m_capa], floor_df[self.cstr.w_capa]
            ):
                week_to_capa[week] = np.array([m_val, w_val])
            floor_capa[floor] = week_to_capa

        self.floor_capa = floor_capa

    def prep_calendar(self):
        self.yymmdd_to_week = {day: week for day, week in zip(self.calendar['yymmdd'], self.calendar['week'])}

    def add_item_capa_info(self, data, item, usage):
        # classify the demand
        non_apply_dmd = data[~data[self.res.res_grp].isin(list(self.res_grp_to_floor.keys()))].copy()
        apply_dmd = data[data[self.res.res_grp].isin(list(self.res_grp_to_floor.keys()))].copy()

        # add item information (item / pkg)
        apply_dmd = pd.merge(apply_dmd, item, on=self.item.sku, how='left')

        # add human usage information
        usage = usage.drop_duplicates()  # Exception
        apply_dmd = pd.merge(apply_dmd, usage, how='left', on=[self.res.res_grp, self.item.item, self.item.pkg])

        apply_dmd[self.res.plant] = self.plant
        apply_dmd[self.cstr.m_capa] = apply_dmd[self.cstr.m_capa].fillna(0)
        apply_dmd[self.cstr.w_capa] = apply_dmd[self.cstr.w_capa].fillna(0)
        apply_dmd[self.item.pkg] = apply_dmd[self.item.pkg].fillna('99999')
        apply_dmd[self.cstr.floor] = [self.res_grp_to_floor[self.res_to_res_grp[res]] for res
                                      in apply_dmd[self.res.res]]

        return apply_dmd, non_apply_dmd

    def separate_dmd_on_capa_existence(self, item: pd.DataFrame, usage: pd.DataFrame, result: pd.DataFrame):
        # add item information (item / pkg)
        dmd = pd.merge(result, item, on=self.item.sku, how='left')

        # separate demands based on item trait
        apply_dmd = dmd[~dmd[self.item.pkg].isnull()]
        non_apply_dmd = dmd[dmd[self.item.pkg].isnull()]

        # add human usage information
        usage = usage.drop_duplicates()
        apply_dmd = pd.merge(apply_dmd, usage, how='left', on=[self.res.res_grp, self.item.item, self.item.pkg])

        # separate demands based on resource (internal/external)
        non_apply_dmd = pd.concat([non_apply_dmd, apply_dmd[apply_dmd[self.cstr.floor].isnull()]])
        apply_dmd = apply_dmd[~apply_dmd[self.cstr.floor].isnull()]

        drop_col = [self.res.plant, self.item.item, self.item.pkg]
        non_apply_dmd = non_apply_dmd.drop(columns=drop_col)
        apply_dmd = apply_dmd.drop(columns=drop_col)

        return apply_dmd, non_apply_dmd

    def set_dmd_list_by_floor(self, data: pd.DataFrame) -> Dict[Hashable, pd.DataFrame]:
        data[self.dmd.duration] = data[self.dmd.end_time] - data[self.dmd.start_time]

        dmd_by_floor = {}
        for floor, floor_df in data.groupby(by=self.cstr.floor):
            dmd_by_floor[floor] = floor_df

        return dmd_by_floor

    def write_timeline_chg_log(self, data: pd.DataFrame):
        for dmd, floor, res_grp, res, sku in zip(
                data[self.dmd.dmd], data[self.cstr.floor], data[self.res.res_grp],
                data[self.res.res], data[self.item.sku]
        ):
            log = f'Demand[{dmd}] - Plant[{self.plant}] - Floor[{floor}] - Resource Group[{res_grp}] - ' \
                  f'Resource[{res}] - SKU[{sku}]: Constraint of human capacity is applied.'
            self.log.append(log)

    def set_capa_profile_dtl(self):
        profile = pd.DataFrame(
            self.capa_profile_dtl,
            columns=[self.cstr.floor, self.res.res_grp, self.res.res, self.item.sku, self.post.from_time,
                     self.post.to_time, self.post.use_m_capa, self.post.use_w_capa, self.post.res_use_capa
                     ])
        profile[self.res.plant] = self.plant
        profile[self.post.from_time] = [
            self.plant_start_time + dt.timedelta(seconds=time) for time in profile[self.post.from_time]
        ]
        profile[self.post.to_time] = [
            self.plant_start_time + dt.timedelta(seconds=time) for time in profile[self.post.to_time]
        ]
        profile[self.post.from_yymmdd] = profile[self.post.from_time].dt.strftime('%Y%m%d')
        profile[self.post.to_yymmdd] = profile[self.post.to_time].dt.strftime('%Y%m%d')
        profile[self.post.from_time] = profile[self.post.from_time].dt.strftime('%Y%m%d %h%M%s')
        profile[self.post.to_time] = profile[self.post.to_time].dt.strftime('%Y%m%d %h%M%s')
        profile[self.post.from_time] = profile[self.post.from_time] \
            .str.replace(' ', '').str.replace(':', '').str.replace('-', '')
        profile[self.post.to_time] = profile[self.post.to_time] \
            .str.replace(' ', '').str.replace(':', '').str.replace('-', '')

        tot_m_capa, tot_w_capa = [], []
        for floor, yymmdd in zip(profile[self.cstr.floor], profile[self.post.from_yymmdd]):
            m_capa, w_capa = self.floor_capa[floor].get(
                self.yymmdd_to_week[yymmdd],
                self.floor_capa[floor][list(self.floor_capa[floor])[-1]]
            )
            tot_m_capa.append(m_capa)
            tot_w_capa.append(w_capa)

        profile[self.post.tot_m_capa] = tot_m_capa
        profile[self.post.tot_w_capa] = tot_w_capa

        profile = pd.merge(
            profile,
            self.item_mst[[self.item.sku, self.item.sku_nm, self.item.item_type]].drop_duplicates(),
            on=self.item.sku,
            how='left'
        )

        return profile

    def make_capa_profile_day_night(self, data: pd.DataFrame):
        profile_list = []
        for floor, stime, etime, tot_m_capa, tot_w_capa, use_m_capa, use_w_capa in zip(
                data[self.cstr.floor], data[self.post.from_time], data[self.post.to_time], data[self.post.tot_m_capa],
                data[self.post.tot_w_capa], data[self.post.use_m_capa], data[self.post.use_w_capa]
        ):
            # Date range
            days = pd.date_range(stime, etime + dt.timedelta(days=1), freq='D')
            days = [dt.datetime.strftime(time, '%Y%m%d') for time in days]
            day_capa = [tot_m_capa, tot_w_capa, use_m_capa, use_w_capa]

            # Classify the day and night
            s_dn = self.classify_day_night(hour=stime.hour)  # Classify the time as day or night
            # e_dn = self.classify_day_night(hour=etime.hour)    # Classify the time as day or night

            for i, day in enumerate(days):
                if i == 0:
                    if s_dn == 'D':
                        profile_list.append([floor, day, 'D'] + day_capa)
                        profile_list.append([floor, day, 'N'] + day_capa)
                    else:
                        profile_list.append([floor, day, 'N'] + day_capa)
                elif i == len(day) - 1:
                    if s_dn == 'D':
                        profile_list.append([floor, day, 'D'] + day_capa)
                    else:
                        profile_list.append([floor, day, 'D'] + day_capa)
                        profile_list.append([floor, day, 'N'] + day_capa)
                else:
                    profile_list.append([floor, day, 'D'] + day_capa)
                    profile_list.append([floor, day, 'N'] + day_capa)

        profile_df = pd.DataFrame(
            profile_list,
            columns=[self.cstr.floor, 'yymmdd', self.post.time_idx, self.post.tot_m_capa, self.post.tot_w_capa,
                     self.post.use_m_capa, self.post.use_w_capa]
        )

        profile_max = profile_df.groupby(by=[self.cstr.floor, 'yymmdd', self.post.time_idx]).max().reset_index()
        profile_max[self.post.avail_m_capa] = profile_max[self.post.tot_m_capa] - profile_max[self.post.use_m_capa]
        profile_max[self.post.avail_w_capa] = profile_max[self.post.tot_w_capa] - profile_max[self.post.use_w_capa]

        return profile_max

    @staticmethod
    def classify_day_night(hour):
        if (hour >= 0) and (hour < 12):
            day_night = 'D'
        else:
            day_night = 'N'

        return day_night

    def set_capa_profile(self):
        capa_profile = []
        for floor, floor_capa_profile in self.capa_profile.items():
            from_time, prev_m_capa, prev_w_capa = floor_capa_profile[0]
            for time, m_capa, w_capa in floor_capa_profile[1:]:
                # Get week information
                week = self.yymmdd_to_week[
                    dt.datetime.strftime(self.plant_start_time + dt.timedelta(seconds=time), '%Y%m%d')
                ]
                # Get weekly human capacity
                tot_m_capa, tot_w_capa = self.floor_capa[floor].get(
                    week,
                    self.floor_capa[floor][list(self.floor_capa[floor])[-1]]
                )
                if (m_capa == prev_m_capa) and (w_capa == prev_w_capa):
                    continue
                else:
                    capa_profile.append([floor, from_time, time, tot_m_capa, tot_w_capa,
                                         tot_m_capa - prev_m_capa, tot_w_capa - prev_w_capa])
                    prev_m_capa = m_capa
                    prev_w_capa = w_capa
                    from_time = time

        profile = pd.DataFrame(
            capa_profile,
            columns=[self.cstr.floor, self.post.from_time, self.post.to_time, self.post.tot_m_capa,
                     self.post.tot_w_capa, self.post.use_m_capa, self.post.use_w_capa])
        profile[self.res.plant] = self.plant
        profile[self.post.from_time] = [
            self.plant_start_time + dt.timedelta(seconds=time) for time in profile[self.post.from_time]
        ]
        profile[self.post.to_time] = [
            self.plant_start_time + dt.timedelta(seconds=time) for time in profile[self.post.to_time]
        ]
        profile[self.post.use_m_capa] = profile[self.post.use_m_capa].astype(int)
        profile[self.post.use_w_capa] = profile[self.post.use_w_capa].astype(int)

        return profile
