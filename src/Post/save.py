import common.util as util
import common.config as config
from common.name import Key, Demand, Resource, Post

import os
import numpy as np
import pandas as pd
import datetime as dt


class Save(object):
    col_fp_version_id = 'fp_vrsn_id'
    col_fp_version_seq = 'fp_vrsn_seq'

    def __init__(
            self,
            data,
            io,
            query,
            plant: str,
            fp_seq: str,
            fp_name: str,
            fp_version: str,
            res_avail_time,
            res_grp_mst,
            res_to_res_grp
    ):
        self._data = data

        # Class instance attribute
        self.io = io
        self.query = query

        self._plant = plant
        self.fp_seq = fp_seq
        self._fp_name = fp_name
        self.fp_version = fp_version
        self._project_cd = config.project_cd

        # Name instance attribute
        self._key = Key()
        self._dmd = Demand()
        self._res = Resource()
        self._post = Post()

        self._schedule_range = []
        self._split_hour = dt.timedelta(hours=12)
        self._time_idx_map = {'D': 0, 'N': 1}
        self._time_multiple = config.time_multiple     # Minute -> Seconds
        self._schedule_weeks = config.schedule_weeks
        self._sec_of_half_day = config.sec_of_day // 2

        # Data instance attribute
        self._res_grp_mst = res_grp_mst
        self._res_to_res_grp = res_to_res_grp
        self._res_avail_time = res_avail_time
        self._res_day_avail_time = {}

        self.res_idx_col = [
            self._res.res, 'day', 'res_capa_val', self._post.res_use_capa, self._post.res_avail_capa,
            self._post.res_unavail_capa, self._post.res_jc_capa
        ]

    def to_csv(self, path, name: str) -> None:
        save_dir = os.path.join(path, 'opt', 'csv', self.fp_version)
        util.make_dir(path=save_dir)

        # Save the optimization result
        self._data.to_csv(os.path.join(save_dir, name + '_' + self._fp_name + '.csv'), index=False, encoding='cp949')

    ##################
    # Resource Status
    ##################
    def res_status(self):
        self._prep_res_info()

        # Get resource timeline
        timeline = self._get_res_timeline()

        # Preprocess resource status dataset
        res_status = self._prep_res_status(data=timeline)

        # Divide demand / job change result
        res_final = self._divide_dmd_jc_data(data=res_status)

        # Set resource capacity
        res_final = self._set_res_capacity(data=res_final)

        # Split resource capacity to day and night
        res_final = self._split_res_capa_day_night(data=res_final)

        # Set the resource unavailable time
        res_final = self._set_res_unavail_time(data=res_final)

        # Resource available time
        res_final[self._post.res_avail_capa] = res_final['res_capa_val'] - res_final[self._post.res_use_capa] \
                                               - res_final[self._post.res_jc_capa]

        # Fill unused timeline
        # res_final = self.fill_unused_capa_info(data=res_final)

        # Add information
        res_final = self.add_model_info(data=res_final)

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self._plant}
        self.io.delete_from_db(sql=self.query.del_res_status_result(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=res_final, tb_name='M4E_O402050')

    def _prep_res_info(self):
        start_day = dt.datetime.strptime(self.fp_version[3:7] + '-' + self.fp_version[7:10] + '-1', "%Y-W%W-%w")
        end_day = start_day + dt.timedelta(days=self._schedule_weeks)
        schedule_range = pd.date_range(start=start_day, end=end_day, freq='D').tolist()
        self._schedule_range = [date for date in schedule_range]

        res_day_avail_time = {}
        for res, avail_time_list in self._res_avail_time.items():
            day_to_avail_time = {}
            for i, (day_time, night_time) in enumerate(avail_time_list):
                # day_time = self.calc_day_night_res_capacity(day=i, time_idx_type='D', capacity=avail_time * 60)
                # night_time = self.calc_day_night_res_capacity(day=i, time_idx_type='N', capacity=avail_time * 60)
                day_time = day_time * self._time_multiple
                night_time = night_time * self._time_multiple
                avail_time_day_night = {
                    'D': pd.Series(
                        [res, i, day_time, 0, day_time, self._sec_of_half_day - day_time, 0],
                        index=self.res_idx_col
                    ),
                    'N': pd.Series(
                        [res, i, night_time, 0, night_time, self._sec_of_half_day - night_time, 0],
                        index=self.res_idx_col
                    )
                }
                day_to_avail_time[i] = avail_time_day_night
            res_day_avail_time[res] = day_to_avail_time

        self._res_day_avail_time = res_day_avail_time

    def _get_res_timeline(self):
        timeline = []
        for res, res_df in self._data.groupby(self._res.res):
            for kind, kind_df in res_df.groupby('kind'):
                for start, end in zip(kind_df['start'], kind_df['end']):
                    timeline.extend(self._calc_res_duration(res=res, kind=kind, start=start, end=end))

        return timeline

    def _prep_res_status(self, data):
        res_status = pd.DataFrame(
            data,
            columns=[self._res.res, 'kind', self._post.date, self._post.time_idx, self._dmd.duration]
        )
        res_status[self._post.date] = res_status[self._post.date].dt.strftime('%Y%m%d')
        res_status[self._dmd.duration] = res_status[self._dmd.duration] / np.timedelta64(1, 's')

        res_status = res_status.groupby(by=[self._res.res, 'kind', self._post.date, self._post.time_idx]) \
            .sum() \
            .reset_index()

        return res_status

    def _divide_dmd_jc_data(self, data):
        # Demand
        dmd = data[data['kind'] == 'demand'].copy()
        dmd = dmd.rename(columns={self._dmd.duration: self._post.res_use_capa})
        dmd = dmd.drop(columns=['kind'])

        # Job change
        jc = data[data['kind'] == 'job_change'].copy()
        jc = jc.rename(columns={self._dmd.duration: self._post.res_jc_capa})
        jc = jc.drop(columns=['kind'])

        result = pd.merge(dmd, jc, how='left', on=[self._res.res, self._post.date, self._post.time_idx]).fillna(0)

        return result

    def _set_res_capacity(self, data):
        # Resource usage
        data['day'] = [dt.datetime.strptime(day, '%Y%m%d').weekday() for day in data[self._post.date]]

        res_capa = []
        for res, day, time_idx in zip(data[self._res.res], data['day'], data[self._post.time_idx]):
            res_avail_time = self._res_avail_time[res]
            # res_capa.append(res_avail_time[day][self.time_idx_map[time_idx]] * 60)
            res_capa.append(sum(res_avail_time[day]) * 60)    # Todo

        data[self._res.res_capa] = res_capa

        return data

    def _split_res_capa_day_night(self, data):
        res_capa_val = []
        for res, day, time_idx, capacity in zip(
               data[self._res.res], data['day'], data[self._post.time_idx], data[self._res.res_capa]):
            res_avail_time = self._res_avail_time[res][day][self._time_idx_map[time_idx]] * 60
            res_capa_val.append(res_avail_time)
        data['res_capa_val'] = res_capa_val

        return data

    def _set_res_unavail_time(self, data):
        res_unavail_val = []
        for day, capacity in zip(data['day'], data['res_capa_val']):
            res_unavail_val.append(self._sec_of_half_day - capacity)

        data[self._post.res_unavail_capa] = res_unavail_val

        return data

    def add_model_info(self, data):
        res_grp_mst = self._res_grp_mst[self._res_grp_mst[self._res.plant] == self._plant].copy()
        res_grp_mst = res_grp_mst[[self._res.res, 'res_type_cd']]
        data = pd.merge(data, res_grp_mst, how='left', on=self._res.res).fillna('UNDEFINED')
        data = data.rename(columns={'res_type_cd': 'capa_type_cd'})

        data[self._res.plant] = self._plant
        data[self._res.res_grp] = [self._res_to_res_grp.get(res_cd, 'UNDEFINED')
                                   for res_cd in data[self._res.res]]
        data = self.add_version_info(data=data)

        data = data.drop(columns=[self._res.res_capa, 'day'])

        return data

    def _calc_res_duration(self, res, kind, start, end):
        timeline = []
        start_day = dt.datetime.strptime(dt.datetime.strftime(start, '%Y%m%d'), '%Y%m%d')
        start_time = dt.timedelta(hours=start.hour, minutes=start.minute, seconds=start.second)
        end_day = dt.datetime.strptime(dt.datetime.strftime(end, '%Y%m%d'), '%Y%m%d')
        end_time = dt.timedelta(hours=end.hour, minutes=end.minute, seconds=end.second)

        if end_time == dt.timedelta(seconds=0):
            end_day = end_day - dt.timedelta(days=1)
            end_time = dt.timedelta(hours=24)

        diff_day = (end_day - start_day).days

        if diff_day == 0:
            duration_day = dt.timedelta(hours=0)
            duration_night = dt.timedelta(hours=0)
            if end_time < self._split_hour:
                duration_day = end_time - start_time
            elif start_time > self._split_hour:
                duration_night = end_time - start_time
            else:
                duration_day = self._split_hour - start_time
                duration_night = end_time - self._split_hour

            timeline.append([res, kind, start_day, 'D', duration_day])
            timeline.append([res, kind, start_day, 'N', duration_night])

        elif diff_day == 1:
            prev_duration_day, prev_duration_night = self._calc_timeline_prev(start_time=start_time)
            next_duration_day, next_duration_night = self._calc_timeline_next(end_time=end_time)

            timeline.append([res, kind, start_day, 'D', prev_duration_day])
            timeline.append([res, kind, start_day, 'N', prev_duration_night])
            timeline.append([res, kind, end_day, 'D', next_duration_day])
            timeline.append([res, kind, end_day, 'N', next_duration_night])

        else:
            prev_duration_day, prev_duration_night = self._calc_timeline_prev(start_time=start_time)
            next_duration_day, next_duration_night = self._calc_timeline_next(end_time=end_time)

            timeline.append([res, kind, start_day, 'D', prev_duration_day])
            timeline.append([res, kind, start_day, 'N', prev_duration_night])
            timeline.append([res, kind, end_day, 'D', next_duration_day])
            timeline.append([res, kind, end_day, 'N', next_duration_night])

            for i in range(diff_day - 1):
                timeline.append(
                    [res, kind, start_day + dt.timedelta(days=i + 1), 'D', self._split_hour])
                timeline.append(
                    [res, kind, start_day + dt.timedelta(days=i + 1), 'N', self._split_hour])

        return timeline

    def _calc_timeline_prev(self, start_time):
        # Previous day
        duration_day = dt.timedelta(hours=0)
        if start_time < self._split_hour:
            duration_day = self._split_hour - start_time
            duration_night = self._split_hour
        else:
            duration_night = dt.timedelta(hours=24) - start_time

        return duration_day, duration_night

    def _calc_timeline_next(self, end_time):
        duration_night = dt.timedelta(hours=0)
        if end_time < self._split_hour:
            duration_day = end_time
        else:
            duration_day = self._split_hour
            duration_night = end_time - self._split_hour

        return duration_day, duration_night

    def add_version_info(self, data: pd.DataFrame):
        data['project_cd'] = self._project_cd
        data['create_user_cd'] = 'SYSTEM'
        data[self.col_fp_version_id] = self.fp_version
        data[self.col_fp_version_seq] = self.fp_seq

        return data

    def fill_unused_capa_info(self, data):
        add_df = pd.DataFrame()

        for res, res_df in data.groupby(by=self._res.res):
            for date in self._schedule_range:
                for time_type in ['D', 'N']:
                    if len(res_df[(res_df[self._post.date] == dt.datetime.strftime(date, '%Y%m%d'))
                                  & (res_df[self._post.time_idx] == time_type)]) == 0:
                        series = self._res_day_avail_time[res][date.day_of_week][time_type]
