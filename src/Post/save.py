import common.util as util
import common.config as config

import os
import numpy as np
import pandas as pd
import datetime as dt


class Save(object):
    ############################################
    # Columns configuration
    ############################################
    # Demand
    col_date = 'yymmdd'
    col_plant = config.col_plant
    col_time_idx_type = 'time_index_type'

    # Resource
    col_res = config.col_res
    col_res_grp = config.col_res_grp
    col_res_capa = config.col_res_capa
    col_duration = config.col_duration

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
        self.data = data

        # Class instance attribute
        self.io = io
        self.query = query

        self.plant = plant
        self.fp_seq = fp_seq
        self.fp_name = fp_name
        self.fp_version = fp_version
        self.project_cd = config.project_cd

        self.sec_of_half_day = 43200
        self.split_hour = dt.timedelta(hours=12)

        # Constraint instance attribute
        self.res_avail_time = res_avail_time
        self.res_to_res_grp = res_to_res_grp
        self.res_grp_mst = res_grp_mst

    def to_csv(self, path, name: str) -> None:
        save_dir = os.path.join(path, 'opt', 'csv', self.fp_version)
        util.make_dir(path=save_dir)

        # Save the optimization result
        self.data.to_csv(os.path.join(save_dir, name + '_' + self.fp_name + '.csv'), index=False, encoding='cp949')

    def res_status(self):
        # Get resource timeline
        timeline = self.get_res_timeline()

        # Preprocess resource status dataset
        res_status = self.prep_res_status(data=timeline)

        # Divide demand / job change result
        res_final = self.divide_dmd_jc_data(data=res_status)

        # Set resource capacity
        res_final = self.set_res_capacity(data=res_final)

        # Split resource capacity to day and night
        res_final = self.split_res_capa_day_night(data=res_final)

        # Set the resource unavailable time
        res_final = self.set_res_unavail_time(data=res_final)

        # Resource available time
        res_final['res_avail_val'] = res_final['res_capa_val'] - res_final['res_use_capa_val'] - res_final['res_jc_val']

        # Add information
        res_final = self.add_model_info(data=res_final)

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self.plant}
        self.io.delete_from_db(sql=self.query.del_res_status_result(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=res_final, tb_name='M4E_O402050')

    def get_res_timeline(self):
        timeline = []
        for res, res_df in self.data.groupby(self.col_res):
            for kind, kind_df in res_df.groupby('kind'):
                for start, end in zip(kind_df['start'], kind_df['end']):
                    timeline.extend(self.calc_res_duration(res=res, kind=kind, start=start, end=end))

        return timeline

    def prep_res_status(self, data):
        res_status = pd.DataFrame(
            data,
            columns=[self.col_res, 'kind', self.col_date, self.col_time_idx_type, self.col_duration]
        )
        res_status[self.col_date] = res_status[self.col_date].dt.strftime('%Y%m%d')
        res_status[self.col_duration] = res_status[self.col_duration] / np.timedelta64(1, 's')

        res_status = res_status.groupby(by=[self.col_res, 'kind', self.col_date, self.col_time_idx_type]) \
            .sum() \
            .reset_index()

        return res_status

    def divide_dmd_jc_data(self, data):
        # Demand
        dmd = data[data['kind'] == 'demand'].copy()
        dmd = dmd.rename(columns={self.col_duration: 'res_use_capa_val'})
        dmd = dmd.drop(columns=['kind'])

        # Job change
        jc = data[data['kind'] == 'job_change'].copy()
        jc = jc.rename(columns={self.col_duration: 'res_jc_val'})
        jc = jc.drop(columns=['kind'])

        result = pd.merge(dmd, jc, how='left', on=[self.col_res, self.col_date, self.col_time_idx_type]).fillna(0)

        return result

    def set_res_capacity(self, data):
        # Resource usage
        data['day'] = [dt.datetime.strptime(day, '%Y%m%d').weekday() for day in data[self.col_date]]
        res_capa = []
        for res, day in zip(data[self.col_res], data['day']):
            res_avail_time = self.res_avail_time[res]
            # res_avail_time = [res_avail_time[0]] + [1440, 1440, 1440] + [res_avail_time[-1]]
            res_capa.append(res_avail_time[day] * 60)

        data[self.col_res_capa] = res_capa

        return data

    def split_res_capa_day_night(self, data):
        res_capa_val = []
        for day, time_idx_type, capacity in zip(
                data['day'], data[self.col_time_idx_type], data[self.col_res_capa]):
            val = self.calc_day_night_res_capacity(day=day, time_idx_type=time_idx_type, capacity=capacity)
            res_capa_val.append(val)
        data['res_capa_val'] = res_capa_val

        return data

    def set_res_unavail_time(self, data):
        res_unavail_val = []
        for day, capacity in zip(data['day'], data['res_capa_val']):
            val = self.calc_res_unavail_time(day=day, capacity=capacity)
            res_unavail_val.append(val)

        data['res_unavail_val'] = res_unavail_val

        return data

    def calc_res_unavail_time(self, day: int, capacity: int):
        val = 0
        if day in [0, 4]:
            val = self.sec_of_half_day - capacity
        elif day in [1, 2, 3]:
            val = 0   # ToDo: temp
            # val = self.sec_of_half_day * 2 - capacity    # ToDo: will be used

        return val

    def add_model_info(self, data):
        res_grp_mst = self.res_grp_mst[[self.col_res, 'res_type_cd']]
        data = pd.merge(data, res_grp_mst, how='left', on=self.col_res).fillna('UNDEFINED')
        data = data.rename(columns={'res_type_cd': 'capa_type_cd'})

        data[self.col_plant] = self.plant
        data[self.col_res_grp] = [self.res_to_res_grp.get(res_cd, 'UNDEFINED')
                                  for res_cd in data[self.col_res]]
        data = self.add_version_info(data=data)

        data = data.drop(columns=[self.col_res_capa, 'day', 'res_capa_val'])

        return data

    def calc_res_duration(self, res, kind, start, end):
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
            if end_time < self.split_hour:
                duration_day = end_time - start_time
            elif start_time > self.split_hour:
                duration_night = end_time - start_time
            else:
                duration_day = self.split_hour - start_time
                duration_night = end_time - self.split_hour

            timeline.append([res, kind, start_day, 'D', duration_day])
            timeline.append([res, kind, start_day, 'N', duration_night])

        elif diff_day == 1:
            prev_duration_day, prev_duration_night = self.calc_timeline_prev(start_time=start_time)
            next_duration_day, next_duration_night = self.calc_timeline_next(end_time=end_time)

            timeline.append([res, kind, start_day, 'D', prev_duration_day])
            timeline.append([res, kind, start_day, 'N', prev_duration_night])
            timeline.append([res, kind, end_day, 'D', next_duration_day])
            timeline.append([res, kind, end_day, 'N', next_duration_night])

        else:
            prev_duration_day, prev_duration_night = self.calc_timeline_prev(start_time=start_time)
            next_duration_day, next_duration_night = self.calc_timeline_next(end_time=end_time)

            timeline.append([res, kind, start_day, 'D', prev_duration_day])
            timeline.append([res, kind, start_day, 'N', prev_duration_night])
            timeline.append([res, kind, end_day, 'D', next_duration_day])
            timeline.append([res, kind, end_day, 'N', next_duration_night])

            for i in range(diff_day - 1):
                timeline.append(
                    [res, kind, start_day + dt.timedelta(days=i + 1), 'D', self.split_hour])
                timeline.append(
                    [res, kind, start_day + dt.timedelta(days=i + 1), 'N', self.split_hour])

        return timeline

    def calc_timeline_prev(self, start_time):
        # Previous day
        duration_day = dt.timedelta(hours=0)
        if start_time < self.split_hour:
            duration_day = self.split_hour - start_time
            duration_night = self.split_hour
        else:
            duration_night = dt.timedelta(hours=24) - start_time

        return duration_day, duration_night

    def calc_timeline_next(self, end_time):
        duration_night = dt.timedelta(hours=0)
        if end_time < self.split_hour:
            duration_day = end_time
        else:
            duration_day = self.split_hour
            duration_night = end_time - self.split_hour

        return duration_day, duration_night

    def calc_day_night_res_capacity(self, day: int, time_idx_type: str, capacity: int):
        val = 0
        if day == 0:
            if time_idx_type == 'D':
                val = max(0, capacity - self.sec_of_half_day)
            elif time_idx_type == 'N':
                val = min(capacity, self.sec_of_half_day)
        elif day in [1, 2, 3]:
            val = self.sec_of_half_day    # ToDo: temp
            # val = capacity    # ToDo: will be used
        else:
            if time_idx_type == 'D':
                val = min(capacity, self.sec_of_half_day)
            elif time_idx_type == 'N':
                val = max(0, capacity - self.sec_of_half_day)

        return val

    def add_version_info(self, data: pd.DataFrame):
        data['project_cd'] = self.project_cd
        data['create_user_cd'] = 'SYSTEM'
        data[self.col_fp_version_id] = self.fp_version
        data[self.col_fp_version_seq] = self.fp_seq

        return data

    def req_prod_qty(self):
        pass

    def gantt(self):
        pass

    def res_qty(self):
        pass
