import common.util as util

import os
import numpy as np
import pandas as pd
import datetime as dt
from ast import literal_eval
import plotly.express as px
import matplotlib.pyplot as plt


class PostProcess(object):
    # Columns instance attribute
    col_fp_version_id = 'fp_vrsn_id'
    col_fp_version_seq = 'fp_vrsn_seq'
    col_date = 'yymmdd'

    default_activity = ['source', 'sink']
    res_schd_cols = ['res_cd', 'start_num', 'end_num', 'capacity']
    act_cols = ['dmd_id', 'item_cd', 'res_grp', 'resource', 'start_num', 'end_num']
    split_symbol = '@'

    # optseq output instance attribute
    act_start_phase = '--- best solution ---'
    act_end_phase = '--- tardy activity ---'
    res_start_phase = '--- resource residuals ---'
    res_end_phase = '--- best activity list ---'
    act_name = 'Act'
    setup_name = 'Setup'

    def __init__(self, io, sql_conf, exec_cfg: dict, fp_version: str, plant_cd: str, plant_start_time,
                 item_mst, prep_data, demand, model_init):
        # Class instance attribute
        self.io = io
        self.sql_conf = sql_conf

        # Execute instance attribute
        self.exec_cfg = exec_cfg
        self.fp_version = fp_version

        self.rm_act_list = model_init['rm_act_list']
        self.act_mode_name_map = model_init['act_mode_name']

        # Plant instance attribute
        self.plant_cd = plant_cd
        self.plant_start_time = plant_start_time

        # Data instance attribute
        self.item_mst = item_mst
        self.demand = demand
        self.res_grp = prep_data['resource']['plant_res_grp'][plant_cd]
        self.res_grp_nm = prep_data['resource']['plant_res_grp_nm'][plant_cd]
        self.res_nm_map = prep_data['resource']['plant_res_nm'][plant_cd]
        self.res_to_res_grp = {}

        # Capacity instance attribute
        self.choose_human_capa = True
        self.used_res_filter_yn = False

        # Timeline instance attribute
        self.split_hour = dt.timedelta(hours=12)
        self.item_avg_duration = {}
        self.item_res_duration = prep_data['resource']['plant_item_res_duration'][plant_cd]
        self.inf_val = 10**7 - 1

        # Path instance attribute
        self.optseq_output_path = os.path.join('..', 'test', 'optseq_output.txt')
        self.save_path = os.path.join('..', '..', 'result')

    def post_process(self):
        # Set resource to resource group
        self.set_res_to_res_grp()

        # Calculate the average duration of producing item
        self.calc_item_res_avg_duration()

        # Save the original result
        self.save_org_result()

        # Best activity
        self.post_process_act()

        # Resource usage timeline
        # self.post_process_res()

    def set_res_to_res_grp(self) -> None:
        res_grp = self.res_grp.copy()
        res_to_res_grp = {}
        for res_grp_cd, res_list in res_grp.items():
            for res_cd, _, _, _ in res_list:
                res_to_res_grp[res_cd] = res_grp_cd

        self.res_to_res_grp = res_to_res_grp

    def calc_item_res_avg_duration(self) -> None:
        duration = self.item_res_duration.copy()
        item_avg_duration = {}
        for item_cd, res_rate in duration.items():
            rate_list = []
            for rate in res_rate.values():
                rate_list.append(rate)
            avg_duration = round(sum(rate_list) / len(rate_list))
            item_avg_duration[item_cd] = avg_duration

        self.item_avg_duration = item_avg_duration

    def post_process_act(self) -> None:
        # Get the best sequence result
        activity = self.get_best_activity()

        # Post processing
        activity_df = self.conv_to_df(data=activity, kind='activity')
        activity_df = self.fill_na(data=activity_df)
        activity_df = self.change_timeline(data=activity_df)
        qty_df = self.calc_timeline_prod_qty(data=activity_df)
        # qty_df = self.add_miss_demand(data=qty_df)

        if self.exec_cfg['save_step_yn']:
            self.save_opt_result(data=activity_df, name='act.csv')
            self.save_opt_result(data=qty_df, name='qty.csv')

        if self.exec_cfg['save_db_yn']:
            self.save_on_db(data=qty_df)

        if self.exec_cfg['save_graph_yn']:
            self.draw_gantt(data=activity_df)

    def add_miss_demand(self, data: pd.DataFrame):
        rm_dmd_list = [act[act.index('[')+1: act.index('@')] for act in self.rm_act_list]
        rm_dmd_df = self.demand[self.demand['dmd_id'].isin(rm_dmd_list)].copy()
        miss_dmd_qty = rm_dmd_df['qty'].sum()
        miss_dmd = {'res_grp_cd': ['MISS'], 'res_grp_nm': ['MISS'], 'res_cd': ['MISS'], 'res_nm': ['MISS'],
                    'item_cd': ['MISS'], 'item_nm': ['MISS'], 'yymmdd': [self.plant_start_time], 'type':['D'],
                    'qty': [miss_dmd_qty]}
        data = data.append(pd.DataFrame(miss_dmd))

        return data

    def calc_timeline_prod_qty(self, data: pd.DataFrame):
        timeline_list = []
        for res_cd, res_df in data.groupby('resource'):
            for item_cd, item_df in res_df.groupby('item_cd'):
                for start, end in zip(item_df['start'], item_df['end']):
                    timeline_list.extend(self.calc_duration(res_cd, item_cd, start, end))

        qty_df = pd.DataFrame(timeline_list, columns=['res_cd', 'item_cd', 'yymmdd', 'type', 'duration'])
        qty_df = self.set_item_res_capa_rate(data=qty_df)
        qty_df['duration'] = qty_df['duration'] / np.timedelta64(1, 's')
        qty_df['qty'] = np.round(qty_df['duration'] / qty_df['capa_rate'], 2)

        #
        qty_df['qty'] = qty_df['qty'].fillna(0)
        qty_df['qty'] = np.nan_to_num(qty_df['qty'].values, posinf=self.inf_val, neginf=self.inf_val)

        # Data processing
        qty_df = qty_df.drop(columns=['duration'])

        qty_df = self.add_name_info(data=qty_df)

        return qty_df

    def add_name_info(self, data: pd.DataFrame):
        # Add item naming
        item_mst = self.item_mst[['item_cd', 'item_nm']].copy()
        data = pd.merge(data, item_mst, how='left', on='item_cd')

        # Add resource naming
        data['res_grp_cd'] = [self.res_to_res_grp.get(res_cd, 'UNDEFINED') for res_cd in data['res_cd']]
        data['res_grp_nm'] = [self.res_grp_nm.get(res_grp_cd, 'UNDEFINED') for res_grp_cd in data['res_grp_cd']]
        data['res_nm'] = [self.res_nm_map.get(res_cd, 'UNDEFINED') for res_cd in data['res_cd']]

        data = data[['res_grp_cd', 'res_grp_nm', 'res_cd', 'res_nm', 'item_cd', 'item_nm', 'yymmdd', 'type', 'qty']]

        return data

    def set_item_res_capa_rate(self, data):
        capa_rate_list = []
        for item_cd, res_cd in zip(data['item_cd'], data['res_cd']):
            capa_rate = self.item_res_duration[item_cd].get(res_cd, self.item_avg_duration[item_cd])
            capa_rate_list.append(capa_rate)

        data['capa_rate'] = capa_rate_list

        return data

    def calc_duration(self, res_cd, item_cd, start, end):
        start_day = dt.datetime.strptime(dt.datetime.strftime(start, '%Y%m%d'), '%Y%m%d')
        start_time = dt.timedelta(hours=start.hour, minutes=start.minute, seconds=start.second)
        end_day = dt.datetime.strptime(dt.datetime.strftime(end, '%Y%m%d'), '%Y%m%d')
        end_time = dt.timedelta(hours=end.hour, minutes=end.minute, seconds=end.second)

        diff_day = (end_day - start_day).days

        timeline = []
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

            timeline.append([res_cd, item_cd, start_day, 'D', duration_day])
            timeline.append([res_cd, item_cd, start_day, 'N', duration_night])

        elif diff_day == 1:
            prev_duration_day, prev_duration_night = self.calc_timeline_prev(start_time=start_time)
            next_duration_day, next_duration_night = self.calc_timeline_next(end_time=end_time)

            timeline.append([res_cd, item_cd, start_day, 'D', prev_duration_day])
            timeline.append([res_cd, item_cd, start_day, 'N', prev_duration_night])
            timeline.append([res_cd, item_cd, end_day, 'D', next_duration_day])
            timeline.append([res_cd, item_cd, end_day, 'N', next_duration_night])

        else:
            prev_duration_day, prev_duration_night = self.calc_timeline_prev(start_time=start_time)
            next_duration_day, next_duration_night = self.calc_timeline_next(end_time=end_time)

            timeline.append([res_cd, item_cd, start_day, 'D', prev_duration_day])
            timeline.append([res_cd, item_cd, start_day, 'N', prev_duration_night])
            timeline.append([res_cd, item_cd, end_day, 'D', next_duration_day])
            timeline.append([res_cd, item_cd, end_day, 'N', next_duration_night])

            for i in range(diff_day-1):
                timeline.append([res_cd, item_cd, start_day + dt.timedelta(days=i+1), 'D', self.split_hour])
                timeline.append([res_cd, item_cd, start_day + dt.timedelta(days=i+1), 'N', self.split_hour])

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

    @staticmethod
    def fill_na(data: pd.DataFrame) -> pd.DataFrame:
        data['resource'] = data['resource'].fillna('UNDEFINED')

        return data

    def conv_res_format(self, data: pd.DataFrame):
        data['resource'] = data['resource'].str.split(self.split_symbol).str[1]

        return data

    def post_process_res(self):
        # Get the resource timeline result
        res_schedule = self.get_res_result()
        res_schd_df = self.conv_to_df(data=res_schedule, kind='resource')

        if self.exec_cfg['save_step_yn']:
            self.save_opt_result(data=res_schd_df, name='resource.csv')

        # if self.draw_graph_yn:
        #     self.draw_human_usage(data=res_schd_df)

    def get_res_result(self) -> list:
        with open(self.optseq_output_path) as f:
            res_schedule = []
            add_phase_yn = False
            for line in f:
                # Check the end line
                if line.strip() == self.res_end_phase:
                    break
                if add_phase_yn and line.strip() != '':
                    res_timeline = line.strip().split(' ')
                    if self.used_res_filter_yn:
                        if len(res_timeline) > 3:
                            time_list = self.get_res_timeline(data=res_timeline)
                            res_schedule.extend(time_list)
                    else:
                        time_list = self.get_res_timeline(data=res_timeline)
                        if time_list is not None:
                            res_schedule.extend(time_list)

                if line.strip() == self.res_start_phase:
                    add_phase_yn = True

        f.close()

        return res_schedule

    def get_res_timeline(self, data):
        # Resource code
        resource = data[0][:-1]
        timeline = data[1:]

        if self.choose_human_capa:
            if ('F' not in resource) and ('M' not in resource):
                return None

        temp, time_list = ([], [])
        for i, row in enumerate(timeline):
            if i % 2 == 0:
                time_from, time_to = literal_eval(row)
                temp.extend([resource, time_from, time_to])
            else:
                temp.append(literal_eval(row))
                time_list.append(temp)
                temp = []

        return time_list

    def get_best_activity(self) -> list:
        with open(self.optseq_output_path) as f:
            add_phase_yn = False
            solve_result = []
            for line in f:
                if line.strip() == self.act_end_phase:
                    break
                if add_phase_yn and line.strip() != '':
                    result = self.prep_act_mode_result(line)
                    if result:
                        solve_result.append(result)
                if line.strip() == self.act_start_phase:
                    add_phase_yn = True

        f.close()

        return solve_result

    def prep_act_mode_result(self, line: str):
        # Split result
        activity, mode, schedule = line.strip().split(',')
        if activity not in self.default_activity:
            # optseq exception (if activity has 1 mode then result:'---')
            if mode == '---':
                mode = self.act_mode_name_map[activity]

            # preprocess the activity
            demand_id, item_cd, res_grp = activity.split('@')
            demand_id = demand_id[demand_id.index('[') + 1:]
            res_grp = res_grp[:res_grp.index(']')]

            # preprocess the mode
            mode = mode.split('@')[-1]
            mode = mode[:mode.index(']')]

            # preprocess the schedule
            schedule = schedule.strip().split(' ')
            duration_start = int(schedule[0])
            duration_end = int(schedule[-1])

            result = [demand_id, item_cd, res_grp, mode, duration_start, duration_end]

            return result

        else:
            return None

    def conv_to_df(self, data: list, kind: str):
        df = None
        if kind == 'activity':
            df = pd.DataFrame(data, columns=self.act_cols)
            df = df.sort_values(by=['dmd_id'])

        elif kind == 'resource':
            df = pd.DataFrame(data, columns=self.res_schd_cols)
            df = df.sort_values(by=['res_cd'])

        return df

    def change_timeline(self, data: pd.DataFrame):

        data['start'] = data['start_num'].apply(
            lambda x: self.plant_start_time + dt.timedelta(seconds=x)
        )
        data['end'] = data['end_num'].apply(
            lambda x: self.plant_start_time + dt.timedelta(seconds=x)
        )

        data = data.drop(columns=['start_num', 'end_num'])

        return data

    def draw_gantt(self, data: pd.DataFrame) -> None:
        # data = self.change_timeline(data=data)

        self.draw_activity(data=data, kind='demand')
        self.draw_activity(data=data, kind='resource')

    def draw_activity(self, data, kind: str) -> None:
        if kind == 'demand':
            # By demand
            fig = px.timeline(data, x_start='start', x_end='end', y='dmd_id', color='resource',
                              color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_yaxes(autorange="reversed")

            # Save the graph
            self.save_fig(fig=fig, name='gantt_act_dmd.html')

        elif kind == 'resource':
            # By resource
            fig = px.timeline(data, x_start='start', x_end='end', y='resource', color='dmd_id',
                              color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_yaxes(autorange="reversed")

            # Save the graph
            self.save_fig(fig=fig, name='gantt_act_res.html')

        plt.close()

    def draw_human_usage(self, data) -> None:
        data = data[data['capacity'] == 0].copy()

        fig = px.timeline(data, x_start='start', x_end='end', y='res_cd', color='res_cd')
        fig.update_yaxes(autorange="reversed")

        # Save the graph
        self.save_fig(fig=fig, name='gantt_resource.html')
        plt.close()

    def save_org_result(self) -> None:
        save_dir = os.path.join(self.save_path, 'opt', 'org', self.fp_version)
        util.make_dir(path=save_dir)

        result = open(os.path.join(save_dir, self.fp_version + '_result.txt'), 'w')
        with open(self.optseq_output_path, 'r') as f:
            for line in f:
                result.write(line)

        f.close()
        result.close()

    def save_opt_result(self, data: pd.DataFrame, name: str) -> None:
        save_dir = os.path.join(self.save_path, 'opt', 'csv', self.fp_version)
        util.make_dir(path=save_dir)

        data.to_csv(os.path.join(save_dir, self.fp_version + '_' + name), index=False, encoding='cp949')

    def save_on_db(self, data: pd.DataFrame):
        fp_seq_df = self.io.get_df_from_db(
            sql=self.sql_conf.sql_fp_seq_list(**{self.col_fp_version_id: self.fp_version})
        )
        if len(fp_seq_df) == 0:
            last_seq = '001'
        else:
            last_seq = fp_seq_df[self.col_fp_version_seq].max()
            last_seq = str(int(last_seq) + 1).zfill(3)

        data[self.col_fp_version_id] = self.fp_version
        data[self.col_fp_version_seq] = last_seq
        data[self.col_date] = data[self.col_date].dt.strftime('%Y%m%d')

        # Save the result on DB
        self.io.insert_to_db(df=data, tb_name='M4E_O402122')

    def save_fig(self, fig, name: str) -> None:
        save_dir = os.path.join(self.save_path, 'gantt', self.fp_version)
        util.make_dir(path=save_dir)

        fig.write_html(os.path.join(save_dir, name))
        # fig.write_image(os.path.join(save_dir, name))
