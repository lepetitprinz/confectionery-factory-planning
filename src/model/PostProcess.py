import os
import numpy as np
import pandas as pd
import datetime as dt
from ast import literal_eval
import plotly.express as px
import matplotlib.pyplot as plt


class PostProcess(object):
    default_demand = ['source', 'sink']
    res_schd_cols = ['res_cd', 'start_num', 'end_num', 'capacity']
    act_cols = ['dmd_id', 'item_cd', 'res_grp', 'resource', 'start_num', 'end_num']
    time_unit = 'sec'
    gantt_time_unit = 'h'    # h / m / s
    gantt_time_map = {'h': 3600, 'm': 60}

    def __init__(self):
        # Execute instance attribute
        self.save_optseq_result_yn = False
        self.draw_graph_yn = True

        # path instance attribute
        self.optseq_output_path = os.path.join('..', 'test', 'optseq_output.txt')
        self.save_path = os.path.join('..', '..', 'result')

        # time instance attribute
        self.start_time = self.set_start_time()
        self.choose_human_capa = True
        self.used_res_filter_yn = False

        # optseq output instance attribute
        self.act_start_phase = '--- best solution ---'
        self.act_end_phase = '--- tardy activity ---'
        self.res_start_phase = '--- resource residuals ---'
        self.res_end_phase = '--- best activity list ---'

    def post_process(self):
        # Best activity
        self.post_process_act()

        # Resource usage timeline
        self.post_process_res()

    def post_process_act(self):
        # Get the best sequence result
        activity = self.get_best_activity()
        activity_df = self.conv_to_df(data=activity, kind='activity')

        if self.save_optseq_result_yn:
            self.save_opt_result(data=activity_df)

        if self.draw_graph_yn:
            self.draw_gantt(data=activity_df)

    def post_process_res(self):
        # Get the resource timeline result
        res_schedule = self.get_res_result()
        res_schd_df = self.conv_to_df(data=res_schedule, kind='resource')

        if self.save_optseq_result_yn:
            self.save_opt_result(data=res_schd_df)

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
            solve_result = []
            add_phase_yn = False
            for line in f:
                if line.strip() == self.act_end_phase:
                    break
                if add_phase_yn and line.strip() != '':
                    demand, resource, schedule = line.strip().split(',')
                    if demand not in self.default_demand:
                        demand_id, item_cd, res_grp = demand.split('@')
                        schedule = schedule.strip().split(' ')
                        duration_start = int(schedule[0])
                        duration_end = int(schedule[-1])
                        solve_result.append(
                            # [demand, item_cd, res_grp[:-1], resource, duration_start, duration_end]
                            [demand_id[4:], item_cd, res_grp[:-1], resource[5:-1], duration_start, duration_end]
                        )
                if line.strip() == self.act_start_phase:
                    add_phase_yn = True

        f.close()

        return solve_result

    def conv_to_df(self, data: list, kind: str):
        df = None
        if kind == 'activity':
            df = pd.DataFrame(data, columns=self.act_cols)
            df = df.sort_values(by=['dmd_id'])

        elif kind == 'resource':
            df = pd.DataFrame(data, columns=self.res_schd_cols)
            df = df.sort_values(by=['res_cd'])

        return df

    def change_time_freq(self, data):
        data['start_num'] = np.round(data['start_num'] / self.gantt_time_map[self.gantt_time_unit], 2)
        data['end_num'] = np.round(data['end_num'] / self.gantt_time_map[self.gantt_time_unit], 2)

        return data

    def change_timeline(self, data: pd.DataFrame):

        data['start'] = data['start_num'].apply(
            lambda x: self.start_time + dt.timedelta(seconds=x)
        )
        data['end'] = data['end_num'].apply(
            lambda x: self.start_time + dt.timedelta(seconds=x)
        )
        # data['duration'] = data['end'] - data['start']

        # # Change the time frequency
        # data['duration'] = np.ceil(data['duration'] / np.timedelta64(1, self.gantt_time_unit))
        # data['duration'] = data['duration'].apply(lambda x: dt.timedelta(hours=x))

        data = data.drop(columns=['start_num', 'end_num'])

        return data

    def save_opt_result(self, data: pd.DataFrame):
        data.to_csv(os.path.join(self.save_path, 'opt_result.csv'), index=False)

    def draw_gantt(self, data: pd.DataFrame):
        data = self.change_timeline(data=data)

        self.draw_activity(data=data, kind='demand')
        self.draw_activity(data=data, kind='resource')

    def draw_activity(self, data, kind: str):
        if kind == 'demand':
            # By demand
            fig_dmd = px.timeline(data, x_start='start', x_end='end', y='dmd_id', color='resource',
                                  color_discrete_sequence=px.colors.qualitative.Set3)
            fig_dmd.update_yaxes(autorange="reversed")
            fig_dmd.write_html(os.path.join(self.save_path, 'gantt', 'gantt_act_dmd.html'))    # Save the graph
            # fig.write_image(os.path.join(self.save_path, 'gantt', 'gantt_activity.png'))

        elif kind == 'resource':
            # By resource
            fig_res = px.timeline(data, x_start='start', x_end='end', y='resource', color='dmd_id',
                                  color_discrete_sequence=px.colors.qualitative.Set3)
            fig_res.update_yaxes(autorange="reversed")
            fig_res.write_html(os.path.join(self.save_path, 'gantt', 'gantt_act_res.html'))  # Save the graph
            # fig.write_image(os.path.join(self.save_path, 'gantt', 'gantt_activity.png')

    def draw_human_usage(self, data):
        data = data[data['capacity'] == 0].copy()

        fig = px.timeline(data, x_start='start', x_end='end', y='res_cd', color='res_cd')
        fig.update_yaxes(autorange="reversed")

        # Save the graph
        fig.write_html(os.path.join(self.save_path, 'gantt', 'gantt_resource.html'))
        # fig.write_image(os.path.join(self.save_path, 'gantt', 'gantt_activity.png'))
        plt.close()

    def set_start_time(self):
        start_time = dt.datetime.today() \
            .replace(microsecond=0) \
            .replace(second=0) \
            .replace(minute=0) \
            .replace(hour=0)

        return start_time

        # def draw_activity_bak(self, data):
        #     fig, ax = plt.subplots(1, figsize=(16, 6))
        #     ax.barh(data['dmd_id'], data['duration'], left=data['start'])
        #
        #     # Ticks
        #     xticks = pd.date_range(self.start_time, end=data['end'].max(), freq='H')
        #     xticks_labels = xticks.strftime('%m/%d %H:%M')
        #     ax.set_xticks(xticks[::6])
        #     ax.set_xticklabels(xticks_labels[::6])
        #
        #     fig.write_image(os.path.join(self.save_path, 'gantt', 'gantt_activity.png'))
        #     plt.close()
