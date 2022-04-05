import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PostProcess(object):
    default_demand = ['source', 'sink']
    gantt_cols = ['dmd_id', 'item_cd', 'res_grp', 'resource', 'start_num', 'end_num']
    time_unit = 'sec'
    gantt_time_unit = 'h'    #  h / m / s
    gantt_time_map = {'h': 3600, 'm': 60}

    def __init__(self, start_time):
        self.output_path = os.path.join('optseq_output.txt')
        self.start_phase = '--- best solution ---'
        self.end_phase = '--- tardy activity ---'
        self.start_time = start_time

    def get_best_result(self) -> list:
        with open(self.output_path) as f:
            result_list = []
            add_yn = False
            for line in f:
                if line.strip() == self.end_phase:
                    break
                if add_yn and line.strip() != '':
                    demand, resource, schedule = line.strip().split(',')
                    if demand not in self.default_demand:
                        demand_id, item_cd, res_grp = demand.split('@')
                        schedule = schedule.strip().split(' ')
                        duration_start = int(schedule[0])
                        duration_end = int(schedule[-1])
                        result_list.append(
                            [demand, item_cd, res_grp[:-1], resource, duration_start, duration_end]
                            # [demand_id[4:], item_cd, res_grp[:-1], resource, duration_start, duration_end]
                        )
                if line.strip() == self.start_phase:
                    add_yn = True

        return result_list

    def conv_to_df(self, data: list):
        df = pd.DataFrame(data, columns=self.gantt_cols)
        df = df.sort_values(by=['dmd_id'])

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
        data['duration'] = data['end'] - data['start']

        # Change the time frequency
        data['duration'] = np.ceil(data['duration'] / np.timedelta64(1, self.gantt_time_unit))
        data['duration'] = data['duration'].apply(lambda x: dt.timedelta(hours=x))

        return data

    def draw_grantt(self, data):
        fig, ax = plt.subplots(1, figsize=(16, 6))
        ax.barh(data['dmd_id'], data['duration'], left=data['start'])

        # # Ticks
        xticks = pd.date_range(self.start_time, end=data['end'].max(), freq='H')
        xticks_labels = xticks.strftime('%m%d %H:%M')
        ax.set_xticks(xticks[::6])
        ax.set_xticklabels(xticks_labels[::6])

        plt.savefig('gantt.png')
        plt.close()


# Test process
pp = PostProcess(
    start_time=dt.datetime.today()
        .replace(microsecond=0)
        .replace(second=0)
        .replace(minute=0)
        .replace(hour=0)
)
result_list = pp.get_best_result()
result_df = pp.conv_to_df(data=result_list)
# result_df = pp.change_time_freq(data=result_df)
result_df = pp.change_timeline(data=result_df)
pp.draw_grantt(data=result_df)


