import common.util as util
import common.config as config
from constraint.constraint import Human

import os
import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
import matplotlib.pyplot as plt


class Process(object):
    ############################################
    # optseq output configuration
    ############################################
    act_name = 'Act'
    setup_name = 'Setup'
    default_activity = ['source', 'sink']
    act_start_phase = '--- best solution ---'
    act_end_phase = '--- tardy activity ---'
    res_start_phase = '--- resource residuals ---'
    res_end_phase = '--- best activity list ---'

    ############################################
    # Dictionary key configuration
    ############################################
    key_dmd = config.key_dmd
    key_res = config.key_res
    key_item = config.key_item
    key_cstr = config.key_cstr
    key_res_grp = config.key_res_grp
    key_res_grp_nm = config.key_res_grp_nm
    key_item_res_duration = config.key_res_duration

    # Constrain
    key_human_res = config.key_human_res
    key_res_avail_time = config.key_res_avail_time

    ############################################
    # Columns configuration
    ############################################
    col_date = 'yymmdd'
    col_fp_key = 'fp_key'
    col_time_idx_type = 'time_index_type'
    col_fp_version_id = 'fp_vrsn_id'
    col_fp_version_seq = 'fp_vrsn_seq'

    # Demand
    col_dmd = config.col_dmd
    col_sku = config.col_sku
    col_prod_qty = 'prod_qty'
    col_plant = config.col_plant
    col_sku_nm = config.col_sku_nm
    col_duration = config.col_duration

    # Resource
    col_res = config.col_res
    col_res_nm = config.col_res_nm
    col_res_capa = config.col_res_capa
    col_res_grp = config.col_res_grp
    col_res_grp_nm = config.col_res_grp_nm

    col_start_time = 'starttime'
    col_end_time = 'endtime'

    split_symbol = '@'
    res_schd_cols = [col_res, col_start_time, col_end_time, col_res_capa]
    act_cols = [col_dmd, col_sku, col_res_grp, col_res, col_start_time, col_end_time, 'kind']

    def __init__(
            self,
            io,
            query,
            exec_cfg: dict,
            cstr_cfg: dict,
            fp_version: str,
            fp_seq: str,
            plant: str,
            plant_start_time: dt.datetime,
            data: dict,
            prep_data: dict,
            model_init: dict
         ):
        # Class instance attribute
        self.io = io
        self.query = query
        self.exec_cfg = exec_cfg
        self.cstr_cfg = cstr_cfg

        # Model version instance attribute
        self.fp_seq = fp_seq
        self.fp_version = fp_version
        self.fp_name = fp_version + '_' + fp_seq + '_' + plant
        self.project_cd = 'ENT001'
        self.act_mode_name_map = model_init['act_mode_name']

        # Plant instance attribute
        self.plant = plant
        self.sec_of_half_day = 43200
        self.plant_start_time = plant_start_time

        # Data instance attribute
        self.cstr = data[self.key_cstr]
        self.demand = data[self.key_dmd]
        self.item_mst = data[self.key_res][self.key_item]

        # Resource instance attribute
        self.res_to_res_grp = {}
        self.item_avg_duration = {}
        self.res_grp_mst = data[self.key_res][self.key_res_grp]
        self.res_grp = prep_data[self.key_res][self.key_res_grp][plant]
        self.res_grp_nm = prep_data[self.key_res][self.key_res_grp_nm][plant]
        self.res_nm_map = prep_data[self.key_res]['plant_res_nm'][plant]
        self.item_res_duration = prep_data[self.key_res][self.key_item_res_duration][plant]

        # Constraint instance attribute
        self.inf_val = 10 ** 7 - 1
        self.split_hour = dt.timedelta(hours=12)
        self.human_res = prep_data[self.key_cstr][self.key_human_res]
        self.res_avail_time = prep_data[self.key_cstr][self.key_res_avail_time][plant]

        # Path instance attribute
        self.save_path = os.path.join('..', '..', 'result')
        self.optseq_output_path = os.path.join('..', 'operation', 'optseq_output.txt')

    def run(self):
        # Set resource to resource group
        self.set_res_to_res_grp()

        # Calculate the average duration of producing item
        self.calc_item_res_avg_duration()

        # Best activity
        self.post_process()

    def set_res_to_res_grp(self) -> None:
        res_grp = self.res_grp.copy()
        res_to_res_grp = {}
        for res_grp_cd, res_list in res_grp.items():
            for res_cd in res_list:
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

    def post_process(self) -> None:
        # Get the best sequence result
        activity = self.get_best_activity()

        # Post process
        result = self.conv_to_df(data=activity, kind='activity')
        result = self.fill_na(data=result)

        if self.cstr_cfg['apply_human_capacity']:
            human_cstr = Human(
                plant=self.plant,
                plant_start_time=self.plant_start_time,
                item=self.item_mst,
                demand=self.demand,
                cstr=self.cstr
            )
            print(f"Apply human capacity: Plant {self.plant}")
            result = human_cstr.apply(data=result)

        result = self.change_timeline(data=result)

        prod_qty = self.calc_timeline_prod_qty(data=result)

        if self.exec_cfg['save_step_yn']:
            # Csv.save(data=result, name='act')

            # Save the activity
            self.save_opt_result(data=result, name='act')

            # Save the result
            self.save_opt_result(data=prod_qty, name='qty')

        if self.exec_cfg['save_db_yn']:
            # seq = self.make_fp_seq()

            self.save_res_status_on_db(data=result, seq=self.fp_seq)

            # Demand (req quantity vs prod quantity)
            self.save_req_prod_qty_on_db(data=result, seq=self.fp_seq)

            # Resource
            self.save_gantt_on_db(data=result, seq=self.fp_seq)

            # Production quantity on day & night
            self.save_res_day_night_qty_on_db(data=prod_qty, seq=self.fp_seq)

        if self.exec_cfg['save_graph_yn']:
            self.draw_gantt(data=result)

    def get_best_activity(self) -> list:
        result_dir = os.path.join(self.save_path, 'opt', 'org', self.fp_version)
        result_path = os.path.join(result_dir, 'result_' + self.fp_name + '.txt')

        with open(result_path) as f:
            add_phase_yn = False
            solve_result = []
            for line in f:
                if line.strip() == self.act_end_phase:
                    break
                if add_phase_yn and line.strip() != '':
                    result = self.prep_act_mode_result(line)
                    if result:
                        solve_result.extend(result)
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
            dmd_id, item_cd, res_grp, res, kind = (None, None, None, None, None)

            act_kind = activity[:activity.index('[')]
            if act_kind == 'Act':
                dmd_id, item_cd, res_grp = activity.split('@')
                dmd_id = dmd_id[dmd_id.index('[') + 1:]
                res_grp = res_grp[:res_grp.index(']')]

                # preprocess the mode
                res = mode.split('@')[-1]
                res = res[:res.index(']')]
                kind = 'demand'

            elif act_kind == 'Setup':
                _, item_cd, res_grp, res = activity.split('@')
                res = res[:res.index(']')]

                from_dmd, to_dmd, mode = mode.split('|')
                from_dmd = from_dmd.split('@')
                if len(from_dmd) > 1:
                    from_dmd = from_dmd[0][from_dmd[0].index('[') + 1:]
                    to_dmd = to_dmd.split('@')[0]
                    dmd_id = from_dmd + '_' + to_dmd
                    kind = 'job_change'
                else:
                    return None

            # preprocess the schedule
            schedule = schedule.strip().split(' ')[1:-1]

            result = []
            for from_to_time in schedule:
                duration_start, duration_end = list(map(int, from_to_time.split('--')))
                result.append([dmd_id, item_cd, res_grp, res, duration_start, duration_end, kind])

            return result

    def calc_timeline_prod_qty(self, data: pd.DataFrame):
        timeline_list = []
        for res_cd, res_df in data.groupby(self.col_res):
            for item_cd, item_df in res_df.groupby(self.col_sku):
                for start, end in zip(item_df['start'], item_df['end']):
                    timeline_list.extend(self.calc_duration(res_cd, item_cd, start, end))

        qty_df = pd.DataFrame(
            timeline_list,
            columns=[self.col_res, self.col_sku, self.col_date, self.col_time_idx_type, self.col_duration]
        )
        qty_df = self.set_item_res_capa_rate(data=qty_df)
        qty_df[self.col_duration] = qty_df[self.col_duration] / np.timedelta64(1, 's')
        qty_df[self.col_prod_qty] = np.round(qty_df[self.col_duration] / qty_df['capa_rate'], 2)

        #
        qty_df[self.col_prod_qty] = qty_df[self.col_prod_qty].fillna(0)
        qty_df[self.col_prod_qty] = np.nan_to_num(
            qty_df[self.col_prod_qty].values,
            posinf=self.inf_val,
            neginf=self.inf_val
        )

        # Data processing
        qty_df = qty_df.drop(columns=[self.col_duration])
        qty_df = qty_df.groupby(by=[self.col_res, self.col_sku, self.col_date, self.col_time_idx_type, 'capa_rate']) \
            .sum() \
            .reset_index()

        qty_df = self.add_name_info(data=qty_df)

        return qty_df

    def add_name_info(self, data: pd.DataFrame):
        # Add item naming
        item_mst = self.item_mst[[self.col_sku, self.col_sku_nm]].drop_duplicates().copy()
        data = pd.merge(data, item_mst, how='left', on=self.col_sku)

        # Add resource naming
        data[self.col_res_grp] = [self.res_to_res_grp.get(res_cd, 'UNDEFINED') for res_cd in data[self.col_res]]
        data[self.col_res_grp_nm] = [self.res_grp_nm.get(res_grp_cd, 'UNDEFINED')
                                     for res_grp_cd in data[self.col_res_grp]]
        data[self.col_res_nm] = [self.res_nm_map.get(res_cd, 'UNDEFINED') for res_cd in data[self.col_res]]

        data = data[[self.col_res_grp, self.col_res_grp_nm, self.col_res, self.col_res_nm, self.col_sku,
                     self.col_sku_nm, self.col_date, self.col_time_idx_type, self.col_prod_qty]]

        return data

    def set_item_res_capa_rate(self, data):
        capa_rate_list = []
        for item_cd, res_cd in zip(data[self.col_sku], data[self.col_res]):
            capa_rate = self.item_res_duration[item_cd].get(res_cd, self.item_avg_duration[item_cd])
            capa_rate_list.append(capa_rate)

        data['capa_rate'] = capa_rate_list

        return data

    def calc_duration(self, res_cd, item_cd, start, end):
        start_day = dt.datetime.strptime(dt.datetime.strftime(start, '%Y%m%d'), '%Y%m%d')
        start_time = dt.timedelta(hours=start.hour, minutes=start.minute, seconds=start.second)
        end_day = dt.datetime.strptime(dt.datetime.strftime(end, '%Y%m%d'), '%Y%m%d')
        end_time = dt.timedelta(hours=end.hour, minutes=end.minute, seconds=end.second)

        #
        if end_time == dt.timedelta(seconds=0):
            end_day = end_day - dt.timedelta(days=1)
            end_time = dt.timedelta(hours=24)

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

            for i in range(diff_day - 1):
                timeline.append([res_cd, item_cd, start_day + dt.timedelta(days=i + 1), 'D', self.split_hour])
                timeline.append([res_cd, item_cd, start_day + dt.timedelta(days=i + 1), 'N', self.split_hour])

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

    def fill_na(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.col_res] = data[self.col_res].fillna('UNDEFINED')

        return data

    def conv_to_df(self, data: list, kind: str):
        df = None
        if kind == 'activity':
            df = pd.DataFrame(data, columns=self.act_cols)
            df = df.sort_values(by=[self.col_dmd])

        elif kind == 'resource':
            df = pd.DataFrame(data, columns=self.res_schd_cols)
            df = df.sort_values(by=[self.col_res])

        return df

    def change_timeline(self, data: pd.DataFrame):
        data['start'] = data[self.col_start_time].apply(
            lambda x: self.plant_start_time + dt.timedelta(seconds=x)
        )
        data['end'] = data[self.col_end_time].apply(
            lambda x: self.plant_start_time + dt.timedelta(seconds=x)
        )

        data = data.drop(columns=[self.col_start_time, self.col_end_time])

        return data

    def draw_gantt(self, data: pd.DataFrame) -> None:
        # data = self.change_timeline(data=data)
        self.draw_activity(data=data[data['kind'] == 'demand'], y=self.col_dmd, color=self.col_res, name='act_demand')
        self.draw_activity(data=data, y=self.col_res, color=self.col_dmd, name='act_resource')

    def draw_activity(self, data: pd.DataFrame, y: str, color: str, name: str):
        fig = px.timeline(data, x_start='start', x_end='end', y=y, color=color,
                          color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(autorange="reversed")

        # Save the graph
        self.save_fig(fig=fig, name=name)

        plt.close()

    #####################
    # Save
    #####################
    def save_org_result(self) -> None:
        save_dir = os.path.join(self.save_path, 'opt', 'org', self.fp_version)
        util.make_dir(path=save_dir)

        result = open(os.path.join(save_dir, 'result_' + self.fp_name + '.txt'), 'w')
        with open(self.optseq_output_path, 'r') as f:
            for line in f:
                result.write(line)

        f.close()
        result.close()

    def save_opt_result(self, data: pd.DataFrame, name: str) -> None:
        save_dir = os.path.join(self.save_path, 'opt', 'csv', self.fp_version)
        util.make_dir(path=save_dir)

        # Save the optimization result
        data.to_csv(os.path.join(save_dir, name + '_' + self.fp_name + '.csv'), index=False, encoding='cp949')

    def save_fig(self, fig, name: str) -> None:
        save_dir = os.path.join(self.save_path, 'gantt', self.fp_version)
        util.make_dir(path=save_dir)

        fig.write_html(os.path.join(save_dir, name + '_' + self.fp_seq + '_' + self.plant +'.html'))
        # fig.write_image(os.path.join(save_dir, name))

    def make_fp_seq(self) -> str:
        fp_seq_df = self.io.load_from_db(
            sql=self.query.sql_fp_seq_list(**{self.col_fp_version_id: self.fp_version})
        )
        if len(fp_seq_df) == 0:
            seq = '1'
        else:
            seq = fp_seq_df[self.col_fp_version_seq].max()
            # eq = str(int(seq) + 1).zfill(3)
            seq = str(int(seq) + 1)

        return seq

    def save_req_prod_qty_on_db(self, data: pd.DataFrame, seq: str):
        data = data.rename(columns={
            self.col_dmd: self.col_fp_key, 'start': self.col_start_time, 'end': self.col_end_time}
        )

        data = data[data['kind'] == 'demand'].copy()
        data = self.calc_prod_qty(data=data)

        prod_qty = data.groupby(by=[self.col_fp_key]).sum()['prod_qty'].reset_index()
        end_date = data.groupby(by=[self.col_fp_key]).max()[self.col_end_time].reset_index()
        merged = pd.merge(prod_qty, end_date, on=self.col_fp_key)
        merged[self.col_end_time] = merged[self.col_end_time].dt.strftime('%Y%m%d')

        # Add demand information
        demand = self.demand[[self.col_dmd, self.col_sku, 'qty']].rename(
            columns={self.col_dmd: self.col_fp_key, 'qty': 'req_fp_qty'}
        )
        merged = pd.merge(demand, merged, how='left', on=self.col_fp_key)
        merged[self.col_prod_qty] = merged[self.col_prod_qty].fillna(0)
        merged[self.col_end_time] = merged[self.col_end_time].fillna('99991231')

        merged = self.add_version_info(data=merged, seq=seq)

        merged = merged.rename(columns={
            self.col_sku: 'eng_item_cd', self.col_end_time: self.col_date, self.col_prod_qty: 'fp_qty'})

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq}
        self.io.delete_from_db(sql=self.query.del_dmd_result(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=merged, tb_name='M4E_O402010')

    def save_res_day_night_qty_on_db(self, data: pd.DataFrame, seq: str) -> None:
        data = self.add_version_info(data=data, seq=seq)
        data[self.col_plant] = self.plant
        data[self.col_date] = data[self.col_date].dt.strftime('%Y%m%d')

        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self.plant}
        self.io.delete_from_db(sql=self.query.del_res_day_night_qty(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=data, tb_name='M4E_O402130')

    def calc_prod_qty(self, data: pd.DataFrame):
        data = self.set_item_res_capa_rate(data=data)
        data['res_used_capa_val'] = data[self.col_end_time] - data[self.col_start_time]
        data['res_used_capa_val'] = data['res_used_capa_val'] / np.timedelta64(1, 's')
        data[self.col_prod_qty] = np.round(data['res_used_capa_val'] / data['capa_rate'], 2)

        return data

    def save_gantt_on_db(self, data: pd.DataFrame, seq: str) -> None:
        data = data.rename(columns={
            self.col_dmd: self.col_fp_key, 'start': self.col_start_time, 'end': self.col_end_time
        })

        data = data[data['kind'] == 'demand'].copy()
        data = self.calc_prod_qty(data=data)

        # Add information
        data = self.add_version_info(data=data, seq=seq)

        # add resource & item names
        data[self.col_res_grp_nm] = data[self.col_res_grp].apply(lambda x: self.res_grp_nm.get(x, 'UNDEFINED'))
        data[self.col_res_nm] = data[self.col_res].apply(lambda x: self.res_nm_map.get(x, 'UNDEFINED'))
        item_mst = self.item_mst[[self.col_sku, self.col_sku_nm]].drop_duplicates().copy()
        data = pd.merge(data, item_mst, how='left', on=self.col_sku)

        # Change data type (datetime -> string)
        data[self.col_start_time] = data[self.col_start_time].dt.strftime('%y%m%d%H%m%s')
        data[self.col_start_time] = data[self.col_start_time] \
            .str.replace('-', '') \
            .str.replace(':', '') \
            .str.replace(' ', '')
        data[self.col_end_time] = data[self.col_end_time].dt.strftime('%y%m%d%h%M%s')
        data[self.col_end_time] = data[self.col_end_time] \
            .str.replace('-', '') \
            .str.replace(':', '') \
            .str.replace(' ', '')

        data[self.col_plant] = self.plant

        data = data.drop(columns=['kind', 'capa_rate'])

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self.plant}
        self.io.delete_from_db(sql=self.query.del_gantt_result(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=data, tb_name='M4E_O402140')

    def save_res_status_on_db(self, data: pd.DataFrame, seq: str):
        #
        timeline_list = []
        for res, res_df in data.groupby(self.col_res):
            for kind, kind_df in res_df.groupby('kind'):
                for start, end in zip(kind_df['start'], kind_df['end']):
                    timeline_list.extend(self.calc_res_duration(res=res, kind=kind, start=start, end=end))

        res_status = pd.DataFrame(
            timeline_list,
            columns=[self.col_res, 'kind', self.col_date, self.col_time_idx_type, self.col_duration]
        )
        res_status[self.col_date] = res_status[self.col_date].dt.strftime('%Y%m%d')
        res_status[self.col_duration] = res_status[self.col_duration] / np.timedelta64(1, 's')

        res_status = res_status.groupby(by=[self.col_res, 'kind', self.col_date, self.col_time_idx_type]) \
            .sum() \
            .reset_index()

        res_status_dmd = res_status[res_status['kind'] == 'demand'].copy()
        res_status_dmd = res_status_dmd.rename(columns={self.col_duration: 'res_use_capa_val'})
        res_status_dmd = res_status_dmd.drop(columns=['kind'])
        res_status_jc = res_status[res_status['kind'] == 'job_change'].copy()
        res_status_jc = res_status_jc.rename(columns={self.col_duration: 'res_jc_val'})
        res_status_jc = res_status_jc.drop(columns=['kind'])

        res_final = pd.merge(res_status_dmd, res_status_jc, how='left',
                             on=[self.col_res, self.col_date, self.col_time_idx_type]).fillna(0)

        # Resource usage
        res_final['day'] = [dt.datetime.strptime(day, '%Y%m%d').weekday() for day in res_final[self.col_date]]
        res_capa = []
        for res, day in zip(res_final[self.col_res], res_final['day']):
            res_avail_time = self.res_avail_time[res]
            # Todo : Temp
            res_avail_time = [res_avail_time[0]] + [1440, 1440, 1440] + [res_avail_time[-1]]
            res_capa.append(res_avail_time[day] * 60)
        res_final[self.col_res_capa] = res_capa

        # Resource capacity time
        res_capa_val = []
        for day, time_idx_type, capacity in zip(
                res_final['day'], res_final[self.col_time_idx_type], res_final[self.col_res_capa]):
            val = self.calc_day_night_res_capacity(day=day, time_idx_type=time_idx_type, capacity=capacity)
            res_capa_val.append(val)
        res_final['res_capa_val'] = res_capa_val

        # Resource unavailable time
        res_unavail_val = []
        for day, time_idx_type, capacity in zip(
                res_final['day'], res_final[self.col_time_idx_type], res_final['res_capa_val']):
            val = self.calc_res_unavail_time(day=day, time_idx_type=time_idx_type, capacity=capacity)
            res_unavail_val.append(val)
        res_final['res_unavail_val'] = res_unavail_val

        # Resource available time
        res_final['res_avail_val'] = res_final['res_capa_val'] - res_final['res_use_capa_val'] - res_final['res_jc_val']

        res_grp_mst = self.res_grp_mst[[self.col_res, 'res_type_cd']]
        res_final = pd.merge(res_final, res_grp_mst, how='left', on=self.col_res).fillna('UNDEFINED')
        res_final = res_final.rename(columns={'res_type_cd': 'capa_type_cd'})

        # Add information
        res_final[self.col_plant] = self.plant
        res_final[self.col_res_grp] = [self.res_to_res_grp.get(res_cd, 'UNDEFINED')
                                       for res_cd in res_final[self.col_res]]
        res_final = self.add_version_info(data=res_final, seq=seq)

        res_final = res_final.drop(columns=[self.col_res_capa, 'day', 'res_capa_val'])

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self.plant}
        self.io.delete_from_db(sql=self.query.del_res_status_result(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=res_final, tb_name='M4E_O402050')

    def calc_res_unavail_time(self, day: int, time_idx_type: str, capacity: int):
        val = 0
        if day in [0, 4]:
            val = self.sec_of_half_day - capacity
        elif day in [1, 2, 3]:
            val = 0   # ToDo: temp
            # val = self.sec_of_half_day * 2 - capacity    # ToDo: will be used

        return val

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

    def add_version_info(self, data: pd.DataFrame, seq: str):
        data['project_cd'] = self.project_cd
        data['create_user_cd'] = 'SYSTEM'
        data[self.col_fp_version_id] = self.fp_version
        data[self.col_fp_version_seq] = seq

        return data
