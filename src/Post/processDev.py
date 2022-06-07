import common.config as config
import common.util as util
from common.name import Key, Demand, Item, Resource, Constraint, Post
from constraint.capacityDev import Human
from constraint.simultaneous import Necessary

from Post.save import Save
from Post.plot import Gantt

import os
import numpy as np
import pandas as pd
import datetime as dt


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
    split_symbol = '@'

    ############################################
    # Columns configuration
    ############################################
    col_date = 'yymmdd'
    col_start_time = 'starttime'
    col_end_time = 'endtime'
    col_fp_version_id = 'fp_vrsn_id'
    col_fp_version_seq = 'fp_vrsn_seq'

    def __init__(
            self,
            io,
            query,
            cfg: dict,
            version,
            plant: str,
            plant_start_time: dt.datetime,
            data: dict,
            prep_data: dict,
            model_init: dict,
            calendar: pd.DataFrame
         ):
        # Class instance attribute
        self.io = io
        self.query = query
        self.cfg = cfg

        # Model version instance attribute
        self.fp_seq = version.fp_seq
        self.fp_version = version.fp_version
        self.fp_name = version.fp_version + '_' + version.fp_seq + '_' + plant
        self.project_cd = config.project_cd
        self.act_mode_name_map = model_init['act_mode_name']

        # Name instance attribute
        self.key = Key()
        self.dmd = Demand()
        self.res = Resource()
        self.item = Item()
        self.cstr = Constraint()
        self.post = Post()

        # Columns usage instance attribute
        self.res_schd_cols = [self.res.res, self.dmd.start_time, self.dmd.end_time, self.res.res_capa]
        self.act_cols = [self.dmd.dmd, self.item.sku, self.res.res_grp, self.res.res,
                         self.dmd.start_time, self.dmd.end_time, 'kind']
        self.prod_qty_cols = [
            self.res.res_grp, self.res.res_grp_nm, self.res.res, self.res.res_nm, self.item.sku, self.item.sku_nm,
            self.col_date, self.post.time_idx, self.dmd.prod_qty
        ]
        self.prod_dmd_qty_cols = [
            self.dmd.dmd, self.res.res_grp, self.res.res_grp_nm, self.res.res, self.res.res_nm, self.item.sku,
            self.item.sku_nm, self.col_date, self.post.time_idx, self.dmd.prod_qty
        ]

        # Plant instance attribute
        self.plant = plant
        self.sec_of_half_day = 43200
        self.plant_start_time = plant_start_time

        # Data instance attribute
        self.data = data
        self.calendar = calendar
        self.demand = data[self.key.dmd]
        self.res_mst = data[self.key.res][self.key.res_grp]
        self.item_mst = data[self.key.res][self.key.item]
        self.cstr_mst = data[self.key.cstr]

        # Resource instance attribute
        self.res_to_res_grp = {}
        self.item_avg_duration = {}
        self.res_grp = prep_data[self.key.res][self.key.res_grp][plant]
        self.res_grp_nm = prep_data[self.key.res][self.key.res_grp_nm][plant]
        self.res_nm_map = prep_data[self.key.res][self.key.res_nm][plant]
        self.res_duration = prep_data[self.key.res][self.key.res_duration][plant]

        # Constraint instance attribute
        self.inf_val = 10 ** 7 - 1
        self.split_hour = dt.timedelta(hours=12)
        self.res_avail_time = prep_data[self.key.cstr][self.key.res_avail_time][plant]
        if self.cfg['cstr']['apply_sim_prod_cstr']:
            self.sim_prod_cstr = prep_data[self.key.cstr][self.key.sim_prod_cstr]['necessary'].get(plant, None)

        # Path instance attribute
        self.save_path = os.path.join('..', '..', 'result')
        self.optseq_output_path = os.path.join('..', 'operation', 'optseq_output.txt')

        self.log = []

    def run(self):
        # Set resource to resource group
        self.set_res_to_res_grp()

        # Calculate the average duration of producing item
        self.calc_item_res_avg_duration()

        result = self.post_process_opt_result()

        # Apply the constraint: Human capacity
        if self.cfg['cstr']['apply_human_capacity']:
            result, log, capa_profile, capa_profile_dtl = self.apply_human_capa_const(data=result)
            # if self.cfg['exec']['save_db_yn']:
            if len(capa_profile) > 0:
                self.save_capa_profile(data=capa_profile)
            if len(capa_profile_dtl) > 0:
                self.save_capa_profile_dtl(data=capa_profile_dtl)

            # log information
            self.log.extend(log)

        if self.cfg['cstr']['apply_sim_prod_cstr']:
            if self.sim_prod_cstr is not None:
                result, log = self.apply_sim_prod_cstr(data=result)
                self.log.extend(log)

        util.save_log(
            log=self.log,
            path=os.path.join(self.save_path, 'constraint'),
            version=self.fp_version,
            name=self.fp_name
        )

        # Best activity
        # self.save(result=result)

    def save(self, result) -> None:
        result = self.conv_num_to_datetime(data=result)
        prod_qty = self.calc_timeline_prod_qty(data=result)
        prod_dmd_qty = self.calc_timeline_dmd_prod_qty(data=result)

        if self.cfg['exec']['save_step_yn'] or self.cfg['exec']['save_db_yn']:
            save = Save(
                data=result,
                io=self.io,
                query=self.query,
                plant=self.plant,
                fp_seq=self.fp_seq,
                fp_name=self.fp_name,
                fp_version=self.fp_version,
                res_avail_time=self.res_avail_time,
                res_grp_mst=self.res_mst,
                res_to_res_grp=self.res_to_res_grp
            )

            if self.cfg['exec']['save_step_yn']:
                # Save the activity
                save.to_csv(path=self.save_path, name='act')

            if self.cfg['exec']['save_db_yn']:
                # Resource status
                save.res_status()

                # Demand (req quantity vs prod quantity)
                self.save_req_prod_qty_on_db(data=result, seq=self.fp_seq)

                # Resource
                self.save_gantt_on_db(data=result, seq=self.fp_seq)

                # Production quantity on day & night
                self.save_res_day_night_qty_on_db(data=prod_qty, seq=self.fp_seq)
                self.save_res_day_night_dmd_qty_in_db(data=prod_dmd_qty, seq=self.fp_seq)

        if self.cfg['exec']['save_graph_yn']:
            gantt = Gantt(
                fp_version=self.fp_version,
                fp_seq=self.fp_seq,
                plant=self.plant,
                path=self.save_path
            )
            # Draw demand
            gantt.draw(
                data=result[result['kind'] == 'demand'],
                y=self.dmd.dmd,
                color=self.res.res,
                name='act_demand'
            )

            gantt.draw(
                data=result,
                y=self.res.res,
                color=self.dmd.dmd,
                name='act_resource'
            )

    def set_res_to_res_grp(self) -> None:
        res_grp = self.res_grp.copy()
        res_to_res_grp = {}
        for res_grp_cd, res_list in res_grp.items():
            for res_cd in res_list:
                res_to_res_grp[res_cd] = res_grp_cd

        self.res_to_res_grp = res_to_res_grp

    def calc_item_res_avg_duration(self) -> None:
        duration = self.res_duration.copy()
        item_avg_duration = {}
        for item_cd, res_rate in duration.items():
            rate_list = []
            for rate in res_rate.values():
                rate_list.append(rate)
            avg_duration = round(sum(rate_list) / len(rate_list))
            item_avg_duration[item_cd] = avg_duration

        self.item_avg_duration = item_avg_duration

    def post_process_opt_result(self):
        # Get the best sequence result
        activity = self.get_best_activity()

        result = self.conv_to_df(data=activity, kind='activity')

        # Fill nan values
        result = self.fill_na(data=result)

        # Correct the job change error
        result = self.correct_job_change_error(data=result)

        return result

    def apply_human_capa_const(self, data):
        human_cstr = Human(
            plant=self.plant,
            plant_start_time=self.plant_start_time,
            item=self.item_mst,
            cstr=self.cstr_mst,
            demand=self.demand,
            calendar=self.calendar,
            res_to_res_grp=self.res_to_res_grp,
        )
        # print(f"Apply human capacity: Plant {self.plant}")
        result, log, capa_profile, capa_profile_dtl = human_cstr.apply(data=data)

        return result, log, capa_profile, capa_profile_dtl

    def apply_sim_prod_cstr(self, data):
        sim_prod_cstr = Necessary(
            plant=self.plant,
            plant_start_time=self.plant_start_time,
            demand=self.demand,
            org_data=self.data,
            sim_prod_cstr=self.sim_prod_cstr,
        )
        result, log = sim_prod_cstr.apply(data=data)

        return result, log

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

    def correct_job_change_error(self, data) -> pd.DataFrame:
        jc = data[data['kind'] == 'job_change'].copy()
        jc_dmd_list = [[idx, res] + dmd for idx, res, dmd in zip(
            jc.index,
            jc[self.res.res],
            jc[self.dmd.dmd].str.split(self.split_symbol)
        )]

        filter_idx_list = []
        for idx, jc_res, from_dmd, to_dmd in jc_dmd_list:
            from_res = data[data[self.dmd.dmd] == from_dmd][self.res.res].unique()
            to_res = data[data[self.dmd.dmd] == to_dmd][self.res.res].unique()
            if from_res != to_res:
                filter_idx_list.append(idx)
            elif (jc_res != from_res) or (jc_res != to_res):
                filter_idx_list.append(idx)

        data = data.drop(labels=filter_idx_list)

        return data

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
                if len(activity.split(self.split_symbol)) == 3:
                    dmd_id, item_cd, res_grp = activity.split(self.split_symbol)
                    dmd_id = dmd_id[dmd_id.index('[') + 1:]
                    res_grp = res_grp[:res_grp.index(']')]

                    # preprocess the mode
                    res = mode.split(self.split_symbol)[-1]
                    res = res[:res.index(']')]

                elif len(activity.split(self.split_symbol)) == 2:
                    dmd_id, item_cd = activity.split(self.split_symbol)
                    dmd_id = dmd_id[dmd_id.index('[') + 1:]
                    item_cd = item_cd[:item_cd.index(']')]
                    res = mode.split(self.split_symbol)[-1]
                    res = res[:res.index(']')]
                    res_grp = self.res_to_res_grp[res]

                kind = 'demand'

            elif act_kind == 'Setup':
                if len(activity.split(self.split_symbol)) == 4:
                    _, item_cd, res_grp, res = activity.split(self.split_symbol)
                    res = res[:res.index(']')]

                else:
                    _, item_cd, res = activity.split(self.split_symbol)
                    res = res[:res.index(']')]
                    res_grp = self.res_to_res_grp[res]

                from_dmd, to_dmd, mode = mode.split('|')
                from_dmd = from_dmd.split(self.split_symbol)
                if len(from_dmd) > 1:
                    from_dmd = from_dmd[0][from_dmd[0].index('[') + 1:]
                    to_dmd = to_dmd.split(self.split_symbol)[0]
                    dmd_id = from_dmd + self.split_symbol + to_dmd
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

    def calc_timeline_dmd_prod_qty(self, data: pd.DataFrame):
        timeline_list = []
        for dmd_id, dmd_df in data.groupby(self.dmd.dmd):
            for res_cd, res_df in dmd_df.groupby(self.res.res):
                for item_cd, item_df in res_df.groupby(self.item.sku):
                    for start, end in zip(item_df['start'], item_df['end']):
                        duration = self.calc_duration(res_cd, item_cd, start, end)
                        temp = [[dmd_id] + dur for dur in duration]
                        timeline_list.extend(temp)

        qty_df = pd.DataFrame(
            timeline_list,
            columns=[self.dmd.dmd, self.res.res, self.item.sku, self.col_date, self.post.time_idx,
                     self.dmd.duration]
        )
        qty_df = self.set_item_res_capa_rate(data=qty_df)
        qty_df[self.dmd.duration] = qty_df[self.dmd.duration] / np.timedelta64(1, 's')
        qty_df[self.dmd.prod_qty] = np.round(qty_df[self.dmd.duration] / qty_df['capa_rate'], 2)

        qty_df[self.dmd.prod_qty] = qty_df[self.dmd.prod_qty].fillna(0)
        qty_df[self.dmd.prod_qty] = np.nan_to_num(
            qty_df[self.dmd.prod_qty].values,
            posinf=self.inf_val,
            neginf=self.inf_val
        )

        # Data processing
        qty_df = qty_df.drop(columns=[self.dmd.duration])
        qty_df = qty_df.groupby(by=[self.dmd.dmd, self.res.res, self.item.sku,
                                    self.col_date, self.post.time_idx, 'capa_rate']) \
            .sum() \
            .reset_index()

        qty_df = self.add_name_info(data=qty_df, cols=self.prod_dmd_qty_cols)

        return qty_df

    def calc_timeline_prod_qty(self, data: pd.DataFrame):
        timeline_list = []
        for res_cd, res_df in data.groupby(self.res.res):
            for item_cd, item_df in res_df.groupby(self.item.sku):
                for start, end in zip(item_df['start'], item_df['end']):
                    timeline_list.extend(self.calc_duration(res_cd, item_cd, start, end))

        qty_df = pd.DataFrame(
            timeline_list,
            columns=[self.res.res, self.item.sku, self.col_date, self.post.time_idx, self.dmd.duration]
        )
        qty_df = self.set_item_res_capa_rate(data=qty_df)
        qty_df[self.dmd.duration] = qty_df[self.dmd.duration] / np.timedelta64(1, 's')
        qty_df[self.dmd.prod_qty] = np.round(qty_df[self.dmd.duration] / qty_df['capa_rate'], 2)

        #
        qty_df[self.dmd.prod_qty] = qty_df[self.dmd.prod_qty].fillna(0)
        qty_df[self.dmd.prod_qty] = np.nan_to_num(
            qty_df[self.dmd.prod_qty].values,
            posinf=self.inf_val,
            neginf=self.inf_val
        )

        # Data processing
        qty_df = qty_df.drop(columns=[self.dmd.duration])
        qty_df = qty_df.groupby(by=[self.res.res, self.item.sku, self.col_date, self.post.time_idx, 'capa_rate']) \
            .sum() \
            .reset_index()

        qty_df = self.add_name_info(data=qty_df,  cols=self.prod_qty_cols)

        return qty_df

    def add_name_info(self, data: pd.DataFrame, cols: list):
        # Add item naming
        item_mst = self.item_mst[[self.item.sku, self.item.sku_nm]].drop_duplicates().copy()
        data = pd.merge(data, item_mst, how='left', on=self.item.sku)

        # Add resource naming
        data[self.res.res_grp] = [self.res_to_res_grp.get(res_cd, 'UNDEFINED') for res_cd in data[self.res.res]]
        data[self.res.res_grp_nm] = [self.res_grp_nm.get(res_grp_cd, 'UNDEFINED')
                                     for res_grp_cd in data[self.res.res_grp]]
        data[self.res.res_nm] = [self.res_nm_map.get(res_cd, 'UNDEFINED') for res_cd in data[self.res.res]]

        data = data[cols]

        return data

    def set_item_res_capa_rate(self, data):
        capa_rate_list = []
        for item_cd, res_cd in zip(data[self.item.sku], data[self.res.res]):
            capa_rate = self.res_duration[item_cd].get(res_cd, self.item_avg_duration[item_cd])
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
        data[self.res.res] = data[self.res.res].fillna('UNDEFINED')

        return data

    def conv_to_df(self, data: list, kind: str):
        df = None
        if kind == 'activity':
            df = pd.DataFrame(data, columns=self.act_cols)
            df = df.sort_values(by=[self.dmd.dmd])

        elif kind == 'resource':
            df = pd.DataFrame(data, columns=self.res_schd_cols)
            df = df.sort_values(by=[self.res.res])

        return df

    def conv_num_to_datetime(self, data: pd.DataFrame):
        data['start'] = data[self.dmd.start_time].apply(
            lambda x: self.plant_start_time + dt.timedelta(seconds=x)
        )
        data['end'] = data[self.dmd.end_time].apply(
            lambda x: self.plant_start_time + dt.timedelta(seconds=x)
        )

        data = data.drop(columns=[self.dmd.start_time, self.dmd.end_time])

        return data

    #####################
    # Save
    #####################
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
            self.dmd.dmd: self.post.fp_key, 'start': self.dmd.start_time, 'end': self.dmd.end_time}
        )

        data = data[data['kind'] == 'demand'].copy()
        data = self.calc_prod_qty(data=data)

        prod_qty = data.groupby(by=[self.post.fp_key]).sum()[self.dmd.prod_qty].reset_index()
        end_date = data.groupby(by=[self.post.fp_key]).max()[self.dmd.end_time].reset_index()
        merged = pd.merge(prod_qty, end_date, on=self.post.fp_key)
        merged[self.dmd.end_time] = merged[self.dmd.end_time].dt.strftime('%Y%m%d')

        # Add demand information
        demand = self.demand[[self.dmd.dmd, self.item.sku, 'qty']].rename(
            columns={self.dmd.dmd: self.post.fp_key, 'qty': 'req_fp_qty'}
        )
        merged = pd.merge(demand, merged, how='inner', on=self.post.fp_key)
        merged[self.dmd.prod_qty] = merged[self.dmd.prod_qty].fillna(0)
        merged[self.dmd.end_time] = merged[self.dmd.end_time].fillna('99991231')

        merged = self.add_version_info(data=merged, seq=seq)

        merged = merged.rename(columns={
            self.item.sku: self.post.eng_item, self.dmd.end_time: self.col_date, self.dmd.prod_qty: 'fp_qty'})

        # Delete previous result
        kwargs = {
            self.post.fp_version: self.fp_version,
            self.post.fp_seq: self.fp_seq,
            self.post.fp_key: tuple(merged[self.post.fp_key].tolist())
        }
        self.io.delete_from_db(sql=self.query.del_dmd_result(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=merged, tb_name='M4E_O402010')

    def save_res_day_night_qty_on_db(self, data: pd.DataFrame, seq: str) -> None:
        data = self.add_version_info(data=data, seq=seq)
        data[self.res.plant] = self.plant
        data[self.col_date] = data[self.col_date].dt.strftime('%Y%m%d')

        # Add item type code
        data = pd.merge(
            data,
            self.item_mst[[self.item.sku, self.post.item_type]].drop_duplicates(),
            how='left',
            on=self.item.sku
        )
        data[self.post.item_type] = data[self.post.item_type].fillna('-')
        data[self.item.sku_nm] = data[self.item.sku_nm].fillna('-')

        kwargs = {self.post.fp_version: self.fp_version, self.post.fp_seq: self.fp_seq, 'plant_cd': self.plant}
        self.io.delete_from_db(sql=self.query.del_res_day_night_qty(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=data, tb_name='M4E_O402130')

    def save_res_day_night_dmd_qty_in_db(self, data: pd.DataFrame, seq: str):
        data = self.add_version_info(data=data, seq=seq)
        data[self.res.plant] = self.plant
        data[self.col_date] = data[self.col_date].dt.strftime('%Y%m%d')

        # Add item type code
        data = pd.merge(
            data,
            self.item_mst[[self.item.sku, self.post.item_type]].drop_duplicates(),
            how='left',
            on=self.item.sku
        )
        data[self.post.item_type] = data[self.post.item_type].fillna('-')
        data[self.item.sku_nm] = data[self.item.sku_nm].fillna('-')
        data = data.rename(columns={self.dmd.dmd: self.post.fp_key})

        kwargs = {self.post.fp_version: self.fp_version, self.post.fp_seq: self.fp_seq, 'plant_cd': self.plant}
        self.io.delete_from_db(sql=self.query.del_res_day_night_dmd_qty(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=data, tb_name='M4E_O402131')

    def calc_prod_qty(self, data: pd.DataFrame):
        data = self.set_item_res_capa_rate(data=data)
        data[self.post.res_use_capa] = data[self.dmd.end_time] - data[self.dmd.start_time]
        data[self.post.res_use_capa] = data[self.post.res_use_capa] / np.timedelta64(1, 's')
        data[self.dmd.prod_qty] = np.round(data[self.post.res_use_capa] / data['capa_rate'], 2)

        return data

    def save_gantt_on_db(self, data: pd.DataFrame, seq: str) -> None:
        data = data.rename(columns={
            self.dmd.dmd: self.post.fp_key, 'start': self.dmd.start_time, 'end': self.dmd.end_time
        })

        data = data[data['kind'] == 'demand'].copy()
        data = self.calc_prod_qty(data=data)

        # Add information
        data = self.add_version_info(data=data, seq=seq)

        # add resource & item names
        data[self.res.res_grp_nm] = data[self.res.res_grp].apply(lambda x: self.res_grp_nm.get(x, 'UNDEFINED'))
        data[self.res.res_nm] = data[self.res.res].apply(lambda x: self.res_nm_map.get(x, 'UNDEFINED'))
        item_mst = self.item_mst[[self.item.sku, self.item.sku_nm]].drop_duplicates().copy()
        data = pd.merge(data, item_mst, how='left', on=self.item.sku)

        # Change data type (datetime -> string)
        data[self.dmd.start_time] = data[self.dmd.start_time].dt.strftime('%y%m%d%H%m%s')
        data[self.dmd.start_time] = data[self.dmd.start_time] \
            .str.replace('-', '') \
            .str.replace(':', '') \
            .str.replace(' ', '')
        data[self.dmd.end_time] = data[self.dmd.end_time].dt.strftime('%y%m%d%h%M%s')
        data[self.dmd.end_time] = data[self.dmd.end_time] \
            .str.replace('-', '') \
            .str.replace(':', '') \
            .str.replace(' ', '')

        data[self.res.plant] = self.plant

        data = data.drop(columns=['kind', 'capa_rate'])

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self.plant}
        self.io.delete_from_db(sql=self.query.del_gantt_result(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=data, tb_name='M4E_O402140')

    def save_res_status_on_db(self, data: pd.DataFrame, seq: str):
        #
        timeline_list = []
        for res, res_df in data.groupby(self.res.res):
            for kind, kind_df in res_df.groupby('kind'):
                for start, end in zip(kind_df['start'], kind_df['end']):
                    timeline_list.extend(self.calc_res_duration(res=res, kind=kind, start=start, end=end))

        res_status = pd.DataFrame(
            timeline_list,
            columns=[self.res.res, 'kind', self.col_date, self.post.time_idx, self.dmd.duration]
        )
        res_status[self.col_date] = res_status[self.col_date].dt.strftime('%Y%m%d')
        res_status[self.dmd.duration] = res_status[self.dmd.duration] / np.timedelta64(1, 's')

        res_status = res_status.groupby(by=[self.res.res, 'kind', self.col_date, self.post.time_idx]) \
            .sum() \
            .reset_index()

        res_status_dmd = res_status[res_status['kind'] == 'demand'].copy()
        res_status_dmd = res_status_dmd.rename(columns={self.dmd.duration: self.post.res_use_capa})
        res_status_dmd = res_status_dmd.drop(columns=['kind'])
        res_status_jc = res_status[res_status['kind'] == 'job_change'].copy()
        res_status_jc = res_status_jc.rename(columns={self.dmd.duration: 'res_jc_val'})
        res_status_jc = res_status_jc.drop(columns=['kind'])

        res_final = pd.merge(res_status_dmd, res_status_jc, how='left',
                             on=[self.res.res, self.col_date, self.post.time_idx]).fillna(0)

        # Resource usage
        res_final['day'] = [dt.datetime.strptime(day, '%Y%m%d').weekday() for day in res_final[self.col_date]]
        res_capa = []
        for res, day in zip(res_final[self.res.res], res_final['day']):
            res_avail_time = self.res_avail_time[res]
            res_capa.append(res_avail_time[day] * 60)
        res_final[self.res.res_capa] = res_capa

        # Resource capacity time
        res_capa_val = []
        for day, time_idx_type, capacity in zip(
                res_final['day'], res_final[self.post.time_idx], res_final[self.res.res_capa]):
            val = self.calc_day_night_res_capacity(day=day, time_idx_type=time_idx_type, capacity=capacity)
            res_capa_val.append(val)
        res_final['res_capa_val'] = res_capa_val

        # Resource unavailable time
        res_unavail_val = []
        for day, time_idx_type, capacity in zip(
                res_final['day'], res_final[self.post.time_idx], res_final['res_capa_val']):
            val = self.calc_res_unavail_time(day=day, capacity=capacity)
            res_unavail_val.append(val)
        res_final['res_unavail_val'] = res_unavail_val

        # Resource available time
        res_final['res_avail_val'] = res_final['res_capa_val'] - res_final['res_use_capa_val'] - res_final['res_jc_val']

        res_grp_mst = self.res_mst[[self.res.res, 'res_type_cd']]
        res_final = pd.merge(res_final, res_grp_mst, how='left', on=self.res.res).fillna('UNDEFINED')
        res_final = res_final.rename(columns={'res_type_cd': 'capa_type_cd'})

        # Add information
        res_final[self.res.plant] = self.plant
        res_final[self.res.res_grp] = [self.res_to_res_grp.get(res_cd, 'UNDEFINED')
                                       for res_cd in res_final[self.res.res]]
        res_final = self.add_version_info(data=res_final, seq=seq)

        res_final = res_final.drop(columns=[self.res.res_capa, 'day'])

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self.plant}
        self.io.delete_from_db(sql=self.query.del_res_status_result(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=res_final, tb_name='M4E_O402050')

    def calc_res_unavail_time(self, day: int, capacity: int):
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

    def save_capa_profile(self, data: pd.DataFrame) -> None:
        data[self.res.plant] = self.plant
        data = self.add_version_info(data=data, seq=self.fp_seq)

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self.plant}
        self.io.delete_from_db(sql=self.query.del_human_capa_profile(**kwargs))

        # Save result
        self.io.insert_to_db(df=data, tb_name='M4E_O402150')

    def save_capa_profile_dtl(self, data: pd.DataFrame) -> None:
        data[self.res.res_grp_nm] = [self.res_grp_nm.get(res_grp, '-') for res_grp in data[self.res.res_grp]]
        data[self.res.res_nm] = [self.res_nm_map.get(res, '-') for res in data[self.res.res]]

        data['capa_rate'] = [
            self.res_duration[item][res] for item, res in zip(data[self.item.sku], data[self.res.res])
        ]
        data[self.dmd.prod_qty] = np.round((data[self.post.res_use_capa] / data['capa_rate']).values, 2)
        data[self.post.res_use_capa] = np.round((data[self.post.res_use_capa] / 60).values, 2)

        data = self.add_version_info(data=data, seq=self.fp_seq)

        data = data[['project_cd', self.col_fp_version_id, self.col_fp_version_seq, self.res.plant,
                     self.cstr.floor, self.res.res_grp, self.res.res_grp_nm, self.res.res, self.res.res_nm,
                     self.item.sku, self.item.sku_nm, self.item.item_type, self.post.from_time, self.post.to_time,
                     self.post.from_yymmdd, self.post.to_yymmdd, self.dmd.prod_qty, self.post.tot_m_capa,
                     self.post.tot_w_capa, self.post.use_m_capa, self.post.use_w_capa, self.post.res_use_capa]]

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self.plant}
        self.io.delete_from_db(sql=self.query.del_human_capa_profile_dtl(**kwargs))

        # Save result
        self.io.insert_to_db(df=data, tb_name='M4E_O402151')
