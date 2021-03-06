import common.config as config
import common.util as util
from common.name import Key, Demand, Item, Resource, Constraint, Post
from constraint.human import Human
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
        self.cfg = cfg
        self.query = query
        self.log = []

        # Model version instance attribute
        self.fp_seq = version.fp_seq
        self.fp_version = version.fp_version
        self.fp_name = version.fp_version + '_' + version.fp_seq + '_' + plant
        self._project_cd = config.project_cd
        self._act_mode_name_map = model_init['act_mode_name']

        # Name instance attribute
        self._key = Key()
        self._dmd = Demand()
        self._res = Resource()
        self._item = Item()
        self._cstr = Constraint()
        self._post = Post()

        # Plant instance attribute
        self._plant = plant
        self._sec_of_half_day = config.sec_of_day // 2
        self._plant_start_time = plant_start_time

        # Master Data instance attribute
        self._data = data
        self._calendar = calendar
        self._demand = data[self._key.dmd]
        self._res_mst = data[self._key.res][self._key.res_grp]
        self._item_mst = data[self._key.item]
        self._cstr_mst = data[self._key.cstr]

        # Data hash map instance attribute
        self._res_to_res_grp = {}
        self._item_avg_duration = {}
        self._res_nm_map = prep_data[self._key.res][self._key.res_nm][plant]
        self._res_grp_map = prep_data[self._key.res][self._key.res_grp][plant]
        self._res_grp_nm_map = prep_data[self._key.res][self._key.res_grp_nm][plant]
        self._res_duration = prep_data[self._key.res][self._key.res_duration][plant]

        # Constraint instance attribute
        self._inf_val = config.inf_val
        self._split_hour = dt.timedelta(hours=12)
        self._res_avail_time = prep_data[self._key.cstr][self._key.res_avail_time][plant]

        if self.cfg['cstr']['apply_sim_prod_cstr']:
            self._sim_prod_cstr = prep_data[self._key.cstr][self._key.sim_prod_cstr]['necessary'].get(plant, None)

        # Path instance attribute
        self.save_path = os.path.join('..', '..', 'result')
        self.optseq_output_path = os.path.join('..', 'operation', 'optseq_output.txt')

        # Columns usage instance attribute
        self._res_schd_cols = [self._res.res, self._dmd.start_time, self._dmd.end_time, self._res.res_capa]
        self._act_cols = [
            self._dmd.dmd, self._item.sku, self._res.res_grp, self._res.res, self._dmd.start_time, self._dmd.end_time,
            'kind'
        ]
        self._prod_qty_cols = [
            self._res.res_grp, self._res.res_grp_nm, self._res.res, self._res.res_nm, self._item.sku, self._item.sku_nm,
            self._post.date, self._post.time_idx, self._dmd.prod_qty
        ]
        self._prod_dmd_qty_cols = [
            self._dmd.dmd, self._res.res_grp, self._res.res_grp_nm, self._res.res, self._res.res_nm, self._item.sku,
            self._item.sku_nm, self._post.date, self._post.time_idx, self._dmd.prod_qty
        ]

    def run(self):
        # Set resource to resource group
        self._set_res_to_res_grp()

        # Calculate the average duration of producing item
        self._calc_item_res_avg_duration()

        result = self._post_process_opt_result()

        # Apply constraint: Human capacity
        if self.cfg['cstr']['apply_human_capacity']:
            result, log, capa_profile, capa_profile_dtl = self._apply_human_capa_const(data=result)
            if self.cfg['exec']['save_db_yn']:
                if len(capa_profile) > 0:
                    self._save_capa_profile(data=capa_profile)
                if len(capa_profile_dtl) > 0:
                    self._save_capa_profile_dtl(data=capa_profile_dtl)

            # log information
            self.log.extend(log)

        # Apply constraint: Simultaneous production constraint
        if self.cfg['cstr']['apply_sim_prod_cstr']:
            if self._sim_prod_cstr is not None:
                result, log = self._apply_sim_prod_cstr(data=result)
                self.log.extend(log)

        util.save_log(
            log=self.log,
            path=os.path.join(self.save_path, 'constraint'),
            version=self.fp_version,
            name=self.fp_name
        )

        # Best activity
        self.save(result=result)

    def save(self, result) -> None:
        result = self._conv_num_to_datetime(data=result)
        prod_qty = self._calc_timeline_prod_qty(data=result)
        prod_dmd_qty = self._calc_timeline_dmd_prod_qty(data=result)

        if self.cfg['exec']['save_step_yn'] or self.cfg['exec']['save_db_yn']:
            save = Save(
                data=result,
                io=self.io,
                query=self.query,
                plant=self._plant,
                fp_seq=self.fp_seq,
                fp_name=self.fp_name,
                fp_version=self.fp_version,
                res_grp_mst=self._res_mst,
                res_to_res_grp=self._res_to_res_grp,
                res_avail_time=self._res_avail_time,
            )

            if self.cfg['exec']['save_step_yn']:
                # Save the activity
                save.to_csv(path=self.save_path, name='act')

            if self.cfg['exec']['save_graph_yn']:
                gantt = Gantt(
                    path=self.save_path,
                    plant=self._plant,
                    fp_seq=self.fp_seq,
                    fp_version=self.fp_version,
                )
                # Draw demand
                gantt.draw(
                    data=result[result['kind'] == 'demand'],
                    y=self._dmd.dmd,
                    color=self._res.res,
                    name='act_demand'
                )

                gantt.draw(
                    data=result,
                    y=self._res.res,
                    color=self._dmd.dmd,
                    name='act_resource'
                )

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

    def _set_res_to_res_grp(self) -> None:
        res_grp = self._res_grp_map.copy()
        res_to_res_grp = {}
        for res_grp_cd, res_list in res_grp.items():
            for res_cd in res_list:
                res_to_res_grp[res_cd] = res_grp_cd

        self._res_to_res_grp = res_to_res_grp

    def _calc_item_res_avg_duration(self) -> None:
        duration = self._res_duration.copy()
        item_avg_duration = {}
        for item_cd, res_rate in duration.items():
            rate_list = []
            for rate in res_rate.values():
                rate_list.append(rate)
            avg_duration = round(sum(rate_list) / len(rate_list))
            item_avg_duration[item_cd] = avg_duration

        self._item_avg_duration = item_avg_duration

    def _post_process_opt_result(self):
        # Get the best sequence result
        activity = self.get_best_activity()

        result = self.conv_to_df(data=activity, kind='activity')

        # Fill nan values
        result = self.fill_na(data=result)

        # Correct the job change error
        result = self.correct_job_change_error(data=result)

        return result

    def _apply_human_capa_const(self, data):
        human_cstr = Human(
            plant=self._plant,
            plant_start_time=self._plant_start_time,
            item=self._item_mst,
            cstr=self._cstr_mst,
            demand=self._demand,
            calendar=self._calendar,
            res_to_res_grp=self._res_to_res_grp,
        )
        # print(f"Apply human capacity: Plant {self.plant}")
        result, log, capa_profile, capa_profile_dtl = human_cstr.apply(data=data)

        return result, log, capa_profile, capa_profile_dtl

    def _apply_sim_prod_cstr(self, data):
        sim_prod_cstr = Necessary(
            plant=self._plant,
            plant_start_time=self._plant_start_time,
            demand=self._demand,
            org_data=self._data,
            sim_prod_cstr=self._sim_prod_cstr,
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
            jc[self._res.res],
            jc[self._dmd.dmd].str.split(self.split_symbol)
        )]

        filter_idx_list = []
        for idx, jc_res, from_dmd, to_dmd in jc_dmd_list:
            from_res = data[data[self._dmd.dmd] == from_dmd][self._res.res].unique()
            to_res = data[data[self._dmd.dmd] == to_dmd][self._res.res].unique()
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
                mode = self._act_mode_name_map[activity]

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
                    res_grp = self._res_to_res_grp[res]

                kind = 'demand'

            elif act_kind == 'Setup':
                if len(activity.split(self.split_symbol)) == 4:
                    _, item_cd, res_grp, res = activity.split(self.split_symbol)
                    res = res[:res.index(']')]

                else:
                    _, item_cd, res = activity.split(self.split_symbol)
                    res = res[:res.index(']')]
                    res_grp = self._res_to_res_grp[res]

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

    def _calc_timeline_dmd_prod_qty(self, data: pd.DataFrame):
        timeline_list = []
        for dmd_id, dmd_df in data.groupby(self._dmd.dmd):
            for res_cd, res_df in dmd_df.groupby(self._res.res):
                for item_cd, item_df in res_df.groupby(self._item.sku):
                    for start, end in zip(item_df['start'], item_df['end']):
                        duration = self.calc_duration(res_cd, item_cd, start, end)
                        temp = [[dmd_id] + dur for dur in duration]
                        timeline_list.extend(temp)

        qty_df = pd.DataFrame(
            timeline_list,
            columns=[self._dmd.dmd, self._res.res, self._item.sku, self._post.date, self._post.time_idx,
                     self._dmd.duration]
        )
        qty_df = self.set_item_res_capa_rate(data=qty_df)
        qty_df[self._dmd.duration] = qty_df[self._dmd.duration] / np.timedelta64(1, 's')
        qty_df[self._dmd.prod_qty] = np.round(qty_df[self._dmd.duration] / qty_df['capa_rate'], 2)

        qty_df[self._dmd.prod_qty] = qty_df[self._dmd.prod_qty].fillna(0)
        qty_df[self._dmd.prod_qty] = np.nan_to_num(
            qty_df[self._dmd.prod_qty].values,
            posinf=self._inf_val,
            neginf=self._inf_val
        )

        # Data processing
        qty_df = qty_df.drop(columns=[self._dmd.duration])
        qty_df = qty_df.groupby(by=[self._dmd.dmd, self._res.res, self._item.sku,
                                    self._post.date, self._post.time_idx, 'capa_rate']) \
            .sum() \
            .reset_index()

        qty_df = self.add_name_info(data=qty_df, cols=self._prod_dmd_qty_cols)

        return qty_df

    def _calc_timeline_prod_qty(self, data: pd.DataFrame):
        timeline_list = []
        for res_cd, res_df in data.groupby(self._res.res):
            for item_cd, item_df in res_df.groupby(self._item.sku):
                for start, end in zip(item_df['start'], item_df['end']):
                    timeline_list.extend(self.calc_duration(res_cd, item_cd, start, end))

        qty_df = pd.DataFrame(
            timeline_list,
            columns=[self._res.res, self._item.sku, self._post.date, self._post.time_idx, self._dmd.duration]
        )
        qty_df = self.set_item_res_capa_rate(data=qty_df)
        qty_df[self._dmd.duration] = qty_df[self._dmd.duration] / np.timedelta64(1, 's')
        qty_df[self._dmd.prod_qty] = np.round(qty_df[self._dmd.duration] / qty_df['capa_rate'], 2)

        #
        qty_df[self._dmd.prod_qty] = qty_df[self._dmd.prod_qty].fillna(0)
        qty_df[self._dmd.prod_qty] = np.nan_to_num(
            qty_df[self._dmd.prod_qty].values,
            posinf=self._inf_val,
            neginf=self._inf_val
        )

        # Data processing
        qty_df = qty_df.drop(columns=[self._dmd.duration])
        qty_df = qty_df.groupby(by=[self._res.res, self._item.sku, self._post.date, self._post.time_idx,
                                    'capa_rate']).sum().reset_index()

        qty_df = self.add_name_info(data=qty_df, cols=self._prod_qty_cols)

        return qty_df

    def add_name_info(self, data: pd.DataFrame, cols: list):
        # Add item naming
        item_mst = self._item_mst[[self._item.sku, self._item.sku_nm]].drop_duplicates().copy()
        data = pd.merge(data, item_mst, how='left', on=self._item.sku)

        # Add resource naming
        data[self._res.res_grp] = [self._res_to_res_grp.get(res_cd, 'UNDEFINED') for res_cd in data[self._res.res]]
        data[self._res.res_grp_nm] = [self._res_grp_nm_map.get(res_grp_cd, 'UNDEFINED')
                                      for res_grp_cd in data[self._res.res_grp]]
        data[self._res.res_nm] = [self._res_nm_map.get(res_cd, 'UNDEFINED') for res_cd in data[self._res.res]]

        data = data[cols]

        return data

    def set_item_res_capa_rate(self, data):
        capa_rate_list = []
        for item_cd, res_cd in zip(data[self._item.sku], data[self._res.res]):
            capa_rate = self._res_duration[item_cd].get(res_cd, self._item_avg_duration[item_cd])
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
            if end_time < self._split_hour:
                duration_day = end_time - start_time
            elif start_time > self._split_hour:
                duration_night = end_time - start_time
            else:
                duration_day = self._split_hour - start_time
                duration_night = end_time - self._split_hour

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
                timeline.append([res_cd, item_cd, start_day + dt.timedelta(days=i + 1), 'D', self._split_hour])
                timeline.append([res_cd, item_cd, start_day + dt.timedelta(days=i + 1), 'N', self._split_hour])

        return timeline

    def calc_timeline_prev(self, start_time):
        # Previous day
        duration_day = dt.timedelta(hours=0)
        if start_time < self._split_hour:
            duration_day = self._split_hour - start_time
            duration_night = self._split_hour
        else:
            duration_night = dt.timedelta(hours=24) - start_time

        return duration_day, duration_night

    def calc_timeline_next(self, end_time):
        duration_night = dt.timedelta(hours=0)
        if end_time < self._split_hour:
            duration_day = end_time
        else:
            duration_day = self._split_hour
            duration_night = end_time - self._split_hour

        return duration_day, duration_night

    def fill_na(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self._res.res] = data[self._res.res].fillna('UNDEFINED')

        return data

    def conv_to_df(self, data: list, kind: str):
        df = None
        if kind == 'activity':
            df = pd.DataFrame(data, columns=self._act_cols)
            df = df.sort_values(by=[self._dmd.dmd])

        elif kind == 'resource':
            df = pd.DataFrame(data, columns=self._res_schd_cols)
            df = df.sort_values(by=[self._res.res])

        return df

    def _conv_num_to_datetime(self, data: pd.DataFrame):
        data['start'] = data[self._dmd.start_time].apply(
            lambda x: self._plant_start_time + dt.timedelta(seconds=x)
        )
        data['end'] = data[self._dmd.end_time].apply(
            lambda x: self._plant_start_time + dt.timedelta(seconds=x)
        )

        data = data.drop(columns=[self._dmd.start_time, self._dmd.end_time])

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
            self._dmd.dmd: self._post.fp_key, 'start': self._dmd.start_time, 'end': self._dmd.end_time}
        )

        data = data[data['kind'] == 'demand'].copy()
        data = self.calc_prod_qty(data=data)

        prod_qty = data.groupby(by=[self._post.fp_key]).sum()[self._dmd.prod_qty].reset_index()
        end_date = data.groupby(by=[self._post.fp_key]).max()[self._dmd.end_time].reset_index()
        merged = pd.merge(prod_qty, end_date, on=self._post.fp_key)
        merged[self._dmd.end_time] = merged[self._dmd.end_time].dt.strftime('%Y%m%d')

        # Add demand information
        demand = self._demand[[self._dmd.dmd, self._item.sku, 'qty']].rename(
            columns={self._dmd.dmd: self._post.fp_key, 'qty': 'req_fp_qty'}
        )
        merged = pd.merge(demand, merged, how='inner', on=self._post.fp_key)
        merged[self._dmd.prod_qty] = merged[self._dmd.prod_qty].fillna(0)
        merged[self._dmd.end_time] = merged[self._dmd.end_time].fillna('99991231')

        merged = self.add_version_info(data=merged, seq=seq)

        merged = merged.rename(columns={
            self._item.sku: self._post.eng_item, self._dmd.end_time: self._post.date, self._dmd.prod_qty: 'fp_qty'})

        # Delete previous result
        kwargs = {
            self._post.fp_version: self.fp_version,
            self._post.fp_seq: self.fp_seq,
            self._post.fp_key: tuple(merged[self._post.fp_key].tolist())
        }
        self.io.delete_from_db(sql=self.query.del_dmd_result(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=merged, tb_name='M4E_O402010')

    def save_res_day_night_qty_on_db(self, data: pd.DataFrame, seq: str) -> None:
        data = self.set_item_res_capa_rate(data=data)
        data[self._post.res_use_capa] = np.round(data[self._dmd.prod_qty] * data['capa_rate'], 0)

        data = self.add_version_info(data=data, seq=seq)
        data[self._res.plant] = self._plant
        data[self._post.date] = data[self._post.date].dt.strftime('%Y%m%d')

        # Add item type code
        data = pd.merge(
            data,
            self._item_mst[[self._item.sku, self._post.item_type]].drop_duplicates(),
            how='left',
            on=self._item.sku
        )
        data[self._post.item_type] = data[self._post.item_type].fillna('-')
        data[self._item.sku_nm] = data[self._item.sku_nm].fillna('-')

        kwargs = {self._post.fp_version: self.fp_version, self._post.fp_seq: self.fp_seq, 'plant_cd': self._plant}
        self.io.delete_from_db(sql=self.query.del_res_day_night_qty(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=data, tb_name='M4E_O402130')

    def save_res_day_night_dmd_qty_in_db(self, data: pd.DataFrame, seq: str):
        data = self.add_version_info(data=data, seq=seq)
        data[self._res.plant] = self._plant
        data[self._post.date] = data[self._post.date].dt.strftime('%Y%m%d')

        # Add item type code
        data = pd.merge(
            data,
            self._item_mst[[self._item.sku, self._post.item_type]].drop_duplicates(),
            how='left',
            on=self._item.sku
        )
        data[self._post.item_type] = data[self._post.item_type].fillna('-')
        data[self._item.sku_nm] = data[self._item.sku_nm].fillna('-')
        data = data.rename(columns={self._dmd.dmd: self._post.fp_key})

        kwargs = {self._post.fp_version: self.fp_version, self._post.fp_seq: self.fp_seq, 'plant_cd': self._plant}
        self.io.delete_from_db(sql=self.query.del_res_day_night_dmd_qty(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=data, tb_name='M4E_O402131')

    def calc_prod_qty(self, data: pd.DataFrame):
        data = self.set_item_res_capa_rate(data=data)
        data[self._post.res_use_capa] = data[self._dmd.end_time] - data[self._dmd.start_time]
        data[self._post.res_use_capa] = data[self._post.res_use_capa] / np.timedelta64(1, 's')
        data[self._dmd.prod_qty] = np.round(data[self._post.res_use_capa] / data['capa_rate'], 2)

        return data

    def save_gantt_on_db(self, data: pd.DataFrame, seq: str) -> None:
        data = data.rename(columns={
            self._dmd.dmd: self._post.fp_key, 'start': self._dmd.start_time, 'end': self._dmd.end_time
        })

        data = data[data['kind'] == 'demand'].copy()
        data = self.calc_prod_qty(data=data)

        # Add information
        data = self.add_version_info(data=data, seq=seq)

        # add resource & item names
        data[self._res.res_grp_nm] = data[self._res.res_grp].apply(lambda x: self._res_grp_nm_map.get(x, 'UNDEFINED'))
        data[self._res.res_nm] = data[self._res.res].apply(lambda x: self._res_nm_map.get(x, 'UNDEFINED'))
        item_mst = self._item_mst[[self._item.sku, self._item.sku_nm]].drop_duplicates().copy()
        data = pd.merge(data, item_mst, how='left', on=self._item.sku)

        # Change data type (datetime -> string)
        data[self._dmd.start_time] = data[self._dmd.start_time].dt.strftime('%y%m%d%H%m%s')
        data[self._dmd.start_time] = data[self._dmd.start_time] \
            .str.replace('-', '') \
            .str.replace(':', '') \
            .str.replace(' ', '')
        data[self._dmd.end_time] = data[self._dmd.end_time].dt.strftime('%y%m%d%h%M%s')
        data[self._dmd.end_time] = data[self._dmd.end_time] \
            .str.replace('-', '') \
            .str.replace(':', '') \
            .str.replace(' ', '')

        data[self._res.plant] = self._plant

        data = data.drop(columns=['kind', 'capa_rate'])

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self._plant}
        self.io.delete_from_db(sql=self.query.del_gantt_result(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=data, tb_name='M4E_O402140')

    def save_res_status_on_db(self, data: pd.DataFrame, seq: str):
        #
        timeline_list = []
        for res, res_df in data.groupby(self._res.res):
            for kind, kind_df in res_df.groupby('kind'):
                for start, end in zip(kind_df['start'], kind_df['end']):
                    timeline_list.extend(self.calc_res_duration(res=res, kind=kind, start=start, end=end))

        res_status = pd.DataFrame(
            timeline_list,
            columns=[self._res.res, 'kind', self._post.date, self._post.time_idx, self._dmd.duration]
        )
        res_status[self._post.date] = res_status[self._post.date].dt.strftime('%Y%m%d')
        res_status[self._dmd.duration] = res_status[self._dmd.duration] / np.timedelta64(1, 's')

        res_status = res_status.groupby(by=[self._res.res, 'kind', self._post.date, self._post.time_idx]) \
            .sum() \
            .reset_index()

        res_status_dmd = res_status[res_status['kind'] == 'demand'].copy()
        res_status_dmd = res_status_dmd.rename(columns={self._dmd.duration: self._post.res_use_capa})
        res_status_dmd = res_status_dmd.drop(columns=['kind'])
        res_status_jc = res_status[res_status['kind'] == 'job_change'].copy()
        res_status_jc = res_status_jc.rename(columns={self._dmd.duration: 'res_jc_val'})
        res_status_jc = res_status_jc.drop(columns=['kind'])

        res_final = pd.merge(res_status_dmd, res_status_jc, how='left',
                             on=[self._res.res, self._post.date, self._post.time_idx]).fillna(0)

        # Resource usage
        res_final['day'] = [dt.datetime.strptime(day, '%Y%m%d').weekday() for day in res_final[self._post.date]]
        res_capa = []
        for res, day in zip(res_final[self._res.res], res_final['day']):
            res_avail_time = self._res_avail_time[res]
            res_capa.append(res_avail_time[day] * 60)
        res_final[self._res.res_capa] = res_capa

        # Resource capacity time
        res_capa_val = []
        for day, time_idx_type, capacity in zip(
                res_final['day'], res_final[self._post.time_idx], res_final[self._res.res_capa]):
            val = self.calc_day_night_res_capacity(day=day, time_idx_type=time_idx_type, capacity=capacity)
            res_capa_val.append(val)
        res_final['res_capa_val'] = res_capa_val

        # Resource unavailable time
        res_unavail_val = []
        for day, time_idx_type, capacity in zip(
                res_final['day'], res_final[self._post.time_idx], res_final['res_capa_val']):
            val = self.calc_res_unavail_time(day=day, capacity=capacity)
            res_unavail_val.append(val)
        res_final['res_unavail_val'] = res_unavail_val

        # Resource available time
        res_final['res_avail_val'] = res_final['res_capa_val'] - res_final['res_use_capa_val'] - res_final['res_jc_val']

        res_grp_mst = self._res_mst[[self._res.res, 'res_type_cd']]
        res_final = pd.merge(res_final, res_grp_mst, how='left', on=self._res.res).fillna('UNDEFINED')
        res_final = res_final.rename(columns={'res_type_cd': 'capa_type_cd'})

        # Add information
        res_final[self._res.plant] = self._plant
        res_final[self._res.res_grp] = [self._res_to_res_grp.get(res_cd, 'UNDEFINED')
                                        for res_cd in res_final[self._res.res]]
        res_final = self.add_version_info(data=res_final, seq=seq)

        res_final = res_final.drop(columns=[self._res.res_capa, 'day'])

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self._plant}
        self.io.delete_from_db(sql=self.query.del_res_status_result(**kwargs))

        # Save the result on DB
        self.io.insert_to_db(df=res_final, tb_name='M4E_O402050')

    def calc_res_unavail_time(self, day: int, capacity: int):
        val = 0
        if day in [0, 4]:
            val = self._sec_of_half_day - capacity
        elif day in [1, 2, 3]:
            val = 0   # ToDo: temp
            # val = self.sec_of_half_day * 2 - capacity    # ToDo: will be used

        return val

    def calc_day_night_res_capacity(self, day: int, time_idx_type: str, capacity: int):
        val = 0
        if day == 0:
            if time_idx_type == 'D':
                val = max(0, capacity - self._sec_of_half_day)
            elif time_idx_type == 'N':
                val = min(capacity, self._sec_of_half_day)
        elif day in [1, 2, 3]:
            val = self._sec_of_half_day    # ToDo: temp
            # val = capacity    # ToDo: will be used
        else:
            if time_idx_type == 'D':
                val = min(capacity, self._sec_of_half_day)
            elif time_idx_type == 'N':
                val = max(0, capacity - self._sec_of_half_day)

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
                    [res, kind, start_day + dt.timedelta(days=i + 1), 'D', self._split_hour])
                timeline.append(
                    [res, kind, start_day + dt.timedelta(days=i + 1), 'N', self._split_hour])

        return timeline

    def add_version_info(self, data: pd.DataFrame, seq: str):
        data['project_cd'] = self._project_cd
        data['create_user_cd'] = 'SYSTEM'
        data[self.col_fp_version_id] = self.fp_version
        data[self.col_fp_version_seq] = seq

        return data

    def _save_capa_profile(self, data: pd.DataFrame) -> None:
        data[self._res.plant] = self._plant
        data = self.add_version_info(data=data, seq=self.fp_seq)

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self._plant}
        self.io.delete_from_db(sql=self.query.del_human_capa_profile(**kwargs))

        # Save result
        self.io.insert_to_db(df=data, tb_name='M4E_O402150')

    def _save_capa_profile_dtl(self, data: pd.DataFrame) -> None:
        data[self._res.res_grp_nm] = [self._res_grp_nm_map.get(res_grp, '-') for res_grp in data[self._res.res_grp]]
        data[self._res.res_nm] = [self._res_nm_map.get(res, '-') for res in data[self._res.res]]

        data['capa_rate'] = [
            self._res_duration[item][res] for item, res in zip(data[self._item.sku], data[self._res.res])
        ]
        data[self._dmd.prod_qty] = np.round((data[self._post.res_use_capa] / data['capa_rate']).values, 2)
        data[self._post.res_use_capa] = np.round((data[self._post.res_use_capa] / 60).values, 2)

        data = self.add_version_info(data=data, seq=self.fp_seq)

        data = data[['project_cd', self.col_fp_version_id, self.col_fp_version_seq, self._res.plant,
                     self._cstr.floor, self._res.res_grp, self._res.res_grp_nm, self._res.res, self._res.res_nm,
                     self._item.sku, self._item.sku_nm, self._item.item_type, self._post.from_time, self._post.to_time,
                     self._post.from_yymmdd, self._post.to_yymmdd, self._dmd.prod_qty, self._post.tot_m_capa,
                     self._post.tot_w_capa, self._post.use_m_capa, self._post.use_w_capa, self._post.res_use_capa]]

        # Delete previous result
        kwargs = {'fp_version': self.fp_version, 'fp_seq': self.fp_seq, 'plant_cd': self._plant}
        self.io.delete_from_db(sql=self.query.del_human_capa_profile_dtl(**kwargs))

        # Save result
        self.io.insert_to_db(df=data, tb_name='M4E_O402151')
