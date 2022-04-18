import common.config as config

import numpy as np
import pandas as pd
import datetime as dt
from typing import List, Tuple, Dict, Any, Hashable
from itertools import permutations


class Preprocess(object):
    # Time setting
    time_uom = 'sec'    # min / sec

    # Column name setting
    col_dmd = config.col_dmd
    col_plant = config.col_plant
    col_brand = config.col_brand
    col_sku = config.col_sku
    col_due_date = config.col_due_date
    col_qty = config.col_qty
    col_res = config.col_res
    col_res_nm = config.col_res_nm
    col_res_grp = config.col_res_grp
    col_res_grp_nm = config.col_res_grp_nm
    col_res_map = config.col_res_map
    col_res_type = config.col_res_type
    col_capacity = config.col_capacity
    col_capa_unit = config.col_capa_unit
    col_duration = config.col_duration

    col_job_change_from = config.col_job_change_from
    col_job_change_to = config.col_job_change_to
    col_job_change_time = config.col_job_change_time

    # Column usage setting
    use_col_dmd = ['dmd_id', 'item_cd', 'res_grp_cd', 'qty', 'due_date']
    use_col_res_grp = ['plant_cd', 'res_grp_cd', 'res_cd', 'res_nm', 'capacity', 'capa_unit_cd', 'res_type_cd']
    use_col_item_res_duration = ['plant_cd', 'item_cd', 'res_cd', 'duration']
    use_col_res_grp_job_change = []
    use_col_res_people_map = ['plant_cd', 'res_grp_cd', 'res_cd', 'res_map_cd']

    def __init__(self, cstr_cfg):
        self.cstr_cfg = cstr_cfg

        # Plant instance attribute
        self.plant_dmd_list = []
        self.plant_dmd_res_list = []
        self.plant_dmd_item = {}

        # Job change  instance attribute
        self.job_change_time_unit = 'MIN'

    def preprocess(self, demand, master):
        dmd_prep = self.set_dmd_info(data=demand)    # Demand
        res_prep = self.set_res_info(data=master)    # Resource

        job_change, sku_to_brand = (None, None)
        if self.cstr_cfg['apply_job_change']:  # Job change
            job_change = self.set_plant_job_change(demand=demand, master=master)
            sku_to_brand = self.set_sku_to_brand_map(data=master['item'])

        prep_data = {
            'demand': dmd_prep,
            'resource': res_prep,
            'job_change': job_change,
            'sku_to_brand': sku_to_brand
        }

        return prep_data

    def set_sku_to_brand_map(self, data: pd.DataFrame) -> Dict[str, str]:
        sku_to_brand = {item_cd: brand_cd for item_cd, brand_cd in zip(data[self.col_sku], data[self.col_brand])}

        return sku_to_brand

    def set_dmd_info(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        # Get plant list of demand list
        plant_dmd_list = list(set(data[self.col_plant]))

        # Calculate the due date
        self.calc_due_date(data=data)

        # Change data type
        data[self.col_sku] = data[self.col_sku].astype(str)
        data[self.col_res_grp] = data[self.col_res_grp].astype(str)
        data[self.col_qty] = np.ceil(data[self.col_qty]).astype(int)    # Ceiling quantity

        # Group demand by each plant
        plant_dmd, plant_dmd_item, plant_dmd_res, plant_dmd_due = {}, {}, {}, {}
        for plant in plant_dmd_list:
            # Filter data by each plant
            dmd = data[data[self.col_plant] == plant]

            # Convert form of demand dataset
            plant_dmd[plant] = self.convert_dmd_form(data=dmd)

            # Get resource group only contained in demand
            plant_dmd_res[plant] = list(set(dmd[self.col_res_grp].values))

            # All of demand item list by plant
            plant_dmd_item[plant] = list(set(dmd[self.col_sku]))

            # Set demand due date by plant
            plant_dmd_due[plant] = self.set_plant_dmd_due(data=dmd)

        self.plant_dmd_list = plant_dmd_list
        self.plant_dmd_item = plant_dmd_item
        self.plant_dmd_res_list = plant_dmd_res

        dmd_prep = {
            'plant_dmd_list': plant_dmd,
            'plant_dmd_item': plant_dmd_item,
            'plant_dmd_due': plant_dmd_due
        }

        return dmd_prep

    def calc_due_date(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.col_due_date] = data[self.col_due_date] * 24 * 60 * 60
        data[self.col_due_date] = data[self.col_due_date].astype(int)

        return data

    def convert_dmd_form(self, data: pd.DataFrame) -> List[Tuple[Any]]:
        data_use = data[self.use_col_dmd].copy()
        data_tuple = [tuple(row) for row in data_use.to_numpy()]

        return data_tuple

    def set_plant_dmd_due(self, data: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        temp = data[[self.col_dmd, self.col_sku, self.col_due_date]].copy()
        plant_dmd_due = {}
        for demand, group in temp.groupby(self.col_dmd):
            plant_dmd_due[demand] = group[[self.col_sku, self.col_due_date]]\
                .set_index(self.col_sku)\
                .to_dict()[self.col_due_date]

        return plant_dmd_due

    def set_res_info(self, data: dict) -> Dict[str, Dict[str, Any]]:
        #
        plant_res_grp, plant_res_nm = self.set_res_grp(data=data['res_grp'])

        # Resource group naming
        plant_res_grp_nm = self.set_res_grp_nm(data=data['res_grp_nm'])

        # Resource group duration
        plant_item_res_duration = self.set_item_res_duration(data=data['item_res_duration'])

        # Resource job change
        # plant_res_grp_job_change = self.set_res_grp_job_change(data=data['job_change'])

        # plant_res_human = self.set_res_grp(data=data['res_human'])
        # plant_res_human_map = self.set_res_human_map(data=data['res_human_map'])

        res_prep = {
            'plant_res_grp': plant_res_grp,
            'plant_res_grp_nm': plant_res_grp_nm,
            'plant_res_nm': plant_res_nm,
            'plant_item_res_duration': plant_item_res_duration,
            # 'plant_res_grp_job_change': plant_res_grp_job_change,
            # 'plant_res_human': plant_res_human,
            # 'plant_res_human_map': plant_res_human_map,
        }

        return res_prep

    def set_plant_job_change(self, demand: pd.DataFrame, master: dict) -> Dict[str, Dict[str, Any]]:
        merged = pd.merge(
            demand,
            master['item'][[self.col_brand, self.col_sku]],
            on=[self.col_sku],
            how='left'
        )
        # resource group by brand
        res_grp_brand = merged[[self.col_plant, self.col_res_grp, self.col_brand]].drop_duplicates()
        plant_job_change_cand = self.set_job_chage_cand(data=res_grp_brand)

        plant_job_change = self.match_job_change_time(
            candidate=plant_job_change_cand,
            job_change_master=master['job_change']
        )

        return plant_job_change

    def match_job_change_time(self, candidate: dict, job_change_master: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        # Change time unit
        if self.job_change_time_unit == 'MIN':
            job_change_master[self.col_job_change_time] = job_change_master[self.col_job_change_time] * 60

        job_change_by_plant = {}
        for plant_cd, job_change_cand in candidate.items():
            job_change = pd.merge(
                job_change_cand,
                job_change_master,
                how='left',
                on=[self.col_res_grp, self.col_job_change_from, self.col_job_change_to]
            )

            job_change[self.col_job_change_time] = job_change[self.col_job_change_time].fillna(0)

            res_grp_job_change = {}
            for res_grp_cd, res_grp_df in job_change.groupby(by=self.col_res_grp):
                from_to_res = {}
                for from_res, to_res, job_change_time in zip(
                        res_grp_df[self.col_job_change_from],
                        res_grp_df[self.col_job_change_to],
                        res_grp_df[self.col_job_change_time]):
                    from_to_res[(from_res, to_res)] = job_change_time
                res_grp_job_change[res_grp_cd] = from_to_res
            job_change_by_plant[plant_cd] = res_grp_job_change

        return job_change_by_plant

    def set_job_chage_cand(self, data) -> Dict[str, pd.DataFrame]:
        job_change_cand_by_plant = {}

        for plant_cd, plant_df in data.groupby(by=self.col_plant):
            job_change_cand = pd.DataFrame()
            for res_grp, res_grp_df in plant_df.groupby(by=self.col_res_grp):
                if len(res_grp_df) > 1:
                    brand_seq_list = list(permutations(res_grp_df[self.col_brand], 2))
                    temp = pd.DataFrame(brand_seq_list, columns=['from_res_cd', 'to_res_cd'])
                    temp[self.col_res_grp] = res_grp
                    job_change_cand = pd.concat([job_change_cand, temp])

            job_change_cand_by_plant[plant_cd] = job_change_cand

        return job_change_cand_by_plant

    def set_res_grp(self, data: pd.DataFrame) -> Tuple[Dict[str, Dict[str, List[Tuple]]], Dict[str, Dict[str, str]]]:
        # Rename columns
        data = data.rename(columns={'res_capa_val': self.col_capacity})

        # Choose columns used in model
        data = data[self.use_col_res_grp].copy()

        # Change the data type
        data[self.col_res] = data[self.col_res].astype(str)
        data[self.col_res_grp] = data[self.col_res_grp].astype(str)
        data[self.col_capacity] = data[self.col_capacity].astype(int)

        # Choose plants of demand list
        data = data[data[self.col_plant].isin(self.plant_dmd_list)].copy()

        # Group resource by each plant
        res_grp_by_plant = {}
        res_nm_by_plant = {}
        for plant in self.plant_dmd_list:
            # Filter
            res_grp_df = data[data[self.col_plant] == plant]
            res_grp_df = res_grp_df[res_grp_df[self.col_res_grp].isin(self.plant_dmd_res_list[plant])]
            # Resource group -> (resource / capacity / resource type)
            res_grp_to_res = {}
            for res_grp, group in res_grp_df.groupby(self.col_res_grp):
                res_grp_to_res[res_grp] = [tuple(row) for row in group[
                    [self.col_res, self.col_capacity, self.col_capa_unit, self.col_res_type]].to_numpy()]

            res_grp_by_plant[plant] = res_grp_to_res

            res_nm_by_plant[plant] = {str(res_cd): res_nm for res_cd, res_nm in
                                      zip(res_grp_df[self.col_res], res_grp_df[self.col_res_nm])}

        return res_grp_by_plant, res_nm_by_plant

    def set_res_grp_nm(self, data: pd.DataFrame) -> Dict[Hashable, dict]:
        res_grp_nm_by_plant = {}
        for plant_cd, plant_df in data.groupby(by=self.col_plant):
            res_grp_nm_by_plant[plant_cd] = {res_grp_cd: res_grp_nm for res_grp_cd, res_grp_nm
                                             in zip(plant_df[self.col_res_grp], plant_df[self.col_res_grp_nm])}

        return res_grp_nm_by_plant

    def set_item_res_duration(self, data: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, int]]]:
        # Choose columns used in model
        data = data[self.use_col_item_res_duration].copy()

        # Change data type
        data[self.col_res] = data[self.col_res].astype(str)
        data[self.col_sku] = data[self.col_sku].astype(str)
        data[self.col_duration] = data[self.col_duration].astype(int)

        # Group bom route by each plant
        item_res_duration_by_plant = {}
        for plant in self.plant_dmd_list:
            # Filter data contained in each plant
            item_res_duration = data[data[self.col_plant] == plant].copy()

            # Filter items of each plant involved in demand
            item_res_duration = item_res_duration[item_res_duration[self.col_sku].isin(self.plant_dmd_item[plant])]

            # item -> resource -> duration mapping
            item_res_grp_duration_map = {}
            for item, group in item_res_duration.groupby(self.col_sku):
                item_res_grp_duration_map[item] = group[[self.col_res, self.col_duration]]\
                    .set_index(self.col_res)\
                    .to_dict()[self.col_duration]

            item_res_duration_by_plant[plant] = item_res_grp_duration_map

        return item_res_duration_by_plant

    def calc_deadline(self, data: pd.DataFrame) -> pd.DataFrame:
        days = data[self.col_due_date] - dt.datetime.now()   # ToDo : need to revise start day

        due_date = None
        if self.time_uom == 'min':
            due_date = np.round(days / np.timedelta64(1, 'm'), 0)
        elif self.time_uom == 'sec':
            due_date = np.round(days / np.timedelta64(1, 's'), 0)

        data[self.col_due_date] = due_date.astype(int)

        return data

    def set_res_human_map(self, data: pd.DataFrame) -> dict:
        # Choose columns used in model
        data = data[self.use_col_res_people_map].copy()

        # Change data type
        data[self.col_res] = data[self.col_res].astype(str)
        data[self.col_res_grp] = data[self.col_res_grp].astype(str)
        data[self.col_res_map] = data[self.col_res_map].astype(str)

        # Choose plants of demand list
        data = data[data[self.col_plant].isin(self.plant_dmd_list)].copy()

        # Group resource by each plant
        res_human_by_plant = {}
        for plant in self.plant_dmd_list:
            res_human_df = data[data[self.col_plant] == plant]

            res_to_human = {}
            for res_grp, group in res_human_df.groupby(self.col_res_grp):
                res_to_human[res_grp] = group\
                    .groupby(by=self.col_res_map)[self.col_res]\
                    .apply(list)\
                    .to_dict()

            res_human_by_plant[plant] = res_to_human

        return res_human_by_plant
