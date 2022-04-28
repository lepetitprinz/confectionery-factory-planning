import common.util as util
import common.config as config

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Hashable
from itertools import permutations


class Preprocess(object):
    # Time configuration
    time_uom = 'sec'    # min / sec

    ############################################
    # Dictionary key configuration
    ############################################
    key_dmd = config.key_dmd      # Demand
    key_res = config.key_res      # Resource
    key_item = config.key_item    # Item master
    key_cstr = config.key_cstr    # Constraint

    # Demand
    key_dmd_list_by_plant = config.key_dmd_list_by_plant
    key_dmd_item_list_by_plant = config.key_dmd_item_list_by_plant
    key_dmd_res_grp_list_by_plant = config.key_dmd_res_grp_list_by_plant

    # Resource
    key_res_grp = config.key_res_grp          # Resource group code
    key_res_grp_nm = config.key_res_grp_nm    # Resource group name
    key_res_duration = config.key_res_duration    # Resource duration

    # Constraint
    key_jc = config.key_jc    # Job change
    key_sku_type = config.key_sku_type
    key_human_res = config.key_human_res
    key_human_capa = config.key_human_capa      # Human capacity
    key_human_usage = config.key_human_usage    # Human usage
    key_sim_prod_cstr = config.key_sim_prod_cstr      # Simultaneous Production constraint
    key_res_avail_time = config.key_res_avail_time    # Resource available time

    ############################################
    # Column name configuration
    ############################################
    # Demand
    col_dmd = config.col_dmd
    col_plant = config.col_plant
    col_qty = config.col_qty
    col_duration = config.col_duration
    col_due_date = config.col_due_date

    # Item
    col_sku = config.col_sku
    col_pkg = config.col_pkg
    col_brand = config.col_brand
    col_flavor = config.col_flavor

    # Resource
    col_res = config.col_res
    col_res_nm = config.col_res_nm
    col_res_grp = config.col_res_grp
    col_res_grp_nm = config.col_res_grp_nm
    col_res_type = config.col_res_type
    col_res_capa = config.col_res_capa
    col_capa_unit = config.col_capa_unit

    # Job change
    col_jc_type = config.col_job_change_type
    col_jc_from = config.col_job_change_from
    col_jc_to = config.col_job_change_to
    col_jc_time = config.col_job_change_time
    col_jc_unit = config.col_job_change_unit

    ############################################
    # Column usage configuration
    ############################################
    use_col_dmd = [col_dmd, col_sku, col_res_grp, col_qty, col_due_date]
    use_col_res_grp = [col_plant, col_res_grp, col_res, col_res_nm]
    use_col_res_duration = [col_plant, col_sku, col_res, col_duration]
    use_col_res_avail_time = [col_plant, col_res, 'capacity1', 'capacity2', 'capacity3', 'capacity4', 'capacity5']
    use_col_res_grp_job_change = [col_plant, col_res_grp, col_jc_from, col_jc_to, col_jc_type, col_jc_time, col_jc_unit]

    def __init__(self, cstr_cfg: dict, fp_version: str):
        self.cstr_cfg = cstr_cfg
        self.fp_version = fp_version

        # Plant instance attribute
        self.plant_list_in_dmd = []
        self.item_list_by_plant = {}
        self.res_grp_list_by_plant = []

        # Job change  instance attribute
        self.job_change_time_unit = 'MIN'

        # capacity
        self.work_day = 5

    def preprocess(self, data):
        ######################################
        # Demand & Resource
        ######################################
        demand = data[self.key_dmd]
        resource = data[self.key_res]
        constraint = data[self.key_cstr]

        if self.cstr_cfg['apply_prod_qty_multiple']:
            demand = util.change_dmd_qty(data=demand, method='multiple')

        dmd_prep = self.set_dmd_info(data=demand)
        res_prep = self.set_res_info(resource=resource, constraint=constraint)

        ######################################
        # Constraint (Option)
        ######################################
        # Resource available time
        res_avail_time = None
        if self.cstr_cfg['apply_res_available_time']:
            res_avail_time = self.set_res_avail_time(data=constraint[self.key_res_avail_time])

        # Job change
        job_change, sku_type = (None, None)
        if self.cstr_cfg['apply_job_change']:
            sku_type = self.set_sku_type_map(data=resource[self.key_item])
            job_change = self.set_job_change(data=constraint[self.key_jc])

        # Simultaneous production constraint
        sim_prod_cstr = None
        if self.cstr_cfg['apply_sim_prod_cstr']:
            sim_prod_cstr = self.set_sim_prod_cstr(data=constraint[self.key_sim_prod_cstr])

        # Human resource constraint
        human_res = None
        if self.cstr_cfg['apply_human_capacity']:
            human_res = self.set_human_cstr(
                capacity=constraint[self.key_human_capa],
                usage=constraint[self.key_human_usage]
            )

        prep_data = {
            self.key_dmd: dmd_prep,
            self.key_res: res_prep,
            self.key_cstr: {
                self.key_jc: job_change,
                self.key_sku_type: sku_type,
                self.key_human_capa: human_res,
                self.key_sim_prod_cstr: sim_prod_cstr,
                self.key_res_avail_time: res_avail_time,
            },
        }

        return prep_data

    def set_human_cstr(self, capacity: pd.DataFrame, usage: pd.DataFrame) -> Dict[str, Any]:
        human_resource = {
            self.key_human_capa: self.set_human_capacity(data=capacity),
            self.key_human_usage: self.set_human_usage(data=usage)
        }

        return human_resource

    def set_human_capacity(self, data: pd.DataFrame):
        # Change data type
        data['yy'] = data['yy'].astype(str)

        data = data[data[self.col_plant].isin(self.plant_list_in_dmd)].copy()

        # Choose current week capacity
        data['fp_version'] = 'FP_' + data['yy'] + data['week']
        data = data[data['fp_version'] == self.fp_version[:self.fp_version.index('.')]]

        human_capacity = {}
        for plant, plant_df in data.groupby(by=self.col_plant):
            floor_capa = {}
            for floor, m_capa, w_capa in zip(plant_df['floor'], plant_df['m_val'], plant_df['w_val']):
                floor_capa[floor] = (m_capa, w_capa)
            human_capacity[plant] = floor_capa

        return human_capacity

    def set_human_usage(self, data: pd.DataFrame):
        # Temp
        data.columns = [col.lower() for col in data.columns]

        # Change data type
        data[self.col_res_grp] = data[self.col_res_grp].astype(str)
        data[self.col_pkg] = data[self.col_pkg].astype(str)

        data = data[data[self.col_plant].isin(self.plant_list_in_dmd)].copy()

        human_usage = {}
        for plant, plant_df in data.groupby(by=self.col_plant):
            floor_res_grp = {}
            for floor, floor_df in plant_df.groupby(by='floor'):
                res_grp_item = {}
                for res_grp, res_grp_df in floor_df.groupby(by=self.col_res_grp):
                    item_capa = {}
                    for item, pkg, m_capa, w_capa in zip(
                            floor_df['item_attr04_cd'], floor_df['pkg'], floor_df['m_val'], floor_df['w_val']):
                        item_capa[(item, pkg)] = (m_capa, w_capa)
                    res_grp_item[res_grp] = item_capa
                floor_res_grp[floor] = res_grp_item
            human_usage[plant] = floor_res_grp

        return human_usage

    def set_sim_prod_cstr(self, data: pd.DataFrame):
        # Get plant list of demand list
        data = data[data[self.col_plant].isin(self.plant_list_in_dmd)].copy()

        # Change data type
        data[self.col_res_grp] = data[self.col_res_grp].astype(str)
        data[self.col_res + '1'] = data[self.col_res + '1'].astype(str)
        data[self.col_res + '2'] = data[self.col_res + '2'].astype(str)

        plant_sim_prod_cstr = {}
        for plant, plant_df in data.groupby(by=self.col_plant):
            res_grp_res = {}
            for res_grp, res_grp_df in plant_df.groupby(by=self.col_res_grp):
                res_grp_res[res_grp] = res_grp_df[[self.col_res + '1', self.col_res + '2']]\
                    .set_index(self.col_res + '1')\
                    .to_dict()[self.col_res + '2']
            plant_sim_prod_cstr[plant] = res_grp_res

        return plant_sim_prod_cstr

    def set_sku_type_map(self, data: pd.DataFrame) -> Dict[str, Tuple[str, str, str]]:
        sku_to_type = {item_cd: (brand, flavor, pkg) for item_cd, brand, flavor, pkg in
                       zip(data[self.col_sku], data[self.col_brand], data[self.col_flavor], data[self.col_pkg])}

        return sku_to_type

    def set_dmd_info(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        # Get plant list of demand list
        dmd_plant_list = list(set(data[self.col_plant]))

        # Calculate the due date
        self.calc_due_date(data=data)

        # Change data type
        data[self.col_sku] = data[self.col_sku].astype(str)
        data[self.col_res_grp] = data[self.col_res_grp].astype(str)
        data[self.col_qty] = np.ceil(data[self.col_qty]).astype(int)    # Ceiling quantity

        # Group demand by each plant
        dmd_list_by_plant, dmd_item_list_by_plant, dmd_res_grp_list_by_plant = {}, {}, {}
        for plant in dmd_plant_list:
            # Filter data by each plant
            dmd = data[data[self.col_plant] == plant]

            # Convert form of demand dataset
            dmd_list_by_plant[plant] = self.convert_dmd_form(data=dmd)

            # Get resource group only contained in demand
            dmd_res_grp_list_by_plant[plant] = list(set(dmd[self.col_res_grp].values))

            # All of demand item list by plant
            dmd_item_list_by_plant[plant] = list(set(dmd[self.col_sku].values))

        self.plant_list_in_dmd = dmd_plant_list
        self.item_list_by_plant = dmd_item_list_by_plant
        self.res_grp_list_by_plant = dmd_res_grp_list_by_plant

        dmd_prep = {
            self.key_dmd_list_by_plant: dmd_list_by_plant,
            self.key_dmd_item_list_by_plant: dmd_item_list_by_plant,
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

    def set_res_info(self, resource: dict, constraint: dict) -> Dict[str, Dict[str, Any]]:
        res_grp = resource[self.key_res_grp]

        # Filter resources whose available time does not exist
        if self.cstr_cfg['apply_res_available_time']:
            res_grp = self.filter_na_time(res_grp=res_grp, res_avail_time=constraint[self.key_res_avail_time])

        # Resource group <-> Resource information
        plant_res_grp, plant_res_nm = self.set_res_grp(data=res_grp)

        # Resource group naming
        plant_res_grp_nm = self.set_res_grp_nm(data=resource[self.key_res_grp_nm])

        # Resource group duration
        plant_res_duration = self.set_res_duration(data=resource[self.key_res_duration])

        res_prep = {
            self.key_res_grp: plant_res_grp,
            self.key_res_grp_nm: plant_res_grp_nm,
            'plant_res_nm': plant_res_nm,
            self.key_res_duration: plant_res_duration,
        }

        return res_prep

    def filter_na_time(self, res_grp: pd.DataFrame, res_avail_time: pd.DataFrame):
        compare_col = [self.col_plant, self.col_res]
        res_avail_time = res_avail_time[compare_col].copy()
        res_avail_time = res_avail_time.drop_duplicates()

        res_grp = pd.merge(res_grp, res_avail_time, how='inner', on=compare_col)

        return res_grp

    def filter_duplicate_capacity(self, data: pd.DataFrame):
        data = data.drop_duplicates()
        data_filtered = pd.DataFrame()
        for res_grp, res_grp_df in data.groupby(by=self.col_res_grp):
            for res_cd, res_df in res_grp_df.groupby(by=self.col_res):
                if len(res_df) == 1:
                    data_filtered = data_filtered.append(res_df)
                else:
                    data_filtered = data_filtered.append(res_df.max(), ignore_index=True)

        data_filtered = data_filtered.reset_index(drop=True)

        return data_filtered

    def set_res_avail_time(self, data: pd.DataFrame):
        # Choose columns used in model
        data = data[self.use_col_res_avail_time].copy()

        # choose capacity in several item case
        # data = self.filter_duplicate_capacity(data=data)

        # Change the data type
        data[self.col_res] = data[self.col_res].astype(str)

        capa_col_list = []
        for i in range(self.work_day):
            capa_col = self.col_res_capa + str(i + 1)
            capa_col_list.append(capa_col)
            data[capa_col] = data[capa_col].astype(int)

        # Choose plants of demand list
        data = data[data[self.col_plant].isin(self.plant_list_in_dmd)].copy()

        res_avail_time_by_plant = {}
        for plant, res_df in data.groupby(by=self.col_plant):
            res_to_capa = {}
            for res, capa_df in res_df.groupby(by=self.col_res):
                res_to_capa[res] = capa_df[capa_col_list].values.tolist()[0]
            res_avail_time_by_plant[plant] = res_to_capa

        return res_avail_time_by_plant

    def set_res_grp(self, data: pd.DataFrame) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, str]]]:
        # Choose columns used in model
        data = data[self.use_col_res_grp].copy()

        # Change the data type
        data[self.col_res] = data[self.col_res].astype(str)
        data[self.col_res_grp] = data[self.col_res_grp].astype(str)

        # Choose plants of demand list
        data = data[data[self.col_plant].isin(self.plant_list_in_dmd)].copy()

        # Group resource by each plant
        res_grp_by_plant = {}
        res_nm_by_plant = {}
        for plant in self.plant_list_in_dmd:
            # Filter data by each plant
            res_grp_df = data[data[self.col_plant] == plant]
            res_grp_df = res_grp_df[res_grp_df[self.col_res_grp].isin(self.res_grp_list_by_plant[plant])]

            # Resource group -> resource list
            res_grp_to_res = {}
            for res_grp, group in res_grp_df.groupby(self.col_res_grp):
                res_grp_to_res[res_grp] = group[self.col_res].tolist()

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

    def set_res_duration(self, data: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, int]]]:
        # Choose columns used in model
        data = data[self.use_col_res_duration].copy()

        # Change data type
        data[self.col_res] = data[self.col_res].astype(str)
        data[self.col_sku] = data[self.col_sku].astype(str)
        data[self.col_duration] = data[self.col_duration].astype(int)

        # Group bom route by each plant
        res_duration_by_plant = {}
        for plant in self.plant_list_in_dmd:
            # Filter data contained in each plant
            res_duration = data[data[self.col_plant] == plant].copy()

            # Filter items of each plant involved in demand
            res_duration = res_duration[res_duration[self.col_sku].isin(self.item_list_by_plant[plant])]

            # item -> resource -> duration mapping
            res_grp_duration_map = {}
            for item, group in res_duration.groupby(self.col_sku):
                res_grp_duration_map[item] = group[[self.col_res, self.col_duration]]\
                    .set_index(self.col_res)\
                    .to_dict()[self.col_duration]

            res_duration_by_plant[plant] = res_grp_duration_map

        return res_duration_by_plant

    def set_job_change(self, data):
        # Change time unit to second
        unit = list(set(data[self.col_jc_unit]))[0]

        if unit == 'MIN':
            data[self.col_jc_time] = data[self.col_jc_time] * 60

        # Filter demand plant
        data = data[data[self.col_plant].isin(self.plant_list_in_dmd)]

        job_change = {}
        for plant, res_grp_df in data.groupby(by=self.col_plant):
            res_grp_to_type = {}
            for res_grp, type_df in res_grp_df.groupby(by=self.col_res_grp):
                type_to_time = {}
                for jc_type, time_df in type_df.groupby(by=self.col_jc_type):
                    type_to_time[jc_type] = {(from_res, to_res): time for from_res, to_res, time in zip(
                        time_df[self.col_jc_from], time_df[self.col_jc_to], time_df[self.col_jc_time]
                    )}
                res_grp_to_type[res_grp] = type_to_time
            job_change[plant] = res_grp_to_type

        return job_change

    def match_job_change_time(self, candidate: dict, job_change_master: pd.DataFrame) -> Dict[str, Dict[Hashable, Any]]:
        # Change time unit
        if self.job_change_time_unit == 'MIN':
            job_change_master[self.col_jc_time] = job_change_master[self.col_jc_time] * 60

        job_change_by_plant = {}
        for plant_cd, job_change_cand in candidate.items():
            job_change = pd.merge(
                job_change_cand,
                job_change_master,
                how='left',
                on=[self.col_res_grp, self.col_jc_from, self.col_jc_to]
            )

            job_change[self.col_jc_time] = job_change[self.col_jc_time].fillna(0)

            res_grp_job_change = {}
            for res_grp_cd, res_grp_df in job_change.groupby(by=self.col_res_grp):
                from_to_res = {}
                for from_res, to_res, job_change_time in zip(
                        res_grp_df[self.col_jc_from], res_grp_df[self.col_jc_to], res_grp_df[self.col_jc_time]):
                    from_to_res[(from_res, to_res)] = job_change_time
                res_grp_job_change[res_grp_cd] = from_to_res
            job_change_by_plant[plant_cd] = res_grp_job_change

        return job_change_by_plant

    def set_plant_job_change_bak(self, demand: pd.DataFrame, resource: dict) -> Dict[str, Dict[Hashable, Any]]:
        # add brand information
        merged = pd.merge(
            demand,
            resource[self.key_item][[self.col_brand, self.col_sku]],
            on=[self.col_sku],
            how='left'
        )
        # resource group by brand
        res_grp_brand = merged[[self.col_plant, self.col_res_grp, self.col_brand]].drop_duplicates()

        plant_job_change_cand = self.set_job_chage_cand(data=res_grp_brand)

        plant_job_change = self.match_job_change_time(
            candidate=plant_job_change_cand,
            job_change_master=resource['job_change']
        )

        return plant_job_change

    def set_job_chage_cand(self, data) -> Dict[str, pd.DataFrame]:
        job_change_cand_by_plant = {}

        for plant_cd, plant_df in data.groupby(by=self.col_plant):
            job_change_cand = pd.DataFrame()
            for res_grp, res_grp_df in plant_df.groupby(by=self.col_res_grp):
                if len(res_grp_df) > 1:
                    brand_seq_list = list(permutations(res_grp_df[self.col_brand], 2))
                    temp = pd.DataFrame(brand_seq_list, columns=[self.col_jc_from, self.col_jc_to])
                    temp[self.col_res_grp] = res_grp
                    job_change_cand = pd.concat([job_change_cand, temp])

            job_change_cand_by_plant[plant_cd] = job_change_cand

        return job_change_cand_by_plant
