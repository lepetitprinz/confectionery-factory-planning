import common.util as util
from common.name import Key, Demand, Item, Route, Resource, Constraint

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Hashable
from itertools import permutations


class Preprocess(object):
    # Time configuration
    time_multiple = {
        'DAY': 86400,
        'HOUR': 3600,
        'MIN': 60,
    }

    time_uom = 'sec'    # min / sec
    work_day = 5

    def __init__(self, cstr_cfg: dict, version):
        self.version = version
        self.cstr_cfg = cstr_cfg

        # Name instance attribute
        self.key = Key()
        self.dmd = Demand()
        self.res = Resource()
        self.item = Item()
        self.route = Route()
        self.cstr = Constraint()

        # Column usage
        self.col_dmd = [self.dmd.dmd, self.item.sku, self.res.res_grp, self.dmd.qty, self.dmd.due_date]
        self.col_res_grp = [self.res.plant, self.res.res_grp, self.res.res, self.res.res_nm]
        self.col_res_duration = [self.res.plant, self.item.sku, self.res.res, self.dmd.duration]
        self.col_res_avail_time = [self.res.plant, self.res.res] + \
                                  [self.res.res_capa + str(i+1) for i in range(self.work_day)]
        self.col_res_grp_job_change = [self.res.plant, self.res.res_grp, self.cstr.jc_from, self.cstr.jc_to,
                                       self.cstr.jc_type, self.cstr.jc_time, self.cstr.jc_unit]

        # Plant instance attribute
        self.plant_list_in_dmd = []
        self.item_list_by_plant = {}
        self.res_grp_list_by_plant = []

        # Time UOM instance attribute
        self.jc_time_uom = 'MIN'

    def preprocess(self, data):
        ######################################
        # Demand & Resource
        ######################################
        demand = data[self.key.dmd]
        resource = data[self.key.res]
        constraint = data[self.key.cstr]

        if self.cstr_cfg['apply_prod_qty_multiple']:
            demand = util.change_dmd_qty(data=demand, method='multiple')

        dmd_prep = self.set_dmd_info(data=demand)
        res_prep = self.set_res_info(resource=resource, constraint=constraint)

        ######################################
        # Route
        ######################################
        route = data[self.key.route]
        route_prep = self.set_route_info(data=route[self.key.route])

        ######################################
        # Constraint (Option)
        ######################################
        # Resource available time
        res_avail_time = None
        if self.cstr_cfg['apply_res_available_time']:
            res_avail_time = self.set_res_avail_time(data=constraint[self.key.res_avail_time])

        # Job change
        job_change, sku_type = (None, None)
        if self.cstr_cfg['apply_job_change']:
            sku_type = self.set_sku_type_map(data=resource[self.key.item])
            job_change = self.set_job_change(data=constraint[self.key.jc])

        # Simultaneous production constraint
        sim_prod_cstr = None
        if self.cstr_cfg['apply_sim_prod_cstr']:
            sim_prod_cstr = self.set_sim_prod_cstr(data=constraint[self.key.sim_prod_cstr])

        # Human resource constraint
        human_res = None
        if self.cstr_cfg['apply_human_capacity']:
            human_res = self.set_human_cstr(
                capacity=constraint[self.key.human_capa],
                usage=constraint[self.key.human_usage]
            )

        prep_data = {
            self.key.dmd: dmd_prep,
            self.key.res: res_prep,
            self.key.cstr: {
                self.key.jc: job_change,
                self.key.sku_type: sku_type,
                self.key.human_res: human_res,
                self.key.sim_prod_cstr: sim_prod_cstr,
                self.key.res_avail_time: res_avail_time,
            },
        }

        return prep_data

    def set_route_info(self, data):
        data[self.route.lead_time] = [int(lt * self.time_multiple[uom]) for lt, uom in zip(
            data[self.route.lead_time], data[self.route.time_uom])]

        data = data.drop(columns=[self.route.time_uom])

        return data

    def set_human_cstr(self, capacity: pd.DataFrame, usage: pd.DataFrame) -> Dict[str, Any]:
        human_resource = {
            self.key.human_capa: self.set_human_capacity(data=capacity),
            self.key.human_usage: self.set_human_usage(data=usage)
        }

        return human_resource

    def set_human_capacity(self, data: pd.DataFrame) -> Dict[str, Dict[str, Tuple[int, int]]]:
        # Change data type
        data['yy'] = data['yy'].astype(str)

        data = data[data[self.res.plant].isin(self.plant_list_in_dmd)].copy()

        # Choose current week capacity
        # ToDo: Temporal code
        data['fp_version'] = 'FP_' + data['yy'] + data['week']
        data = data[data['fp_version'] == self.version.fp_version[:self.version.fp_version.index('.')]]

        human_capacity = {}
        for plant, plant_df in data.groupby(by=self.res.plant):
            floor_capa = {}
            for floor, m_capa, w_capa in zip(
                    plant_df[self.cstr.floor], plant_df[self.cstr.m_capa], plant_df[self.cstr.w_capa]
            ):
                floor_capa[floor] = (m_capa, w_capa)
            human_capacity[plant] = floor_capa

        return human_capacity

    def set_human_usage(self, data: pd.DataFrame):
        # Change data type
        data[self.res.res_grp] = data[self.res.res_grp].astype(str)
        data[self.item.pkg] = data[self.item.pkg].astype(str)

        data = data[data[self.res.plant].isin(self.plant_list_in_dmd)].copy()

        human_usage = {}
        for plant, plant_df in data.groupby(by=self.res.plant):
            floor_res_grp = {}
            for floor, floor_df in plant_df.groupby(by=self.cstr.floor):
                res_grp_item = {}
                for res_grp, res_grp_df in floor_df.groupby(by=self.res.res_grp):
                    item_capa = {}
                    for item, pkg, m_capa, w_capa in zip(
                            floor_df[self.item.item], floor_df[self.item.pkg],
                            floor_df[self.cstr.m_capa], floor_df[self.cstr.w_capa]):
                        item_capa[(item, pkg)] = (m_capa, w_capa)
                    res_grp_item[res_grp] = item_capa
                floor_res_grp[floor] = res_grp_item
            human_usage[plant] = floor_res_grp

        return human_usage

    def set_sim_prod_cstr(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        # Get plant list of demand list
        data = data[data[self.res.plant].isin(self.plant_list_in_dmd)].copy()

        # Change data type
        data[self.res.res_grp] = data[self.res.res_grp].astype(str)
        data[self.item.pkg + '1'] = data[self.item.pkg + '1'].astype(str)
        data[self.item.pkg + '2'] = data[self.item.pkg + '2'].astype(str)

        # Simultaneous type : necessary
        necessary = data[data['sim_type'] == 'NEC']
        nece_sim_prod_cstr = self.make_sim_prod_cstr_map(data=necessary)

        # Simultaneous type : impossible
        impo_sim_prod_cstr = None
        impossible = data[data['sim_type'] == 'IMP']
        if len(impossible) > 0:
            impo_sim_prod_cstr = self.make_sim_prod_cstr_map(data=impossible)

        plant_sim_prod_cstr = {
            'necessary': nece_sim_prod_cstr,
            'impossible': impo_sim_prod_cstr
        }

        return plant_sim_prod_cstr

    def make_sim_prod_cstr_map(self, data):
        sim_prod_cstr = {}
        for plant, plant_df in data.groupby(by=self.res.plant):
            res_grp_brand = {}
            for res_grp, res_grp_df in plant_df.groupby(by=self.res.res_grp):
                brand_pkg = {}
                for brand, brand_df in res_grp_df.groupby(by=self.item.brand):
                    pkg1_pkg2 = {}
                    for pkg1, pkg2 in zip(brand_df[self.item.pkg + '1'], brand_df[self.item.pkg + '2']):
                        pkg1_pkg2[pkg1] = pkg2
                    brand_pkg[brand] = pkg1_pkg2
                res_grp_brand[res_grp] = brand_pkg
            sim_prod_cstr[plant] = res_grp_brand

        return sim_prod_cstr

    def set_sku_type_map(self, data: pd.DataFrame) -> Dict[str, Tuple[str, str, str]]:
        sku_to_type = {item_cd: (brand, flavor, pkg) for item_cd, brand, flavor, pkg in
                       zip(data[self.item.sku], data[self.item.brand], data[self.item.flavor], data[self.item.pkg])}

        return sku_to_type

    def set_dmd_info(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        # Get plant list of demand list
        dmd_plant_list = list(set(data[self.res.plant]))

        # Calculate the due date
        self.calc_due_date(data=data)

        # Change data type
        data[self.item.sku] = data[self.item.sku].astype(str)
        data[self.res.res_grp] = data[self.res.res_grp].astype(str)
        data[self.dmd.qty] = np.ceil(data[self.dmd.qty]).astype(int)    # Ceiling quantity

        # Group demand by each plant
        dmd_list_by_plant, dmd_item_list_by_plant, dmd_res_grp_list_by_plant = {}, {}, {}
        for plant in dmd_plant_list:
            # Filter data by each plant
            dmd = data[data[self.res.plant] == plant]

            # Convert form of demand dataset
            dmd_list_by_plant[plant] = self.convert_dmd_form(data=dmd)

            # Get resource group only contained in demand
            dmd_res_grp_list_by_plant[plant] = list(set(dmd[self.res.res_grp].values))

            # All of demand item list by plant
            dmd_item_list_by_plant[plant] = list(set(dmd[self.item.sku].values))

        self.plant_list_in_dmd = dmd_plant_list
        self.item_list_by_plant = dmd_item_list_by_plant
        self.res_grp_list_by_plant = dmd_res_grp_list_by_plant

        dmd_prep = {
            self.key.dmd_list_by_plant: dmd_list_by_plant,
            self.key.dmd_item_list_by_plant: dmd_item_list_by_plant,
        }

        return dmd_prep

    def calc_due_date(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.dmd.due_date] = data[self.dmd.due_date] * 24 * 60 * 60
        data[self.dmd.due_date] = data[self.dmd.due_date].astype(int)

        return data

    def convert_dmd_form(self, data: pd.DataFrame) -> List[Tuple[Any]]:
        data_use = data[self.col_dmd].copy()
        data_tuple = [tuple(row) for row in data_use.to_numpy()]

        return data_tuple

    def set_res_info(self, resource: dict, constraint: dict) -> Dict[str, Dict[str, Any]]:
        res_grp = resource[self.key.res_grp]

        # Filter resources whose available time does not exist
        if self.cstr_cfg['apply_res_available_time']:
            res_grp = self.filter_na_time(res_grp=res_grp, res_avail_time=constraint[self.key.res_avail_time])

        # Resource group <-> Resource information
        plant_res_grp, plant_res_nm = self.set_res_grp(data=res_grp)

        # Resource group naming
        plant_res_grp_nm = self.set_res_grp_nm(data=resource[self.key.res_grp_nm])

        # Resource group duration
        plant_res_duration = self.set_res_duration(data=resource[self.key.res_duration])

        res_prep = {
            self.key.res_nm: plant_res_nm,
            self.key.res_grp: plant_res_grp,
            self.key.res_grp_nm: plant_res_grp_nm,
            self.key.res_duration: plant_res_duration,
        }

        return res_prep

    def filter_na_time(self, res_grp: pd.DataFrame, res_avail_time: pd.DataFrame) -> pd.DataFrame:
        compare_col = [self.res.plant, self.res.res]
        res_avail_time = res_avail_time[compare_col].copy()
        res_avail_time = res_avail_time.drop_duplicates()

        res_grp = pd.merge(res_grp, res_avail_time, how='inner', on=compare_col)

        return res_grp

    def filter_duplicate_capacity(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop_duplicates()
        data_filtered = pd.DataFrame()
        for res_grp, res_grp_df in data.groupby(by=self.res.res_grp):
            for res_cd, res_df in res_grp_df.groupby(by=self.res.res):
                if len(res_df) == 1:
                    data_filtered = data_filtered.append(res_df)
                else:
                    data_filtered = data_filtered.append(res_df.max(), ignore_index=True)

        data_filtered = data_filtered.reset_index(drop=True)

        return data_filtered

    def set_res_avail_time(self, data: pd.DataFrame):
        # Choose columns used in model
        data = data[self.col_res_avail_time].copy()

        # choose capacity in several item case
        # data = self.filter_duplicate_capacity(data=data)

        # Change the data type
        data[self.res.res] = data[self.res.res].astype(str)

        capa_col_list = []
        for i in range(self.work_day):
            capa_col = self.res.res_capa + str(i + 1)
            capa_col_list.append(capa_col)
            data[capa_col] = data[capa_col].astype(int)

        # Choose plants of demand list
        data = data[data[self.res.plant].isin(self.plant_list_in_dmd)].copy()

        res_avail_time_by_plant = {}
        for plant, res_df in data.groupby(by=self.res.plant):
            res_to_capa = {}
            for res, capa_df in res_df.groupby(by=self.res.res):
                res_to_capa[res] = capa_df[capa_col_list].values.tolist()[0]
            res_avail_time_by_plant[plant] = res_to_capa

        return res_avail_time_by_plant

    def set_res_grp(self, data: pd.DataFrame) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, str]]]:
        # Choose columns used in model
        data = data[self.col_res_grp].copy()

        # Change the data type
        data[self.res.res] = data[self.res.res].astype(str)
        data[self.res.res_grp] = data[self.res.res_grp].astype(str)

        # Choose plants of demand list
        data = data[data[self.res.plant].isin(self.plant_list_in_dmd)].copy()

        # Group resource by each plant
        res_grp_by_plant = {}
        res_nm_by_plant = {}
        for plant in self.plant_list_in_dmd:
            # Filter data by each plant
            res_grp_df = data[data[self.res.plant] == plant]
            res_grp_df = res_grp_df[res_grp_df[self.res.res_grp].isin(self.res_grp_list_by_plant[plant])]

            # Resource group -> resource list
            res_grp_to_res = {}
            for res_grp, group in res_grp_df.groupby(self.res.res_grp):
                res_grp_to_res[res_grp] = group[self.res.res].tolist()

            res_grp_by_plant[plant] = res_grp_to_res

            res_nm_by_plant[plant] = {str(res_cd): res_nm for res_cd, res_nm in
                                      zip(res_grp_df[self.res.res], res_grp_df[self.res.res_nm])}

        return res_grp_by_plant, res_nm_by_plant

    def set_res_grp_nm(self, data: pd.DataFrame) -> Dict[Hashable, dict]:
        res_grp_nm_by_plant = {}
        for plant_cd, plant_df in data.groupby(by=self.res.plant):
            res_grp_nm_by_plant[plant_cd] = {res_grp_cd: res_grp_nm for res_grp_cd, res_grp_nm
                                             in zip(plant_df[self.res.res_grp], plant_df[self.res.res_grp_nm])}

        return res_grp_nm_by_plant

    def set_res_duration(self, data: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, int]]]:
        # Choose columns used in model
        data = data[self.col_res_duration].copy()

        # Change data type
        data[self.res.res] = data[self.res.res].astype(str)
        data[self.item.sku] = data[self.item.sku].astype(str)
        data[self.dmd.duration] = data[self.dmd.duration].astype(int)

        # Group bom route by each plant
        res_duration_by_plant = {}
        for plant in self.plant_list_in_dmd:
            # Filter data contained in each plant
            res_duration = data[data[self.res.plant] == plant].copy()

            # Filter items of each plant involved in demand
            res_duration = res_duration[res_duration[self.item.sku].isin(self.item_list_by_plant[plant])]

            # item -> resource -> duration mapping
            res_grp_duration_map = {}
            for item, group in res_duration.groupby(self.item.sku):
                res_grp_duration_map[item] = group[[self.res.res, self.dmd.duration]]\
                    .set_index(self.res.res)\
                    .to_dict()[self.dmd.duration]

            res_duration_by_plant[plant] = res_grp_duration_map

        return res_duration_by_plant

    def set_job_change(self, data):
        # Change time unit to second
        unit = list(set(data[self.cstr.jc_unit]))[0]

        if unit == 'MIN':
            data[self.cstr.jc_time] = data[self.cstr.jc_time] * 60

        # Filter demand plant
        data = data[data[self.res.plant].isin(self.plant_list_in_dmd)]

        job_change = {}
        for plant, res_grp_df in data.groupby(by=self.res.plant):
            res_grp_to_type = {}
            for res_grp, type_df in res_grp_df.groupby(by=self.res.res_grp):
                type_to_time = {}
                for jc_type, time_df in type_df.groupby(by=self.cstr.jc_type):
                    type_to_time[jc_type] = {(from_res, to_res): time for from_res, to_res, time in zip(
                        time_df[self.cstr.jc_from], time_df[self.cstr.jc_to], time_df[self.cstr.jc_time]
                    )}
                res_grp_to_type[res_grp] = type_to_time
            job_change[plant] = res_grp_to_type

        return job_change

    def match_job_change_time(self, candidate: dict, job_change_master: pd.DataFrame) -> Dict[str, Dict[Hashable, Any]]:
        # Change time unit
        if self.jc_time_uom == 'MIN':
            job_change_master[self.cstr.jc_time] = job_change_master[self.cstr.jc_time] * 60

        job_change_by_plant = {}
        for plant_cd, job_change_cand in candidate.items():
            job_change = pd.merge(
                job_change_cand,
                job_change_master,
                how='left',
                on=[self.res.res_grp, self.cstr.jc_from, self.cstr.jc_to]
            )

            job_change[self.cstr.jc_time] = job_change[self.cstr.jc_time].fillna(0)

            res_grp_job_change = {}
            for res_grp_cd, res_grp_df in job_change.groupby(by=self.res.res_grp):
                from_to_res = {}
                for from_res, to_res, job_change_time in zip(
                        res_grp_df[self.cstr.jc_from], res_grp_df[self.cstr.jc_to], res_grp_df[self.cstr.jc_time]):
                    from_to_res[(from_res, to_res)] = job_change_time
                res_grp_job_change[res_grp_cd] = from_to_res
            job_change_by_plant[plant_cd] = res_grp_job_change

        return job_change_by_plant

    def set_plant_job_change_bak(self, demand: pd.DataFrame, resource: dict) -> Dict[str, Dict[Hashable, Any]]:
        # add brand information
        merged = pd.merge(
            demand,
            resource[self.key.item][[self.item.brand, self.item.sku]],
            on=[self.item.sku],
            how='left'
        )
        # resource group by brand
        res_grp_brand = merged[[self.res.plant, self.res.res_grp, self.item.brand]].drop_duplicates()

        plant_job_change_cand = self.set_job_chage_cand(data=res_grp_brand)

        plant_job_change = self.match_job_change_time(
            candidate=plant_job_change_cand,
            job_change_master=resource['job_change']
        )

        return plant_job_change

    def set_job_chage_cand(self, data) -> Dict[str, pd.DataFrame]:
        job_change_cand_by_plant = {}

        for plant_cd, plant_df in data.groupby(by=self.res.plant):
            job_change_cand = pd.DataFrame()
            for res_grp, res_grp_df in plant_df.groupby(by=self.res.res_grp):
                if len(res_grp_df) > 1:
                    brand_seq_list = list(permutations(res_grp_df[self.item.brand], 2))
                    temp = pd.DataFrame(brand_seq_list, columns=[self.cstr.jc_from, self.cstr.jc_to])
                    temp[self.res.res_grp] = res_grp
                    job_change_cand = pd.concat([job_change_cand, temp])

            job_change_cand_by_plant[plant_cd] = job_change_cand

        return job_change_cand_by_plant
