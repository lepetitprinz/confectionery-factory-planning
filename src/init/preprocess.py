import common.util as util
import common.config as config
from common.name import Key, Demand, Item, Route, Resource, Constraint

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Hashable, Union
from itertools import permutations


class Preprocess(object):
    # Time configuration
    time_uom = config.time_uom
    work_day = config.work_day
    time_multiple = {'DAY': 86400, 'HOUR': 3600, 'MIN': 60}

    def __init__(self, cstr_cfg: dict, version):
        self._version = version
        self._cstr_cfg = cstr_cfg

        # Name instance attribute
        self._key = Key()
        self._dmd = Demand()
        self._res = Resource()
        self._item = Item()
        self._route = Route()
        self._cstr = Constraint()

        # Plant instance attribute
        self._item_list = {}
        self._res_grp_list = []
        self._dmd_plant_list = []

        # Column usage instance attribute
        self._col_dmd = [self._dmd.dmd, self._item.sku, self._res.res_grp, self._dmd.qty, self._dmd.due_date]
        self._col_res_grp = [self._res.plant, self._res.res_grp, self._res.res, self._res.res_nm]
        self._col_res_duration = [self._res.plant, self._item.sku, self._res.res, self._dmd.duration]
        self._col_res_avail_time = [self._res.plant, self._res.res] + \
                                   [self._res.res_capa + str(i + 1) for i in range(self.work_day)]
        self._col_res_grp_job_change = [self._res.plant, self._res.res_grp, self._cstr.jc_from, self._cstr.jc_to,
                                        self._cstr.jc_type, self._cstr.jc_time, self._cstr.jc_unit]

        # Time UOM instance attribute
        self.jc_time_uom = 'MIN'

    def preprocess(self, data):
        ######################################
        # Demand / Resource / Route
        ######################################
        # Demand
        demand = data[self._key.dmd]
        dmd_prep = self._set_dmd_info(data=demand)

        # Resource
        resource = data[self._key.res]
        constraint = data[self._key.cstr]
        res_prep = self._set_res_info(resource=resource, constraint=constraint)

        # Route
        route = data[self._key.route]
        route_prep = self._set_route_info(data=route[self._key.route], res_dur=resource[self._key.res_duration])

        ######################################
        # Constraint (Option)
        ######################################
        # Resource available time (Resource capacity constraint)
        res_avail_time = None
        if self._cstr_cfg['apply_res_available_time']:
            res_avail_time = self._set_res_avail_time(data=constraint[self._key.res_avail_time])

        # Job change
        job_change, sku_type = (None, None)
        if self._cstr_cfg['apply_job_change']:
            sku_type = self._set_sku_type_map(data=resource[self._key.item])
            job_change = self._set_job_change(data=constraint[self._key.jc])

        # Simultaneous production constraint
        sim_prod_cstr = None
        if self._cstr_cfg['apply_sim_prod_cstr']:
            sim_prod_cstr = self._set_sim_prod_cstr(data=constraint[self._key.sim_prod_cstr])

        # Human resource constraint
        human_cstr = None
        if self._cstr_cfg['apply_human_capacity']:
            human_cstr = self._set_human_cstr(
                capacity=constraint[self._key.human_capa],
                usage=constraint[self._key.human_usage]
            )

        # Mold capacity constraint
        mold_cstr = None
        if self._cstr_cfg['apply_mold_capacity']:
            mold_cstr = self._set_mold_cstr(data=constraint[self._key.mold_cstr])

        # Preprocessing result
        prep_data = {
            self._key.dmd: dmd_prep,
            self._key.res: res_prep,
            self._key.route: route_prep,
            self._key.cstr: {
                self._key.jc: job_change,
                self._key.sku_type: sku_type,
                self._key.human_cstr: human_cstr,
                self._key.sim_prod_cstr: sim_prod_cstr,
                self._key.res_avail_time: res_avail_time,
                self._key.mold_cstr: mold_cstr
            },
        }

        return prep_data

    def _set_route_info(self, data: pd.DataFrame, res_dur: pd.DataFrame) -> Dict:
        # Preprocess the bom route & resource duration
        data[self._route.lead_time] = [
            int(lt * self.time_multiple[uom]) for lt, uom in zip(
                data[self._route.lead_time], data[self._route.time_uom])
        ]
        data = data.drop(columns=[self._route.time_uom])

        # Route item & rate
        route_item, route_rate = self._set_route_item_rate(data=data)

        # Route resource
        res_dur = res_dur[res_dur[self._item.sku].isin(data['item_halb_cd'].unique())].copy()
        route_res = self._set_route_res(data=res_dur)

        route = {
            self._key.route_res: route_res,
            self._key.route_item: route_item,
            self._key.route_rate: route_rate
        }

        return route

    def _set_route_item_rate(self, data: pd.DataFrame):
        route_item = {}
        route_rate = {}
        for plant, plant_df in data.groupby(by=self._res.plant):
            item_to_half_item = {}
            item_to_half_item_rate = {}
            for item, item_df in plant_df.groupby(by=self._item.sku):
                item_to_half_item[item] = [half_item for half_item in item_df['item_halb_cd']]
                item_to_half_item_rate[item] = {half: (rate, lt) for half, rate, lt in zip(
                    item_df['item_halb_cd'], item_df['qty_rate'], item_df['lead_time']
                )}
            route_item[plant] = item_to_half_item
            route_rate[plant] = item_to_half_item_rate

        return route_item, route_rate

    def _set_route_res(self, data: pd.DataFrame):
        route_res = {}
        for plant, plant_df in data.groupby(by=self._res.plant):
            item_to_res_dur = {}
            for item, item_df in plant_df.groupby(by=self._item.sku):
                item_to_res_dur[item] = [(res, dur) for res, dur in zip(
                    item_df[self._res.res], item_df[self._dmd.duration]
                )]
            route_res[plant] = item_to_res_dur

        return route_res

    def _set_human_cstr(self, capacity: pd.DataFrame, usage: pd.DataFrame) -> Dict[str, Any]:
        human_resource = {
            self._key.human_usage: self._set_human_usage(data=usage),
            self._key.human_capa: self._set_human_capacity(data=capacity)
        }

        return human_resource

    def _set_human_usage(self, data: pd.DataFrame):
        # Change data type
        data[self._res.res_grp] = data[self._res.res_grp].astype(str)
        data[self._item.pkg] = data[self._item.pkg].astype(str)

        data = data[data[self._res.plant].isin(self._dmd_plant_list)].copy()

        human_usage = {}
        for plant, plant_df in data.groupby(by=self._res.plant):
            floor_res_grp = {}
            for floor, floor_df in plant_df.groupby(by=self._cstr.floor):
                res_grp_item = {}
                for res_grp, res_grp_df in floor_df.groupby(by=self._res.res_grp):
                    item_capa = {}
                    for item, pkg, m_capa, w_capa in zip(
                            floor_df[self._item.item], floor_df[self._item.pkg],
                            floor_df[self._cstr.m_capa], floor_df[self._cstr.w_capa]):
                        item_capa[(item, pkg)] = (m_capa, w_capa)
                    res_grp_item[res_grp] = item_capa
                floor_res_grp[floor] = res_grp_item
            human_usage[plant] = floor_res_grp

        return human_usage

    def _set_human_capacity(self, data: pd.DataFrame) -> Dict[str, Dict[str, Tuple[int, int]]]:
        # Change data type
        data['yy'] = data['yy'].astype(str)

        data = data[data[self._res.plant].isin(self._dmd_plant_list)].copy()

        # Choose current week capacity
        # ToDo: Temporal code
        data['fp_version'] = 'FP_' + data['yy'] + data['week']
        data = data[data['fp_version'] == self._version.fp_version[:self._version.fp_version.index('.')]]

        human_capacity = {}
        for plant, plant_df in data.groupby(by=self._res.plant):
            floor_capa = {}
            for floor, m_capa, w_capa in zip(
                    plant_df[self._cstr.floor], plant_df[self._cstr.m_capa], plant_df[self._cstr.w_capa]
            ):
                floor_capa[floor] = (m_capa, w_capa)
            human_capacity[plant] = floor_capa

        return human_capacity

    def _set_mold_cstr(self, data: pd.DataFrame) -> dict:
        # Get plant list of demand list
        data = data[data[self._res.plant].isin(self._dmd_plant_list)].copy()

        # Change data type
        data[self._res.res_grp] = data[self._res.res_grp].astype(str)
        data[self._item.pkg] = data[self._item.pkg].astype(str)

        mold_capacity = {}
        for plant, plant_df in data.groupby(by=self._res.plant):
            res_grp_brand = {}
            for reg_grp, res_grp_df in plant_df.groupby(by=self._res.res_grp):
                brand_pkg = {}
                for brand, brand_df in res_grp_df.groupby(by=self._item.brand):
                    pkg_capa = {}
                    for pkg, capa in zip(brand_df[self._item.pkg], brand_df[self._res.res_capa]):
                        pkg_capa[pkg] = capa
                    brand_pkg[brand] = pkg_capa
                res_grp_brand[reg_grp] = brand_pkg
            mold_capacity[plant] = res_grp_brand

        return mold_capacity

    def _set_sim_prod_cstr(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        # Get plant list of demand list
        data = data[data[self._res.plant].isin(self._dmd_plant_list)].copy()

        # Change data type
        data[self._res.res_grp] = data[self._res.res_grp].astype(str)
        data[self._item.pkg + '1'] = data[self._item.pkg + '1'].astype(str)
        data[self._item.pkg + '2'] = data[self._item.pkg + '2'].astype(str)

        # Simultaneous type : necessary
        necessary = data[data['sim_type'] == 'NEC']
        nece_sim_prod_cstr = self._make_sim_prod_cstr_map(data=necessary)

        # Simultaneous type : impossible
        impo_sim_prod_cstr = None
        impossible = data[data['sim_type'] == 'IMP']
        if len(impossible) > 0:
            impo_sim_prod_cstr = self._make_sim_prod_cstr_map(data=impossible)

        plant_sim_prod_cstr = {
            'necessary': nece_sim_prod_cstr,
            'impossible': impo_sim_prod_cstr
        }

        return plant_sim_prod_cstr

    def _make_sim_prod_cstr_map(self, data: pd.DataFrame) -> dict:
        sim_prod_cstr = {}
        for plant, plant_df in data.groupby(by=self._res.plant):
            res_grp_brand = {}
            for res_grp, res_grp_df in plant_df.groupby(by=self._res.res_grp):
                brand_pkg = {}
                for brand, brand_df in res_grp_df.groupby(by=self._item.brand):
                    pkg1_pkg2 = {}
                    for pkg1, pkg2 in zip(brand_df[self._item.pkg + '1'], brand_df[self._item.pkg + '2']):
                        pkg1_pkg2[pkg1] = pkg2
                    brand_pkg[brand] = pkg1_pkg2
                res_grp_brand[res_grp] = brand_pkg
            sim_prod_cstr[plant] = res_grp_brand

        return sim_prod_cstr

    def _set_sku_type_map(self, data: pd.DataFrame) -> Dict[str, Tuple[str, str, str]]:
        sku_to_type = {item_cd: (brand, flavor, pkg) for item_cd, brand, flavor, pkg in
                       zip(data[self._item.sku], data[self._item.brand], data[self._item.flavor], data[self._item.pkg])}

        return sku_to_type

    def _set_dmd_info(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        # Quantity constraint
        if self._cstr_cfg['apply_prod_qty_multiple']:
            data = util.change_dmd_qty(data=data, method='multiple')

        # Get plant list of demand list
        dmd_plant_list = list(set(data[self._res.plant]))

        # Calculate the due date
        self._calc_due_date(data=data)

        # Change data type
        data[self._item.sku] = data[self._item.sku].astype(str)
        data[self._res.res_grp] = data[self._res.res_grp].astype(str)
        data[self._dmd.qty] = np.ceil(data[self._dmd.qty]).astype(int)    # Ceiling quantity

        # Group demand by each plant
        dmd_list_by_plant, dmd_item_list_by_plant, dmd_res_grp_list_by_plant = {}, {}, {}
        for plant in dmd_plant_list:
            # Filter data by each plant
            dmd = data[data[self._res.plant] == plant]

            # Convert form of demand dataset
            dmd_list_by_plant[plant] = self._convert_dmd_form(data=dmd)

            # Get resource group only contained in demand
            dmd_res_grp_list_by_plant[plant] = list(set(dmd[self._res.res_grp].values))

            # All of demand item list by plant
            dmd_item_list_by_plant[plant] = list(set(dmd[self._item.sku].values))

        self._dmd_plant_list = dmd_plant_list
        self._item_list = dmd_item_list_by_plant
        self._res_grp_list = dmd_res_grp_list_by_plant

        dmd_prep = {
            self._key.dmd_list: dmd_list_by_plant,
            self._key.dmd_item_list: dmd_item_list_by_plant,
        }

        return dmd_prep

    def _calc_due_date(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self._dmd.due_date] = data[self._dmd.due_date] * 24 * 60 * 60
        data[self._dmd.due_date] = data[self._dmd.due_date].astype(int)

        return data

    def _convert_dmd_form(self, data: pd.DataFrame) -> List[Tuple[Any]]:
        data_use = data[self._col_dmd].copy()
        data_tuple = [tuple(row) for row in data_use.to_numpy()]

        return data_tuple

    def _set_res_info(self, resource: dict, constraint: dict) -> Dict[str, Dict[str, Any]]:
        res_grp = resource[self._key.res_grp]

        # Filter resources whose available time does not exist
        if self._cstr_cfg['apply_res_available_time']:
            res_grp = self._filter_na_time(res_grp=res_grp, res_avail_time=constraint[self._key.res_avail_time])

        # Resource group <-> Resource information
        plant_res_grp, plant_res_nm = self._set_res_grp(data=res_grp)

        # Resource group naming
        plant_res_grp_nm = self.set_res_grp_name(data=resource[self._key.res_grp_nm])

        # Resource group duration
        plant_res_duration = self.set_res_duration(data=resource[self._key.res_duration])

        res_prep = {
            self._key.res_nm: plant_res_nm,
            self._key.res_grp: plant_res_grp,
            self._key.res_grp_nm: plant_res_grp_nm,
            self._key.res_duration: plant_res_duration,
        }

        return res_prep

    def _filter_na_time(self, res_grp: pd.DataFrame, res_avail_time: pd.DataFrame) -> pd.DataFrame:
        compare_col = [self._res.plant, self._res.res]
        res_avail_time = res_avail_time[compare_col].copy()
        res_avail_time = res_avail_time.drop_duplicates()

        res_grp = pd.merge(res_grp, res_avail_time, how='inner', on=compare_col)

        return res_grp

    def _filter_duplicated_capacity(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop_duplicates()
        data_filtered = pd.DataFrame()
        for res_grp, res_grp_df in data.groupby(by=self._res.res_grp):
            for res_cd, res_df in res_grp_df.groupby(by=self._res.res):
                if len(res_df) == 1:
                    data_filtered = data_filtered.append(res_df)
                else:
                    data_filtered = data_filtered.append(res_df.max(), ignore_index=True)

        data_filtered = data_filtered.reset_index(drop=True)

        return data_filtered

    def _set_res_avail_time(self, data: pd.DataFrame):
        # Choose columns used in model
        data = data[self._col_res_avail_time].copy()

        # choose capacity in several item case
        # data = self.filter_duplicate_capacity(data=data)

        # Change the data type
        data[self._res.res] = data[self._res.res].astype(str)

        capa_col_list = []
        for i in range(self.work_day):
            capa_col = self._res.res_capa + str(i + 1)
            capa_col_list.append(capa_col)
            data[capa_col] = data[capa_col].astype(int)

        # Choose plants of demand list
        data = data[data[self._res.plant].isin(self._dmd_plant_list)].copy()

        res_avail_time_by_plant = {}
        for plant, res_df in data.groupby(by=self._res.plant):
            res_to_capa = {}
            for res, capa_df in res_df.groupby(by=self._res.res):
                res_to_capa[res] = capa_df[capa_col_list].values.tolist()[0]
            res_avail_time_by_plant[plant] = res_to_capa

        return res_avail_time_by_plant

    def _set_res_grp(self, data: pd.DataFrame) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, str]]]:
        # Choose columns used in model
        data = data[self._col_res_grp].copy()

        # Change the data type
        data[self._res.res] = data[self._res.res].astype(str)
        data[self._res.res_grp] = data[self._res.res_grp].astype(str)

        # Choose plants of demand list
        data = data[data[self._res.plant].isin(self._dmd_plant_list)].copy()

        # Group resource by each plant
        res_grp_by_plant = {}
        res_nm_by_plant = {}
        for plant in self._dmd_plant_list:
            # Filter data by each plant
            res_grp_df = data[data[self._res.plant] == plant]
            res_grp_df = res_grp_df[res_grp_df[self._res.res_grp].isin(self._res_grp_list[plant])]

            # Resource group -> resource list
            res_grp_to_res = {}
            for res_grp, group in res_grp_df.groupby(self._res.res_grp):
                res_grp_to_res[res_grp] = group[self._res.res].tolist()

            res_grp_by_plant[plant] = res_grp_to_res

            res_nm_by_plant[plant] = {str(res_cd): res_nm for res_cd, res_nm in
                                      zip(res_grp_df[self._res.res], res_grp_df[self._res.res_nm])}

        return res_grp_by_plant, res_nm_by_plant

    def set_res_grp_name(self, data: pd.DataFrame) -> Dict[Hashable, dict]:
        res_grp_nm_by_plant = {}
        for plant_cd, plant_df in data.groupby(by=self._res.plant):
            res_grp_nm_by_plant[plant_cd] = {res_grp_cd: res_grp_nm for res_grp_cd, res_grp_nm
                                             in zip(plant_df[self._res.res_grp], plant_df[self._res.res_grp_nm])}

        return res_grp_nm_by_plant

    def set_res_duration(self, data: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, int]]]:
        # Choose columns used in model
        data = data[self._col_res_duration].copy()

        # Change data type
        data[self._res.res] = data[self._res.res].astype(str)
        data[self._item.sku] = data[self._item.sku].astype(str)
        data[self._dmd.duration] = data[self._dmd.duration].astype(int)

        # Group bom route by each plant
        res_duration_by_plant = {}
        for plant in self._dmd_plant_list:
            # Filter data contained in each plant
            res_duration = data[data[self._res.plant] == plant].copy()

            # item -> resource -> duration mapping
            res_grp_duration_map = {}
            for item, group in res_duration.groupby(self._item.sku):
                res_grp_duration_map[item] = group[[self._res.res, self._dmd.duration]]\
                    .set_index(self._res.res)\
                    .to_dict()[self._dmd.duration]

            res_duration_by_plant[plant] = res_grp_duration_map

        return res_duration_by_plant

    def _set_job_change(self, data: pd.DataFrame):
        # Change time unit to second
        unit = list(set(data[self._cstr.jc_unit]))[0]

        if unit == 'MIN':
            data[self._cstr.jc_time] = data[self._cstr.jc_time] * 60

        # Filter demand plant
        data = data[data[self._res.plant].isin(self._dmd_plant_list)]

        job_change = {}
        for plant, res_grp_df in data.groupby(by=self._res.plant):
            res_grp_to_type = {}
            for res_grp, type_df in res_grp_df.groupby(by=self._res.res_grp):
                type_to_time = {}
                for jc_type, time_df in type_df.groupby(by=self._cstr.jc_type):
                    type_to_time[jc_type] = {(from_res, to_res): time for from_res, to_res, time in zip(
                        time_df[self._cstr.jc_from], time_df[self._cstr.jc_to], time_df[self._cstr.jc_time]
                    )}
                res_grp_to_type[res_grp] = type_to_time
            job_change[plant] = res_grp_to_type

        return job_change

    def match_job_change_time(self, candidate: dict, job_change_master: pd.DataFrame) -> Dict[str, Dict[Hashable, Any]]:
        # Change time unit
        if self.jc_time_uom == 'MIN':
            job_change_master[self._cstr.jc_time] = job_change_master[self._cstr.jc_time] * 60

        job_change_by_plant = {}
        for plant_cd, job_change_cand in candidate.items():
            job_change = pd.merge(
                job_change_cand,
                job_change_master,
                how='left',
                on=[self._res.res_grp, self._cstr.jc_from, self._cstr.jc_to]
            )

            job_change[self._cstr.jc_time] = job_change[self._cstr.jc_time].fillna(0)

            res_grp_job_change = {}
            for res_grp_cd, res_grp_df in job_change.groupby(by=self._res.res_grp):
                from_to_res = {}
                for from_res, to_res, job_change_time in zip(
                        res_grp_df[self._cstr.jc_from], res_grp_df[self._cstr.jc_to], res_grp_df[self._cstr.jc_time]):
                    from_to_res[(from_res, to_res)] = job_change_time
                res_grp_job_change[res_grp_cd] = from_to_res
            job_change_by_plant[plant_cd] = res_grp_job_change

        return job_change_by_plant

    def set_job_change_candidate(self, data: pd.DataFrame) -> Dict[Hashable, Union[pd.DataFrame, pd.Series]]:
        job_change_cand_by_plant = {}

        for plant_cd, plant_df in data.groupby(by=self._res.plant):
            job_change_candidate = pd.DataFrame()
            for res_grp, res_grp_df in plant_df.groupby(by=self._res.res_grp):
                if len(res_grp_df) > 1:
                    brand_seq_list = list(permutations(res_grp_df[self._item.brand], 2))
                    temp = pd.DataFrame(brand_seq_list, columns=[self._cstr.jc_from, self._cstr.jc_to])
                    temp[self._res.res_grp] = res_grp
                    job_change_candidate = pd.concat([job_change_candidate, temp])

            job_change_cand_by_plant[plant_cd] = job_change_candidate

        return job_change_cand_by_plant
