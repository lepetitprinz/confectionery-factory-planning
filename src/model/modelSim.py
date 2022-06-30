import pandas as pd

import common.util as util
import common.config as config
from common.name import Key, Item, Demand

import os
from itertools import permutations
from typing import Dict, Tuple, List
from optimize.optseq import Model, Mode, Parameters, Activity, Resource


class OptSeq(object):
    # job change
    job_change_type = ['BRAND_CHANGE', 'FLAVOR_CHANGE', 'STANDARD_CHANGE']

    def __init__(
            self,
            cfg: dict,
            plant: str,
            plant_data: dict,
            version,
    ):
        # Execution instance attribute
        self._exec_cfg = cfg['exec']
        self._cstr_cfg = cfg['cstr']
        self._except_cfg = cfg['except']

        self._plant = plant
        self.fp_version = version.fp_version
        self._fp_name = version.fp_version + '_' + version.fp_seq + '_' + plant

        # Name instance attribute
        self._key = Key()
        self._item = Item()
        self._dmd = Demand()

        # Resource instance attribute
        self._res_grp = plant_data[self._key.res][self._key.res_grp][plant]
        self._res_to_res_grp = {}

        # Item instance attribute
        self._item_mst = plant_data[self._key.item]
        self._sku_to_pkg = {}
        self._sku_to_brand = {}
        self._brand_pkg_sku_map = {}
        self.col_item = [self._item.sku, self._item.brand, self._item.pkg]

        # Resource capacity instance attribute
        self._sec_of_day = 86400
        self._time_multiple = config.time_multiple
        self._schedule_weeks = config.schedule_weeks
        self._plant_start_hour = config.plant_start_hour

        # Route instance attribute
        self._route_res = plant_data[self._key.route][self._key.route_res].get(plant, None)
        self._route_item = plant_data[self._key.route][self._key.route_item].get(plant, None)
        self._route_rate = plant_data[self._key.route][self._key.route_rate].get(plant, None)

        # Path instance attribute
        self._save_path = os.path.join('..', '..', 'result')
        self._output_path = os.path.join('..', 'operation', 'optseq_output.txt')

        # Constraints
        # Resource duration constraint
        self._default_res_duration = config.default_res_duration
        self._item_res_duration = plant_data[self._key.res][self._key.res_duration][plant]

        # Resource available time constraint
        if self._cstr_cfg['apply_res_available_time']:
            self._res_capa_days = plant_data[self._key.cstr][self._key.res_avail_time].get(plant, None)

        # Job change constraint
        if self._cstr_cfg['apply_job_change']:
            self._sku_type = plant_data[self._key.cstr][self._key.sku_type]
            self._job_change = plant_data[self._key.cstr][self._key.jc].get(plant, None)

        # Simultaneous production constraint
        if self._cstr_cfg['apply_sim_prod_cstr']:
            self._sim_prod_cstr_nec = plant_data[self._key.cstr][self._key.sim_prod_cstr]['necessary'].get(plant, None)
            self._sim_prod_cstr_imp = plant_data[self._key.cstr][self._key.sim_prod_cstr]['impossible'].get(plant, None)

    def init(self, plant: str, dmd_list: list, res_grp_dict: dict):
        # Step 0. Preprocessing
        self._preprocess()

        # Step 1. Instantiate the model
        model = Model(name=plant)

        # Step 2. Set resources
        model_res, res_grp_dict = self._set_resource(model=model, res_grp_dict=res_grp_dict)

        # Step 3. Set activities
        model, activity, rm_act_list = self._set_activity(
            model=model,
            dmd_list=dmd_list,
            res_grp_dict=res_grp_dict,
            model_res=model_res
        )

        # Step 4. Set the job change activity (Optional)
        if (self._cstr_cfg['apply_job_change']) and (self._job_change is not None):
            model = self._set_job_change_activity(
                model=model,
                activity=activity,
                model_res=model_res
            )

        model = self._set_model_parameter(model=model)

        return model, rm_act_list

    def _preprocess(self):
        self._set_res_to_res_grp()
        self._set_sku_to_brand_and_pkg()
        self._set_brand_pkg_sku_map()

    def _set_brand_pkg_sku_map(self) -> None:
        item = self._item_mst.copy()
        item = item[self.col_item]

        brand_pkg_sku_map = {}
        for brand, brand_df in item.groupby(by=self._item.brand):
            pkg_sku = {}
            for pkg, pkg_df in brand_df.groupby(by=self._item.pkg):
                for sku in pkg_df[self._item.sku]:
                    if pkg in pkg_sku:
                        pkg_sku[pkg].append(sku)
                    else:
                        pkg_sku[pkg] = [sku]
            brand_pkg_sku_map[brand] = pkg_sku

        self._brand_pkg_sku_map = brand_pkg_sku_map

    def _set_res_to_res_grp(self) -> None:
        res_grp = self._res_grp.copy()
        res_to_res_grp = {}
        for res_grp_cd, res_list in res_grp.items():
            for res_cd in res_list:
                res_to_res_grp[res_cd] = res_grp_cd

        self._res_to_res_grp = res_to_res_grp

    def _set_sku_to_brand_and_pkg(self) -> None:
        item = self._item_mst.copy()
        item = item[self.col_item]

        # Convert data type
        item[self._item.sku] = item[self._item.sku].astype(str)
        item[self._item.pkg] = item[self._item.pkg].astype(str)

        # Make sku to sku information hash map
        sku_to_pkg = {}
        sku_to_brand = {}
        for sku, brand, pkg in zip(item[self._item.sku], item[self._item.brand], item[self._item.pkg]):
            sku_to_pkg[sku] = pkg
            sku_to_brand[sku] = brand

        self._sku_to_pkg = sku_to_pkg
        self._sku_to_brand = sku_to_brand

    @staticmethod
    def _get_res_to_dmd_list(model: Model):
        res_dmd_list = {}
        for activity in model.act:
            for mode in activity.modes:
                resource = mode.name.split('@')[2][:-1]
                act_name = activity.name
                act_name = act_name[act_name.index('[') + 1: act_name.index(']')]
                if resource not in res_dmd_list:
                    res_dmd_list[resource] = [act_name]
                else:
                    res_dmd_list[resource].append(act_name)

        return res_dmd_list

    # Set resource
    def _set_resource(self, model: Model, res_grp_dict) -> Tuple[Dict[str, Dict[str, Resource]], dict]:
        model_res_grp = {}
        res_grp_list = {**res_grp_dict}

        for res_grp, res_list in res_grp_list.items():
            model_res = {}
            for resource in res_list[:]:
                if resource == '10000039':
                    print("")
                # Add available time of each resource
                add_res = model.addResource(name=resource)

                if self._cstr_cfg['apply_res_available_time']:
                    capa_days = self._res_capa_days.get(resource, None)
                    if capa_days:
                        add_res = self._add_res_capacity(res=add_res, capa_days=capa_days)
                        model_res[resource] = add_res
                    else:
                        # Remove resource candidate from resource group
                        model.res.remove(add_res)
                        res_grp_dict[res_grp].remove(resource)

                else:
                    # Add infinite capacity resource
                    add_res = model.addResource(name=resource, capacity={(0, "inf"): 1})
                    model_res[resource] = add_res
                # model_res[resource] = add_res

            if len(model_res) != 0:
                model_res_grp[res_grp] = model_res
            else:
                res_grp_dict.pop(res_grp)

        # if self._cstr_cfg['apply_sim_prod_cstr']:
        #     if self._sim_prod_cstr_imp is not None:
        #         # apply simultaneously impossible
        #         model_res_grp = self._add_virtual_resource(
        #             model=model,
        #             model_res_grp=model_res_grp,
        #         )

        return model_res_grp, res_grp_dict

    def _add_virtual_resource(self, model: Model, model_res_grp: dict, ):
        for res_grp, res_map in self._sim_prod_cstr_imp.items():
            model_res = {}
            for res1, res2 in res_map.items():
                res_name = res1 + '_' + res2
                add_res = model.addResource(name=res_name, capacity={(0, "inf"): 1})
                model_res['virtual'] = {res1: add_res}
                model_res['virtual'].update({res2: add_res})
            model_res_grp[res_grp].update(model_res)

        return model_res_grp

    def _add_res_capacity(self, res: Resource, capa_days: list) -> Resource:
        day = 0
        for day, (day_time, night_time) in enumerate(capa_days * self._schedule_weeks):
            start_time, end_time = util.calc_daily_avail_time(
                day=day,
                day_time=int(day_time * self._time_multiple),
                night_time=int(night_time * self._time_multiple),
            )

            # Add the capacity
            if start_time != end_time:
                res.addCapacity(start_time, end_time, 1)

        # Exception for over demand
        res.addCapacity((day + 1) * self._sec_of_day, 'inf', 1)

        return res

    # Set activity
    def _set_activity(self, model: Model, dmd_list: list, res_grp_dict: dict, model_res: dict) \
            -> Tuple[Model, dict, List[str]]:
        activity = {}
        for dmd_id, item_cd, res_grp_cd, qty, due_date in dmd_list:
            if res_grp_dict.get(res_grp_cd, None):
                # Make the activity naming
                act_name = util.generate_model_name(name_list=[dmd_id, item_cd, res_grp_cd])

                # Define activities
                activity[act_name] = model.addActivity(
                    name=f'Act[{act_name}]',
                    duedate=due_date,  # duedate='inf',
                    weight=1,  # Penalty per unit time when work completion time is rate for delivery
                )

                # Set modes
                act = self._set_mode(
                    act=activity[act_name],
                    dmd_id=dmd_id,
                    item_cd=item_cd,
                    qty=qty,
                    res_list=res_grp_dict[res_grp_cd],
                    model_res=model_res[res_grp_cd]
                )
                activity = self._remove_empty_mode_from_act(activity=activity, act_name=act_name, act=act)

                # Add route items
                if self._route_item is not None:
                    if self._route_item.get(item_cd, None):
                        model, activity = self._add_route_act(
                            model=model,
                            activity=activity,
                            fwd_act=activity[act_name],
                            dmd_id=dmd_id,
                            item_cd=item_cd,
                            qty=qty,
                            due_date=due_date,
                            model_res=model_res,
                        )

        # Simultaneous production constraint
        if self._cstr_cfg['apply_sim_prod_cstr']:
            # Possible production
            model, activity = self._apply_sim_prod_cstr(
                model=model,
                activity=activity,
                all_dmd=dmd_list,
                res_grp_dict=res_grp_dict,
                model_res=model_res
            )

            # Impossible production
            # model, activity = self._check_virtual_res_on_act(
            #     model=model,
            #     activity=activity,
            #     dmd_list=dmd_list
            # )

        model, rm_act_list = self._remove_empty_mode_from_model(model=model)

        return model, activity, rm_act_list

    def _check_virtual_res_on_act(self, model, activity, dmd_list):
        apply_dmd = self._search_sim_prod_impossible_dmd(data=dmd_list)
        if len(apply_dmd) > 0:
            dmd_pair = self._check_imposible_dmd_pair(data=apply_dmd)
            if len(dmd_pair) > 0:
                print("Apply simultaneous production is impossible")
                model, activity = self._add_virtual_res_on_act(
                    model=model,
                    activity=activity,
                    dmd_pair=dmd_pair
                )

        return model, activity

    def _search_sim_prod_impossible_dmd(self, data: list) -> List[List[str]]:
        imp_dmd = []
        for dmd_id, sku, res_grp, qty, due_date in data:
            cstr_brand = self._sim_prod_cstr_imp.get(res_grp, None)
            if cstr_brand is not None:
                brand = self._sku_to_brand[sku]
                cstr_pkg = cstr_brand.get(brand, None)
                if cstr_pkg is not None:
                    pkg = self._sku_to_pkg[sku]
                    if pkg in cstr_pkg:
                        imp_dmd.append([dmd_id, res_grp, sku, brand, pkg])

        return imp_dmd

    def _check_imposible_dmd_pair(self, data):
        dmd_df = pd.DataFrame(
            data,
            columns=[self._dmd.dmd, 'res_grp_cd', self._item.sku, self._item.brand, self._item.pkg]
        )
        pair = []
        for res_grp, res_grp_df, in dmd_df.groupby(by='res_grp_cd'):
            for brand, brand_df in res_grp_df.groupby(by=self._item.brand):
                pkg_list = brand_df[self._item.pkg].unique()
                if len(pkg_list) > 1:
                    for pkg in pkg_list:
                        pkg_dmd = brand_df[brand_df[self._item.pkg] == pkg]
                        non_pkg_dmd = brand_df[~brand_df[self._item.pkg] == pkg]

                        act1 = util.generate_model_name(
                            name_list=[pkg_dmd[self._dmd.dmd], pkg_dmd[self._item.sku], pkg_dmd['res_grp_cd']])
                        act2 = util.generate_model_name(
                            name_list=[non_pkg_dmd[self._dmd.dmd], non_pkg_dmd[self._item.sku],
                                       non_pkg_dmd['res_grp_cd']]
                        )
                        if (act1, act2) not in pair:
                            pair.append((act1, act2))

        return pair

    def _add_virtual_res_on_act(self, model, activity, dmd_pair):
        for act1, act2 in dmd_pair:
            activity1 = activity[act1]
            activity2 = activity[act2]
            sku1, res_grp1 = activity1.split('@')[1:]
            sku2, res_grp2 = activity2.split('@')[1:]

            virtual_res_name = res_grp1 + '@' + self._sku_to_brand[sku1] + '@' + self._sku_to_pkg[sku1] + '_' + \
                               self._sku_to_pkg[sku2]
            virtual_res = model.addResource(name=virtual_res_name, capacity={(0, "inf"): 1})

            for mode in activity1.modes:
                self._add_virtual_res_on_mode(
                    mode=mode,
                    res=virtual_res,
                    duration=mode.duration
                )

            for mode in activity2.modes:
                self._add_virtual_res_on_mode(
                    mode=mode,
                    res=virtual_res,
                    # res=list(mode.requirement.keys())[0][0],
                    duration=mode.duration
                )

        return model, activity

    def _add_virtual_res_on_mode(self, mode, res, duration):
        mode = self._add_resource(
            mode=mode,
            resource=res,
            duration=duration
        )

        return mode

    def _apply_sim_prod_cstr(self, model, activity, all_dmd, res_grp_dict, model_res):
        possible_dmd = self._classify_sim_prod_possible_dmd(data=all_dmd)
        if len(possible_dmd) > 0:
            model, activity = self._add_sim_prod_act(
                model=model,
                activity=activity,
                all_dmd=all_dmd,
                possible_dmd=possible_dmd,
                res_grp_dict=res_grp_dict,
                model_res=model_res)

        return model, activity

    def _add_sim_prod_act(self, model: Model, activity: dict, all_dmd: list, possible_dmd: list, res_grp_dict: dict,
                          model_res: dict):
        for dmd, sku, res_grp, qty, due_date in possible_dmd:
            # Available package of simultaneously making
            brand = self._sku_to_brand[sku]
            sim_pkg = self._sim_prod_cstr_nec[res_grp][brand][self._sku_to_pkg[sku]]

            # Select SKU to be made
            sim_prod_sku = self._select_sku_to_be_made(brand=brand, pkg=sim_pkg)

            # Search SKU if it is in other demand
            if sim_prod_sku is not None:
                sku_in_other_dmd = self._search_sku_in_other_dmd(all_dmd=all_dmd, sku=sim_prod_sku)
                if len(sku_in_other_dmd) > 0:
                    model, activity = self._correct_other_act(
                        model=model,
                        activity=activity,
                        org_dmd=[dmd, sku, res_grp, qty, due_date],
                        sp_dmd=sku_in_other_dmd
                    )
                else:
                    model, activity = self._add_sim_act(
                        model=model,
                        activity=activity,
                        sp_dmd=[dmd, sim_prod_sku, res_grp, qty, due_date],
                        org_sku=sku,
                        model_res=model_res,
                        res_grp_dict=res_grp_dict
                    )

            else:
                # remove simultaneous product
                model = self._remove_sim_prod_imps_activity(
                    model=model,
                    activity=activity,
                    act_name=util.generate_model_name(name_list=[dmd, sku, res_grp]))

        return model, activity

    def _correct_other_act(self, model: Model, activity: dict, org_dmd, sp_dmd):
        org_dmd_id, org_sku, org_res_grp, org_qty, org_due_date = org_dmd

        # Choose sim production activity
        sp_dmd = sorted(sp_dmd, key=lambda x: x[3], reverse=True)[0]
        sp_dmd_id, sp_sku, sp_res_grp, sp_qty, sp_due_date = sp_dmd

        org_act_name = util.generate_model_name(name_list=[org_dmd_id, org_sku, org_res_grp])
        sp_act_name = util.generate_model_name(name_list=[sp_dmd_id, sp_sku, sp_res_grp])

        org_act = activity[org_act_name]
        sp_act = activity[sp_act_name]

        org_res_dur = self._get_mode_res_dur(act=org_act)
        sp_res_dur = self._get_mode_res_dur(act=sp_act)
        duration = max(org_res_dur + sp_res_dur)

        model = self._update_act_res_duration(model=model, act_list=[org_act_name, sp_act_name], duration=duration)

        model = self.add_sim_prod_temporal(
            model=model,
            fp_act=org_act,
            sp_act=sp_act,
        )

        return model, activity

    @staticmethod
    def _update_act_res_duration(model: Model, act_list: list, duration: int):
        for activity in model.act:
            act_name = activity.name
            act_name = act_name[act_name.index('[') + 1: act_name.index(']')]

            if act_name in act_list:
                for mode in activity.modes:
                    mode.duration = duration
                    mode.breakable = {(0, duration): 'inf'}
                    for key, val in mode.requirement.items():
                        mode.requirement[key] = {(0, duration): 1}

        return model

    @staticmethod
    def _get_mode_res(act: Activity):
        res_list = []
        for mode in act.modes:
            resource = mode.name.split('@')[2][:-1]
            res_list.append(resource)

        return res_list

    @staticmethod
    def _get_mode_res_dur(act: Activity):
        res_dur = []
        for mode in act.modes:
            duration = mode.duration
            res_dur.append(duration)

        return res_dur

    @staticmethod
    def _remove_sim_prod_imps_activity(model: Model, activity: dict, act_name):
        model.act.remove(activity[act_name])

        return model

    def _add_sim_act(self, model: Model, activity: dict, sp_dmd: list, org_sku, model_res: dict, res_grp_dict):
        dmd_id, item_cd, res_grp_cd, qty, due_date = sp_dmd
        sp_dmd_id = 'SP' + dmd_id[2:]
        fp_act_name = util.generate_model_name(name_list=[dmd_id, org_sku, res_grp_cd])
        sp_act_name = util.generate_model_name(name_list=[sp_dmd_id, item_cd, res_grp_cd])

        # Add the activity of simultaneous production
        activity[sp_act_name] = model.addActivity(
            name=f'Act[{sp_act_name}]',
            duedate=due_date,
            weight=1
        )

        # remove resource used in original
        org_res_list = self._get_mode_res(act=activity[fp_act_name])

        res_list = res_grp_dict[res_grp_cd]
        if len(org_res_list) > 1:
            model, fix_res, org_dur = self._fix_act_mode_res(model=model, act_name=fp_act_name)
            res_list.remove(fix_res)
        else:
            if org_res_list[0] in res_list:
                res_list.remove(org_res_list[0])
            org_dur = self._get_mode_res_dur(act=activity[fp_act_name])[0]

        activity[sp_act_name] = self._set_mode(
            act=activity[sp_act_name],
            dmd_id=dmd_id,
            item_cd=item_cd,
            qty=qty,
            res_list=res_list,
            model_res=model_res[res_grp_cd],
            sim_mode=True,
            org_dur=org_dur
        )

        model = self.add_sim_prod_temporal(
            model=model,
            fp_act=activity[fp_act_name],
            sp_act=activity[sp_act_name],
        )

        return model, activity

    def _fix_act_mode_res(self, model: Model, act_name):
        for actitivty in model.act:
            if actitivty.name == 'Act[' + act_name + ']':
                pass

        return model

    @staticmethod
    def add_sim_prod_temporal(model: Model, fp_act: Activity, sp_act: Activity) -> Model:
        model.addTemporal(pred=fp_act, succ=sp_act, tempType='SS', delay=0)
        model.addTemporal(pred=sp_act, succ=fp_act, tempType='SS', delay=0)

        return model

    @staticmethod
    def _search_sku_in_other_dmd(all_dmd: list, sku: str):
        sku_in_dmd = []
        for dmd in all_dmd:
            if sku in dmd[1]:
                sku_in_dmd.append(dmd)

        return sku_in_dmd

    def _select_sku_to_be_made(self, brand, pkg):
        sku = None
        pkg_sku_list = self._brand_pkg_sku_map.get(brand, None)
        if pkg_sku_list is not None:
            sku_list = pkg_sku_list.get(pkg, None)
            if sku_list is not None:
                if len(sku_list) > 1:
                    sku = self.compare_curr_stock(sku_list=sku_list)
                elif len(sku_list) == 1:
                    sku = sku_list[0]

        return sku

    @staticmethod
    def compare_curr_stock(sku_list: list):
        # Todo: need to revise
        return sku_list[0]

    def _classify_sim_prod_possible_dmd(self, data: list) -> List[list]:
        apply_dmd = []
        for dmd_id, sku, res_grp, qty, due_date in data:
            cstr_res_grp = self._sim_prod_cstr_nec.get(res_grp, None)
            if cstr_res_grp is not None:
                brand = self._sku_to_brand[sku]
                cstr_brand = cstr_res_grp.get(brand, None)
                if cstr_brand is not None:
                    pkg = self._sku_to_pkg[sku]
                    cstr_pkg = cstr_brand.get(pkg, None)
                    if cstr_pkg is not None:
                        apply_dmd.append([dmd_id, sku, res_grp, qty, due_date])

        return apply_dmd

    def _add_route_act(self, model, activity, fwd_act, dmd_id, item_cd, qty, due_date, model_res):
        half_item_list = self._route_item[item_cd]
        route_rate = self._route_rate[item_cd]
        for half_item in half_item_list:
            # Make the activity naming
            act_name = util.generate_model_name(name_list=[dmd_id, half_item])

            # Define activities
            activity[act_name] = model.addActivity(
                name=f'Act[{act_name}]',
                duedate=due_date,
                weight=1,
            )

            # Set modes
            route_act = self._set_route_mode(
                act=activity[act_name],
                dmd_id=dmd_id,
                item_cd=half_item,
                qty=qty,
                res_list=self._route_res[half_item],
                model_res=model_res,
                route_rate=route_rate[half_item]
            )

            model = self._add_route_temporal(
                model=model,
                route_act=route_act,
                fwd_act=fwd_act,
                delay=route_rate[half_item][1]
            )

        return model, activity

    @staticmethod
    def _add_route_temporal(model: Model, route_act: Activity, fwd_act: Activity, delay):
        model.addTemporal(pred=route_act, succ=fwd_act, tempType='CS', delay=delay)

        return model

    def _set_route_mode(self, act: Activity, dmd_id, item_cd, qty, res_list, model_res, route_rate):
        for resource, duration_per_unit in res_list:
            duration = int(qty * duration_per_unit / route_rate[0])

            # Make each mode (set each available resource)
            mode = Mode(name=f'Mode[{dmd_id}@{item_cd}@{resource}]', duration=duration)

            # Add break for each mode
            mode.addBreak(start=0, finish=duration, maxtime='inf')

            # Add resource for each mode (resource)
            res_grp = self._res_to_res_grp[resource]
            mode = self._add_resource(
                mode=mode,
                resource=model_res[res_grp][resource],
                duration=duration
            )

            # add mode list to activity
            act.addModes(mode)

        return act

    # Set work processing method
    def _set_mode(self, act: Activity, dmd_id: str, item_cd: str, qty: int, res_list: list, model_res: dict,
                  sim_mode=False, org_dur=None):

        for resource in res_list:
            duration = None
            if not sim_mode:
                # Calculate the duration (the working time of the mode)
                duration_per_unit = self._get_duration_per_unit(item_cd=item_cd, res_cd=resource)

                if duration_per_unit is not None:
                    duration = int(qty * duration_per_unit)

                    if duration <= 0:
                        raise ValueError(f"Duration is not positive integer: item: {item_cd} resource: {resource}")
            else:
                duration = org_dur

            if duration is not None:
                # Make each mode (set each available resource)
                mode = Mode(name=f'Mode[{dmd_id}@{item_cd}@{resource}]', duration=duration)

                # Add break for each mode
                mode.addBreak(start=0, finish=duration, maxtime='inf')

                # Add resource for each mode(resource)
                mode = self._add_resource(
                    mode=mode,
                    resource=model_res[resource],
                    duration=duration
                )

                # add mode list to activity
                act.addModes(mode)

        return act

    # Add the specified resource which amount required when executing the mode
    @staticmethod
    def _add_resource(mode: Mode, resource, duration: int, amount=1) -> Mode:
        # requirement : gives the required amount of resources
        mode.addResource(resource=resource, requirement={(0, duration): amount}, rtype=None)

        return mode

    def _get_duration_per_unit(self, item_cd: str, res_cd: str) -> int:
        duration_per_unit = None

        # Calculate duration
        item_res_duration = self._item_res_duration.get(item_cd, None)

        if item_res_duration is None:
            if self._except_cfg['miss_duration'] == 'add':
                duration_per_unit = self._default_res_duration

            if self._exec_cfg['verbose']:
                print(f"Item: {item_cd} does not have any resource duration.")
        else:
            duration_per_unit = item_res_duration.get(res_cd, None)
            if (duration_per_unit is None) or (duration_per_unit == 0):
                if self._except_cfg['miss_duration'] == 'add':
                    duration_per_unit = self._default_res_duration
                else:
                    duration_per_unit = None

                if self._exec_cfg['verbose']:
                    print(f"Item: {item_cd} - {res_cd} does not have duration")

        return duration_per_unit

    def _set_model_parameter(self, model: Model) -> Model:
        # Set parameters
        params = Parameters()
        params.TimeLimit = config.time_limit[self._plant]
        params.Makespan = config.make_span
        params.OutputFlag = config.optput_flag
        params.MaxIteration = config.max_iteration

        model.Params = params

        return model

    @staticmethod
    def _remove_empty_mode_from_act(activity: dict, act_name: str, act: Activity):
        if len(act.modes) > 0:
            activity[act_name] = act
        else:
            activity.pop(act_name)

        return activity

    @staticmethod
    def _remove_empty_mode_from_model(model: Model):
        rm_act_list = []
        for act in model.act[:]:
            if len(act.modes) == 0:
                rm_act_list.append(act.name)
                model.act.remove(act)

        return model, rm_act_list

    def _set_job_change_activity(self, model: Model, activity: dict, model_res: dict):
        # get activity of resource
        act_by_res = self._get_res_to_dmd_list(model=model)

        # Set state
        state_map = self._set_state_map(data=act_by_res)
        model_state = self._set_model_state(model=model, data=act_by_res, state_map=state_map)

        for resource, act_list in act_by_res.items():
            res_mode = self._set_job_change_mode(
                resource=resource,
                act_list=act_list,
                state_map=state_map,
                model_res=model_res[self._res_to_res_grp[resource]],
                model_state=model_state[resource]
            )
            job_change_activity = self._add_job_change_activity(
                model=model,
                mode=res_mode,
                act_list=act_list,
                resource=resource
            )
            model = self._add_job_change_temporal(model, activity, job_change_activity)

        return model

    @staticmethod
    def _add_job_change_temporal(model, activity, job_change_activity):
        for job_change_act in job_change_activity:
            model.addTemporal(job_change_activity[job_change_act], activity[job_change_act], 'CS')
            model.addTemporal(activity[job_change_act], job_change_activity[job_change_act], 'SC')

        return model

    @staticmethod
    def _add_job_change_activity(model: Model, mode: dict, act_list: list, resource: str):
        job_change_activity = {}
        for act in act_list:
            job_change_activity[act] = model.addActivity(name=f'Setup[{act}@{resource}]', autoselect=True)
            for from_act, to_act in mode:
                if act == to_act:
                    job_change_activity[act].addModes(mode[(from_act, to_act)])

        return job_change_activity

    def _set_job_change_mode(self, resource: str, act_list: list, state_map: dict, model_res: dict, model_state: dict):
        # Make act sequence list
        act_seq_list = self._make_act_sequence(resource=resource, act_list=act_list)

        res_mode = {}
        for from_act, to_act in act_seq_list:
            # Get job change time
            job_change_time = self._calc_job_change_time(
                res_grp=self._res_to_res_grp[resource],
                resource=resource,
                from_act=from_act,
                to_act=to_act
            )
            # Set job change mode
            res_mode[(from_act, to_act)] = Mode(
                name=f'Mode_setup[{from_act}|{to_act}|{resource}]',
                duration=job_change_time
            )

            # Set state
            res_mode[(from_act, to_act)].addState(
                state=model_state,
                fromValue=state_map[from_act],
                toValue=state_map[to_act]
            )
            if job_change_time != 0:
                res_mode[(from_act, to_act)].addResource(
                    resource=model_res[resource],
                    requirement={(0, job_change_time): 1}
                )

        return res_mode

    @staticmethod
    def _make_act_sequence(resource: str, act_list: list) -> List[Tuple[str, str]]:
        act_seq_list = list(permutations(act_list, 2))
        act_seq_list += [(resource, act) for act in act_list]

        return act_seq_list

    def _calc_job_change_time(self, res_grp: str, resource: str, from_act: str, to_act: str) -> int:
        if resource == from_act:
            jc_time = 0
        else:
            from_res = self._sku_type.get(from_act.split('@')[1], None)
            to_res = self._sku_type.get(to_act.split('@')[1], None)

            time_list = []
            for i, jc_type in enumerate(self.job_change_type):
                res_grp_jc = self._job_change.get(res_grp, None)
                if res_grp_jc:
                    time_dict = self._job_change[res_grp].get(jc_type, None)
                    if time_dict is not None:
                        if (from_res is None) or (to_res is None):
                            time = 0
                        else:
                            time = time_dict.get((from_res[i], to_res[i]), 0)
                    else:
                        time = 0
                else:
                    time = 0
                time_list.append(time)

            jc_time = max(time_list)

        return jc_time

    @staticmethod
    def _set_state_map(data: dict) -> Dict[str, int]:
        state = {resource: i for i, resource in enumerate(sorted(data))}

        idx = len(data)
        for res_grp, demand_list in data.items():
            for demand in demand_list:
                state[demand] = idx
                idx += 1

        return state

    @staticmethod
    def _set_model_state(model: Model, data: dict, state_map: dict):
        model_state = {}
        for resource in data:
            state = model.addState("state_" + resource)
            state.addValue(time=0, value=state_map[resource])
            model_state[resource] = state

        return model_state

    def check_and_fix_model_setting(self, model: Model):
        # Check the activity
        for act in model.act:
            if len(act.modes) == 0:
                raise ValueError(f"Activity: {act.name} does not have modes.")

        # Check the mode
        if not self._cstr_cfg['apply_job_change']:
            for mode in model.modes:
                if len(mode.requirement) == 0:
                    raise ValueError(f"Mode: {mode.name} does not have resources.")

        # Check the resource
        res_list = set()
        for resource in model.res[:]:
            if resource.capacity == {}:
                model.res.remove(resource)
            res_list.add(resource.name)
            for (from_time, to_time), req in resource.capacity.items():
                if isinstance(from_time, int) + isinstance(from_time, int) + isinstance(from_time, int) != 3:
                    raise TypeError(f"Resource: {resource.name} contains non-int type.")

        # Compare resource & resource in the mode
        act_mode_res_list = set()
        for act in model.act:
            for mode in act.modes:
                if mode.duration != 0:
                    res = list(mode.requirement.keys())[0][0]
                    act_mode_res_list.add(res)

        # if len(act_mode_res_list - res_list) > 0:
        #     raise ValueError(f"Infeasible Setting")
        #
        # elif len(res_list - act_mode_res_list) > 0:
        #     res_filter_list = list(res_list - act_mode_res_list)
        #
        #     for resource in model.res[:]:
        #         if resource.name in res_filter_list:
        #             model.res.remove(resource)

        return model

    @staticmethod
    def make_act_mode_map(model: Model):
        act_mode_map = {}
        for act in model.act:
            if len(act.modes) == 1:
                act_mode_map[act.name] = act.modes[0].name

        return act_mode_map

    # Optimize the model
    @staticmethod
    def optimize(model: Model):
        model.optimize()

    #####################
    # Save
    #####################
    def save_org_result(self) -> None:
        save_dir = os.path.join(self._save_path, 'opt', 'org', self.fp_version)
        util.make_dir(path=save_dir)

        result = open(os.path.join(save_dir, 'result_' + self._fp_name + '.txt'), 'w')
        with open(self._output_path, 'r') as f:
            for line in f:
                result.write(line)

        f.close()
        result.close()
