import common.config as config


class Key(object):
    def __init__(self):
        # Dataset
        self.dmd = config.key_dmd
        self.res = config.key_res
        self.item = config.key_item
        self.cstr = config.key_cstr

        # Demand
        self.dmd_list_by_plant = config.key_dmd_list_by_plant
        self.dmd_item_list_by_plant = config.key_dmd_item_list_by_plant
        self.dmd_res_grp_list_by_plant = config.key_dmd_res_grp_list_by_plant

        # Resource
        self.res_nm = config.key_res_nm
        self.res_grp = config.key_res_grp
        self.res_grp_nm = config.key_res_grp_nm
        self.res_duration = config.key_res_duration

        # Constraint
        self.jc = config.key_jc
        self.sku_type = config.key_sku_type
        self.sim_prod_cstr = config.key_sim_prod_cstr
        self.res_avail_time = config.key_res_avail_time
        self.human_res = config.key_human_res
        self.human_capa = config.key_human_capa
        self.human_usage = config.key_human_usage

        # Route
        self.route = config.key_bom_route    # BOM Route
        self.route_res = config.key_route_res
        self.route_item = config.key_route_item
        self.route_rate = config.key_route_rate


class Demand(object):
    def __init__(self):
        self.dmd = config.col_dmd
        self.qty = config.col_qty
        self.prod_qty = config.col_prod_qty
        self.due_date = config.col_due_date
        self.duration = config.col_duration
        self.start_time = config.col_start_time
        self.end_time = config.col_end_time


class Resource(object):
    def __init__(self):
        self.plant = config.col_plant
        self.res = config.col_res
        self.res_nm = config.col_res_nm
        self.res_grp = config.col_res_grp
        self.res_grp_nm = config.col_res_grp_nm
        self.res_type = config.col_res_type
        self.res_capa = config.col_res_capa
        self.capa_unit = config.col_capa_unit


class Item(object):
    def __init__(self):
        self.brand = config.col_brand
        self.item = config.col_item
        self.sku = config.col_sku
        self.sku_nm = config.col_sku_nm
        self.pkg = config.col_pkg
        self.flavor = config.col_flavor


class Route(object):
    def __init__(self):
        self.qty_rate = config.col_qty_rate
        self.time_uom = config.col_time_uom
        self.half_item = config.col_half_item
        self.lead_time = config.col_lead_time


class Constraint(object):
    def __init__(self):
        # Job change
        self.jc_from = config.col_job_change_from
        self.jc_to = config.col_job_change_to
        self.jc_type = config.col_job_change_type
        self.jc_time = config.col_job_change_time
        self.jc_unit = config.col_job_change_unit

        # Human Capacity
        self.floor = config.col_floor
        self.m_capa = config.col_man_capa
        self.w_capa = config.col_woman_capa

        # Simultaneous Production
