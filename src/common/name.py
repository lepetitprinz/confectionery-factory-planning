import common.config as config


class Key(object):
    def __init__(self):
        # Dataset
        self.dmd = config.key_dmd
        self.res = config.key_res
        self.item = config.key_item
        self.cstr = config.key_cstr

        # Demand
        self.dmd_list = config.key_dmd_list_by_plant
        self.dmd_item_list = config.key_dmd_item_list_by_plant
        self.dmd_res_grp_list = config.key_dmd_res_grp_list_by_plant

        # Resource
        self.res_nm = config.key_res_nm
        self.res_grp = config.key_res_grp
        self.res_grp_nm = config.key_res_grp_nm
        self.res_duration = config.key_res_duration

        # Constraint
        self.jc = config.key_jc    # Job change
        self.sku_type = config.key_sku_type      # SKU item type
        self.mold_res = config.key_mold_res      # mold resource
        self.mold_capa = config.key_mold_capa    # Mold capacity
        self.mold_cstr = config.key_mold_cstr    # Mold constraint
        self.human_cstr = config.key_human_cstr      # Human constraint
        self.human_capa = config.key_human_capa      # Human capacity
        self.human_usage = config.key_human_usage    # Human usage
        self.sim_prod_cstr = config.key_sim_prod_cstr
        self.res_avail_time = config.key_res_avail_time
        self.item_weight = config.key_item_weight

        # Route
        self.route = config.key_bom_route
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
        self.min_lot = config.col_min_lot
        self.multi_lot = config.col_multi_lot


class Item(object):
    def __init__(self):
        self.brand = config.col_brand
        self.item = config.col_item
        self.sku = config.col_sku
        self.sku_nm = config.col_sku_nm
        self.pkg = config.col_pkg
        self.flavor = config.col_flavor
        self.item_type = config.col_item_type
        self.weight = config.col_weight
        self.weight_uom = config.col_weight_uom


class Route(object):
    def __init__(self):
        self.qty_rate = config.col_qty_rate
        self.time_uom = config.col_time_uom
        self.half_item = config.col_half_item
        self.lead_time = config.col_lead_time


class Constraint(object):
    def __init__(self):
        # Job change
        self.jc_to = config.col_job_change_to
        self.jc_from = config.col_job_change_from
        self.jc_type = config.col_job_change_type
        self.jc_time = config.col_job_change_time
        self.jc_unit = config.col_job_change_unit

        # Human Capacity
        self.floor = config.col_floor
        self.m_capa = config.col_man_capa
        self.w_capa = config.col_woman_capa

        # Mold Capacity
        self.mold_capa = config.col_mold_capa
        self.mold_uom = config.col_mold_uom
        self.mold_res = config.col_mold_res
        self.mold_use_rate = config.col_mold_use_rate


class Post(object):
    def __init__(self):
        # Version
        self.fp_seq = config.fp_seq
        self.fp_key = config.fp_key
        self.fp_version = config.fp_version

        self.eng_item = config.col_eng_item
        self.item_type = config.col_item_type

        # Capacity
        # Resource capacity
        self.res_jc_capa = config.col_res_jc_capa
        self.res_use_capa = config.col_res_use_capa
        self.res_avail_capa = config.col_res_avail_capa
        self.res_unavail_capa = config.col_res_unavail_capa

        # Human capacity
        self.tot_m_capa = config.col_tot_m_capa
        self.tot_w_capa = config.col_tot_w_capa
        self.use_m_capa = config.col_use_m_capa
        self.use_w_capa = config.col_use_w_capa
        self.avail_m_capa = config.col_avail_m_capa
        self.avail_w_capa = config.col_avail_w_capa

        # Time
        self.to_time = config.col_to_time
        self.from_time = config.col_from_time
        self.date = config.col_date
        self.to_yymmdd = config.col_to_yymmdd
        self.from_yymmdd = config.col_from_yymmdd
        self.time_idx = config.col_time_index_type
