#########################################################
# Database Configuration (Operation)
#########################################################
RDMS = 'mssql+pymssql'     # Database
HOST = '10.109.2.143'      # Ops Database IP address
# HOST = '10.109.6.62'     # Dev Database IP address
DATABASE = 'BISCM'         # Database name
PORT = '1433'
USER = 'matrix'            # User name
PASSWORD = 'Diam0nd123!'   # User password

#########################################################
# Factory Planning configuration
#########################################################
project_cd = 'ENT001'
apply_plant = ['K110', 'K120', 'K130', 'K140', 'K170']

work_day = 7
time_uom = 'sec'
time_multiple = 60
# schedule_weeks = 52     # 6 month
schedule_weeks = 104     # 6 month
plant_start_hour = 0
sec_of_day = 86400

#########################################################
# Default setting configuration
#########################################################

# Constraint
inf_val = 10 ** 7 - 1
default_min_lot = 0
default_multi_lot = 1
default_res_duration = 1

#########################################################
# Data dictionary key configuration
#########################################################
# Dataset
key_dmd = 'demand'         # Demand
key_res = 'resource'       # Resource
key_item = 'item'          # Item master
key_cstr = 'constraint'    # Constraint

# Demand
key_dmd_list_by_plant = 'dmd_list_by_plant'
key_dmd_item_list_by_plant = 'dmd_item_list_by_plant'
key_dmd_res_grp_list_by_plant = 'dmd_res_grp_list_by_plant'

# Resource
key_res_nm = 'res_nm'
key_res_grp = 'res_grp'
key_res_grp_nm = 'res_grp_nm'
key_res_duration = 'res_duration'

# Constraint
key_jc = 'job_change'
key_sku_type = 'sku_type'
key_mold_res = 'mold_resource'
key_mold_capa = 'mold_capacity'
key_mold_cstr = 'mold_constraint'
key_human_cstr = 'human_resource'
key_human_capa = 'human_capacity'
key_human_usage = 'human_usage'
key_sim_prod_cstr = 'sim_prod_cstr'
key_res_avail_time = 'res_avail_time'
key_item_weight = 'item_weight'

# Route
key_bom_route = 'bom_route'
key_route_res = 'route_res'
key_route_item = 'route_item'
key_route_rate = 'route_rate'

#########################################################
# Column Configuration
#########################################################
# Column: Planning
fp_seq = 'fp_seq'
fp_key = 'fp_key'
fp_version = 'fp_version'

# Column: Date
col_yy = 'yy'
col_week = 'week'

# Column: Demand
col_dmd = 'dmd_id'
col_qty = 'qty'
col_prod_qty = 'prod_qty'
col_due_date = 'due_date'
col_duration = 'duration'
col_start_time = 'starttime'
col_end_time = 'endtime'

# Column: Resource
col_plant = 'plant_cd'
col_res = 'res_cd'
col_res_nm = 'res_nm'
col_res_grp = 'res_grp_cd'
col_res_grp_nm = 'res_grp_nm'
col_res_capa = 'capacity'
col_res_type = 'res_type_cd'
col_capa_unit = 'capa_unit_cd'
col_min_lot = 'min_lot_size'
col_multi_lot = 'multi_lot_size'

# Column: Item
col_brand = 'item_attr03_cd'
col_item = 'item_attr04_cd'
col_sku = 'item_cd'
col_sku_nm = 'item_nm'
col_pkg = 'pkg'
col_flavor = 'flavor'
col_item_type = 'item_type_cd'
col_weight = 'weight'
col_weight_uom = 'weight_uom'

# Columns: Route
col_lead_time = 'lead_time'
col_half_item = 'item_halb_cd'
col_qty_rate = 'qty_rate'
col_time_uom = 'time_uom'

#########################################################
# Column: Constraint
#########################################################
# Job change
col_job_change_from = 'from_res_cd'
col_job_change_to = 'to_res_cd'
col_job_change_type = 'jc_type'
col_job_change_time = 'jc_time'
col_job_change_unit = 'jc_unit'

# Human capacity
col_floor = 'floor_cd'
col_man_capa = 'm_val'
col_woman_capa = 'w_val'

# Mold capacity
col_mold_capa = 'mold_capa'
col_mold_uom = 'mold_uom'
col_mold_res = 'mold_res_cd'
col_mold_use_rate = 'mold_use_rate'

#########################################################
# Column: Post process
#########################################################
col_kind = 'kind'

# Item
col_eng_item = 'eng_item_cd'

# User & time
col_date = 'yymmdd'
col_to_time = 'to_time'
col_from_time = 'from_time'
col_to_yymmdd = 'to_yymmdd'
col_from_yymmdd = 'from_yymmdd'
col_create_user = 'create_user_cd'
col_time_index_type = 'time_index_type'

# Resource capacity
col_res_jc_capa = 'res_jc_val'
col_res_use_capa = 'res_used_capa_val'
col_res_avail_capa = 'res_avail_val'
col_res_unavail_capa = 'res_unavail_val'

# Human capacity
col_tot_m_capa = 'man_capa'
col_tot_w_capa = 'woman_capa'
col_use_m_capa = 'used_man_capa'
col_use_w_capa = 'used_woman_capa'
col_avail_m_capa = 'avail_man_capa'
col_avail_w_capa = 'avail_woman_capa'

#########################################################
# OptSeq Model
#########################################################
make_span = True
optput_flag = True
max_iteration = 10 ** 20
time_limit = {
    'K110': 90,
    'K120': 90,
    'K130': 90,
    'K140': 90,
    'K150': 90,
    'K170': 90,
}
