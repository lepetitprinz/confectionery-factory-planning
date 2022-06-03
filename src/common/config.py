# Database Configuration (Operation)
RDMS = 'mssql+pymssql'
HOST = '10.109.2.143'      # Ops Database IP address
# HOST = '10.109.6.62'     # Dev Database IP address
DATABASE = 'BISCM'         # Database name
PORT = '1433'
USER = 'matrix'            # User name
PASSWORD = 'Diam0nd123!'   # User password

#################################
# Factory Planning configuration
#################################
project_cd = 'ENT001'
apply_plant = ['K110', 'K120', 'K130', 'K140', 'K170']

time_uom = 'sec'
work_day = 5

####################################
# Data dictionary key configuration
####################################
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
key_sim_prod_cstr = 'sim_prod_cstr'
key_res_avail_time = 'res_avail_time'
key_human_res = 'human_resource'
key_human_capa = 'human_capacity'
key_human_usage = 'human_usage'

# Route
key_bom_route = 'bom_route'
key_route_res = 'route_res'
key_route_item = 'route_item'
key_route_rate = 'route_rate'
########################
# Column Configuration
########################
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
col_res_type = 'res_type_cd'
col_res_capa = 'capacity'
col_capa_unit = 'capa_unit_cd'

# Column: Item
col_brand = 'item_attr03_cd'
col_item = 'item_attr04_cd'
col_sku = 'item_cd'
col_sku_nm = 'item_nm'
col_pkg = 'pkg'
col_flavor = 'flavor'
col_item_type = 'item_type_cd'

# Columns: Route
col_lead_time = 'lead_time'
col_half_item = 'item_halb_cd'
col_qty_rate = 'qty_rate'
col_time_uom = 'time_uom'

#####################
# Column: Constraint
#####################
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

#######################
# Column: Post process
#######################
# Planning version
fp_seq = 'fp_seq'
fp_key = 'fp_key'
fp_version = 'fp_version'

# Item
col_eng_item = 'eng_item_cd'

# User & time
col_to_time = 'to_time'
col_from_time = 'from_time'
col_to_yymmdd = 'to_yymmdd'
col_from_yymmdd = 'from_yymmdd'
col_create_user = 'create_user_cd'
col_time_index_type = 'time_index_type'

# Resource capacity
col_res_jc_capa = 'res_jc_val'
col_res_unavail_capa = 'res_unavail_val'

# Human capacity
col_tot_m_capa = 'man_capa'
col_tot_w_capa = 'woman_capa'
col_use_m_capa = 'used_man_capa'
col_use_w_capa = 'used_woman_capa'
col_avail_m_capa = 'avail_man_capa'
col_avail_w_capa = 'avail_woman_capa'
col_res_use_capa = 'res_used_capa_val'

# Constraint configuration
prod_qty_multiple = 10

# OptSeq Model configuration
time_limit = 60 * 1  # 60*60*6
make_span = True
optput_flag = False
max_iteration = 10**20
report_interval = 10**20
back_truck = 1000
