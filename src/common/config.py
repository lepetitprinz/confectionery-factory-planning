# Database Configuration (Operation)
RDMS = 'mssql+pymssql'
HOST = '10.109.2.143'      # Ops Database IP address
# HOST = '10.109.6.62'     # Dev Database IP address
DATABASE = 'BISCM'         # Database name
PORT = '1433'
USER = 'matrix'            # User name
PASSWORD = 'Diam0nd123!'   # User password

############################################
# Data dictionary key configuration
############################################
# Demand
key_dmd = 'demand'         # Demand
key_res = 'resource'       # Resource
key_item = 'item'          # Item master
key_cstr = 'constraint'    # Constraint

# Demand
key_dmd_list_by_plant = 'dmd_list_by_plant'
key_dmd_item_list_by_plant = 'dmd_item_list_by_plant'
key_dmd_res_grp_list_by_plant = 'dmd_res_grp_list_by_plant'

# Resource
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

# Column Configuration
# Column: Demand
col_dmd = 'dmd_id'
col_qty = 'qty'
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
col_flavor = 'flavor'
col_pkg = 'pkg'
col_item = 'item_attr04_cd'
col_sku = 'item_cd'
col_sku_nm = 'item_nm'

# Column: job change
col_job_change_from = 'from_res_cd'
col_job_change_to = 'to_res_cd'
col_job_change_type = 'jc_type'
col_job_change_time = 'jc_time'
col_job_change_unit = 'jc_unit'

# Constraint configuration
prod_qty_multiple = 10

# OptSeq Model configuration
time_limit = 60 * 1  # 60*60*6
make_span = True
optput_flag = True
max_iteration = 10**20
report_interval = 10**20
back_truck = 1000