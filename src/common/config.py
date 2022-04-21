# Database Configuration (Operation)
RDMS = 'mssql+pymssql'
HOST = '10.109.2.143'      # Ops Database IP address
# HOST = '10.109.6.62'     # Dev Database IP address
DATABASE = 'BISCM'         # Database name
PORT = '1433'
USER = 'matrix'            # User name
PASSWORD = 'Diam0nd123!'   # User password

# Column Configuration
col_dmd = 'dmd_id'
col_plant = 'plant_cd'
col_res = 'res_cd'
col_res_nm = 'res_nm'
col_res_grp = 'res_grp_cd'
col_res_grp_nm = 'res_grp_nm'
col_res_map = 'res_map_cd'
col_res_type = 'res_type_cd'
col_capacity = 'capacity'
col_capa_unit = 'capa_unit_cd'
col_duration = 'duration'
col_due_date = 'due_date'
col_brand = 'item_attr03_cd'
col_sku = 'item_cd'
col_qty = 'qty'
col_job_change_from = 'from_res_cd'
col_job_change_to = 'to_res_cd'
col_job_change_time = 'working_time'

# OptSeq Model configuration
time_limit = 60 * 1  # 60*60*6
make_span = False
optput_flag = True
max_iteration = 10**20
report_interval = 10**20
back_truck = 1000
