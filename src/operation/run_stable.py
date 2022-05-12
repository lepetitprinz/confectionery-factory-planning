from deployment.pipelineDev import Pipeline

import os
import datetime

fp_num = '01'
fp_seq = '1'

base_path = {
    'root': os.path.join('..', '..'),
    'save': os.path.join('..', '..', 'result', 'pipeline')
}

# Step configuration
step_cfg = {
    'cls_load': True,
    'cls_cns': False,
    'cls_prep': True,
    'cls_model': False,
    'cls_pp': False,
}

exec_cfg = {
    'save_step_yn': False,
    'save_db_yn': False,
    'save_graph_yn': False,
    'verbose': False,
}

# Constraint configuration
cstr_cfg = {
    'apply_res_available_time': True,
    'apply_job_change': True,
    'apply_prod_qty_multiple': True,
    'apply_human_capacity': True,
    'apply_sim_prod_cstr': True,
}

except_cfg = {
    'miss_duration': 'remove'    # add / remove
}

# Check start time
print("Optimization Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

pipeline = Pipeline(
    step_cfg=step_cfg,
    exec_cfg=exec_cfg,
    cstr_cfg=cstr_cfg,
    except_cfg=except_cfg,
    base_path=base_path,
    fp_num=fp_num,
    fp_seq=fp_seq
)
pipeline.run()

# Check start time
print("Optimization End: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))