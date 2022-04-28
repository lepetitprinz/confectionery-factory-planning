from deployment.PipelineDev import Pipeline

import os
import datetime

fp_num = '01'
fp_seq = '1'
# 001 : Default (K130) (complete) - 2022W16
# 002 : Resource capacity constraint (K130) (complete) - 2022W16
# 003 : Resource capacity constraint (K120) (complete) - 2022W16
# 004 : Job change constraint (K130) - 2022W16
# 005 : Simultaneous production constraint (K130) - 2022W16
# 1 : res_avail / jc (K130) - 2022W17

base_path = {
    'root': os.path.join('..', '..'),
    'save': os.path.join('..', '..', 'result', 'pipeline')
}

# Step configuration
step_cfg = {
    'cls_load': False,
    'cls_prep': False,
    'cls_model': False,
    'cls_pp': True,
}

# Constraint configuration
cstr_cfg = {
    'apply_res_available_time': True,
    'apply_job_change': True,
    'apply_prod_qty_multiple': True,
    'apply_human_capacity': True,
    'apply_sim_prod_cstr': False,    # Temp
}

exec_cfg = {
    'save_step_yn': True,
    'save_db_yn': False,
    'save_graph_yn': False,
    'verbose': False,
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