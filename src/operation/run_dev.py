from deployment.PipelineDev import Pipeline

import os
import datetime


fp_num = '01'
fp_seq = '005'
# 001 : Default (k130) (complete)
# 002 : Resource capacity constraint (k130) (complete)
# 003 : Resource capacity constraint (k120) (complete)
# 004 : Job change constraint (k130)
# 005 : Simultaneous production constraint (K130)
base_path = {
    'root': os.path.join('..', '..'),
    'save': os.path.join('..', '..', 'result', 'pipeline')
}

step_cfg = {
    'cls_load': False,
    'cls_prep': False,
    'cls_model': True,
    'cls_pp': True,
}

exec_cfg = {
    'save_step_yn': True,
    'save_db_yn': False,
    'save_graph_yn': True,
    'verbose': False,
}

except_cfg = {
    'miss_duration': 'remove'    # add / remove
}

cstr_cfg = {
    'apply_res_available_time': True,
    'apply_job_change': False,
    'apply_prod_qty_multiple': False,
    'apply_sim_prod_cstr': True,
    'apply_human_capacity': False,
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