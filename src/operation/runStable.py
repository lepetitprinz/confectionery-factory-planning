from deployment.pipeline import Pipeline

import os
import datetime

fp_num = '01'
fp_seq = '1'

base_path = {
    'root': os.path.join('..', '..'),
    'save': os.path.join('..', '..', 'result', 'pipeline')
}

cfg = {
    'exec': {
        'save_step_yn': True,
        'save_db_yn': True,
        'save_graph_yn': True,
        'verbose': False,
    },
    'step': {
        'cls_load': False,
        'cls_cns': False,
        'cls_prep': False,
        'cls_model': False,
        'cls_pp': True,
        'cls_save': True,
    },
    'cstr': {
        'apply_res_available_time': True,    # Resource Capacity
        'apply_job_change': True,            # Job Change
        'apply_prod_qty_multiple': True,     # Product Quantity Multiple
        'apply_human_capacity': True,        # Human Capacity
        'apply_sim_prod_cstr': False,         # Simultaneous Production Constraint
        'apply_mold_capa_cstr': False,       # Mold Capacity Constraint
    },
    'except': {
        'miss_duration': 'remove'    # add / remove
    }
}

# Check start time
print("Optimization Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

pipeline = Pipeline(
    cfg=cfg,
    base_path=base_path,
    fp_num=fp_num,
    fp_seq=fp_seq
)
pipeline.run()

# Check start time
print("Optimization End: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
