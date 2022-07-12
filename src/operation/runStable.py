from deployment.pipeline import Pipeline

import os
import datetime

fp_num = '02'
fp_seq = '1'

base_path = {
    'root': os.path.join('..', '..'),
    'save': os.path.join('..', '..', 'result', 'pipeline')
}

cfg = {
    'exec': {
        'save_step_yn': False,
        'save_db_yn': False,
        'save_graph_yn': False,
        'verbose': False,
    },
    'step': {
        'cls_load': True,
        'cls_cns': False,
        'cls_prep': True,
        'cls_model': True,
        'cls_pp': True,
    },
    'cstr': {
        'apply_job_change': True,            # Job Change
        'apply_min_lot_size': True,          # Minimum lot size
        'apply_multi_lot_size': True,        # Multiple lot size
        'apply_human_capacity': False,        # Human Capacity
        'apply_sim_prod_cstr': True,         # Simultaneous Production Constraint
        'apply_mold_capa_cstr': True,        # Mold Capacity Constraint
        'apply_res_available_time': True,    # Resource Capacity
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
