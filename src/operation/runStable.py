from deployment.pipeline import Pipeline

import os
import datetime

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
)
pipeline.run()

# Check start time
print("Optimization End: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
