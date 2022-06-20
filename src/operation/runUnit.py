from deployment.pipelineUnit import Pipeline

import os
import datetime

fp_num = '01'
fp_seq = 'test'

base_path = {
    'root': os.path.join('..', '..'),
    'save': os.path.join('..', '..', 'result', 'pipeline')
}

cfg = {
    'exec': {
        'save_step_yn': True,    # Save each pipeline step
        'save_db_yn': False,      # Save result on database
        'save_graph_yn': False,   # Save gantt graph
        'verbose': False,
    },
    'step': {
        'cls_load': True,     # Load class
        'cls_prep': True,     # Preprocessing class
        'cls_model': True,     # Model class
        'cls_pp': False,        # Post processing class
        'cls_save': False,     # Save result class
    },
    'cstr': {
        'apply_job_change': False,          # Job Change
        'apply_min_lot_size': False,
        'apply_multi_lot_size': False,      # Multi lot size
        'apply_human_capacity': False,      # Human Capacity
        'apply_sim_prod_cstr': False,       # Simultaneous Production Constraint
        'apply_res_available_time': True,   # Resource Capacity
        'apply_mold_capa_cstr': True        # Mold Capacity Constraint
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
