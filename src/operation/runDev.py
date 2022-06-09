from deployment.pipelineDev import Pipeline

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
        'save_step_yn': False,    # Save each pipeline step
        'save_db_yn': False,      # Save result on database
        'save_graph_yn': False,   # Save gantt graph
        'verbose': False,
    },
    'step': {
        'cls_load': False,    # Load class
        'cls_cns': False,     # Consistency Check class
        'cls_prep': False,    # Preprocessing class
        'cls_model': False,   # Model class
        'cls_pp': True,       # Post processing class
        'cls_save': False,    # Save result class
    },
    'cstr': {
        'apply_job_change': True,           # Job Change
        'apply_human_capacity': False,      # Human Capacity
        'apply_sim_prod_cstr': False,       # Simultaneous Production Constraint
        'apply_prod_qty_multiple': True,    # Product Quantity Multiple
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
