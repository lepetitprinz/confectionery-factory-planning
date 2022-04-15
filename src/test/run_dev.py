from deployment.Pipeline import Pipeline

import os

fp_seq = '01'

default_path = {
    'root': os.path.join('..', '..'),
    'save': os.path.join('..', '..', 'result', 'pipeline')
}

step_cfg = {
    'cls_load': True,
    'cls_prep': True,
    'cls_model': True,
    'cls_pp': True,
}

exec_cfg = {
    'save_step_yn': True,
    'save_db_yn': False,
    'save_graph_yn': True,
}

cstr_cfg = {
    'apply_job_change': True,    # Model
    'apply_res_capacity': False,
    'apply_human_capacity': False,
}

pipeline = Pipeline(
    step_cfg=step_cfg,
    exec_cfg=exec_cfg,
    cstr_cfg=cstr_cfg,
    default_path=default_path,
    fp_seq=fp_seq
)
pipeline.run()
