from deployment.Pipeline import Pipeline

import os


default_path = {
    'root': os.path.join('..', '..'),
    'save': os.path.join('..', '..', 'result', 'pipeline')
}

step_cfg = {
    'cls_load': False,
    'cls_prep': True,
    'cls_model': False,
    'cls_pp': False,
}

exec_cfg = {
    'save_step_yn': True,
    'save_db_yn': False,
}

pipeline = Pipeline(
    step_cfg=step_cfg,
    exec_cfg=exec_cfg,
    default_path=default_path,
)
pipeline.run()
