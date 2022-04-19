import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from deployment.Pipeline import Pipeline

fp_seq = '01'
fp_serial = '003'

base_path = {
    'root': os.path.join('/', 'opt', 'DF', 'fp'),
    'save': os.path.join('/', 'opt', 'DF', 'fp', 'result', 'pipeline')
}

step_cfg = {
    'cls_load': True,
    'cls_prep': True,
    'cls_model': True,
    'cls_pp': False,
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

cstr_cfg = {
    'apply_job_change': False,    # Model
    'apply_res_available_time': False,
    'apply_human_capacity': False,
}

pipeline = Pipeline(
    step_cfg=step_cfg,
    exec_cfg=exec_cfg,
    cstr_cfg=cstr_cfg,
    except_cfg=except_cfg,
    base_path=base_path,
    fp_seq=fp_seq,
    fp_serial=fp_serial
)
pipeline.run()
