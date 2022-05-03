import common.util as util
import common.config as config
from dao.DataIO import DataIO
from common.sql import Query
from init.init import Init
from init.load import DataLoad
from init.preprocess import Preprocess
from model.model import OptSeqModel
from Post.process import Process


class Pipeline(object):
    ############################################
    # Data dictionary key configuration
    ############################################
    key_dmd = config.key_dmd
    key_res = config.key_res

    # Demand
    key_dmd_list_by_plant = config.key_dmd_list_by_plant
    key_res_grp = config.key_res_grp  # Resource group code

    def __init__(self, step_cfg: dict, exec_cfg: dict, cstr_cfg: dict, except_cfg: dict,
                 base_path: dict, fp_seq: str, fp_num='01'):
        self.io = DataIO()
        self.query = Query()
        self.step_cfg = step_cfg    # Step configuration
        self.exec_cfg = exec_cfg    # Execution configuration
        self.cstr_cfg = cstr_cfg
        self.except_cfg = except_cfg

        # Path instance attribute
        self.base_path = base_path
        self.path = {}

        # Plant information instance attribute
        self.fp_num = fp_num
        self.fp_seq = fp_seq
        self.fp_version = ''
        self.plant_start_time = None

    def run(self):
        # =================================================================== #
        # 1. Initialization dataset
        # =================================================================== #
        # Instantiate init class
        print("Step 0: Initialize engine information.")
        init = Init(
            io=self.io,
            query=self.query,
            default_path=self.base_path,
            fp_num=self.fp_num,
            fp_seq=self.fp_seq
        )
        init.run()

        # Set initialized object
        self.path = init.pipeline_path
        self.fp_version = init.fp_version
        # self.fp_version = 'FP_2022W16.01'
        self.plant_start_time = init.plant_start_day

        print("Initialization is finished.\n")

        # =================================================================== #
        # 2. Load dataset
        # =================================================================== #
        # Instantiate load class
        load = DataLoad(
            io=self.io,
            query=self.query,
            fp_version=self.fp_version
        )

        data = None
        if self.step_cfg['cls_load']:
            data = load.load_data()

            # Save the master & demand information
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=data, path=self.path['load_data'], data_type='binary')

        # =================================================================== #
        # 3. Data preprocessing
        # =================================================================== #
        prep_data = None
        if self.step_cfg['cls_prep']:
            print("Step: Data Preprocessing\n")

            if not self.step_cfg['cls_load']:
                data = self.io.load_object(path=self.path['load_data'], data_type='binary')

            # Instantiate data preprocessing class
            prep = Preprocess(cstr_cfg=self.cstr_cfg, fp_version=self.fp_version)

            # Preprocess demand / resource / constraint data
            prep_data = prep.preprocess(data=data)

            # Save the preprocessed demand
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=prep_data, path=self.path['prep_data'], data_type='binary')

            print("Data Preprocessing is finished.\n")

        # =================================================================== #
        # Model
        # =================================================================== #
        plant_model = {}
        if self.step_cfg['cls_model']:
            print("Step: Modeling & Optimization\n")

            if not self.step_cfg['cls_prep']:
                prep_data = self.io.load_object(path=self.path['prep_data'], data_type='binary')

            # Modeling by each plant
            for plant in prep_data[self.key_dmd][self.key_dmd_list_by_plant]:
                print(f" - Set the OtpSeq model: {plant}")

                # Instantiate OptSeq class
                opt_seq = OptSeqModel(
                    exec_cfg=self.exec_cfg,
                    cstr_cfg=self.cstr_cfg,
                    except_cfg=self.except_cfg,
                    plant=plant,
                    plant_data=prep_data,
                    fp_seq=self.fp_seq,
                    fp_version=self.fp_version
                )

                # Initialize the each model of plant
                model, rm_act_list = opt_seq.init(
                    plant=plant,
                    dmd_list=prep_data[self.key_dmd][self.key_dmd_list_by_plant][plant],
                    res_grp_dict=prep_data[self.key_res][self.key_res_grp][plant]
                )

                act_mode_name_map = opt_seq.make_act_mode_map(model=model)

                plant_model[plant] = {
                    'model': model,
                    'act_mode_name': act_mode_name_map,
                    'rm_act_list': rm_act_list
                }

                # Check and fix the model setting
                model = opt_seq.check_fix_model_init_set(model=model)

                # Optimization
                print(f" - Optimize the OtpSeq model: {plant}")
                opt_seq.optimize(model=model)

                # Save original result
                opt_seq.save_org_result()

            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=plant_model, path=self.path['model'], data_type='binary')

            print("Modeling & Optimization is finished.\n")
        # =================================================================== #
        # Post Process
        # =================================================================== #
        if self.step_cfg['cls_pp']:
            print("Step: Post Process\n")

            if not self.step_cfg['cls_load']:
                data = self.io.load_object(path=self.path['load_data'], data_type='binary')

                # if self.cstr_cfg['apply_prod_qty_multiple']:
                #     data[self.key_dmd] = util.change_dmd_qty(data=data[self.key_dmd], method='multiple')

            if not self.step_cfg['cls_prep']:
                prep_data = self.io.load_object(path=self.path['prep_data'], data_type='binary')

            if not self.step_cfg['cls_model']:
                plant_model = self.io.load_object(path=self.path['model'], data_type='binary')

            # Post Process after optimization
            for plant in prep_data[self.key_dmd][self.key_dmd_list_by_plant]:
                if len(plant_model[plant]['model'].act) > 0:
                    pp = Process(
                        io=self.io,
                        query=self.query,
                        exec_cfg=self.exec_cfg,
                        cstr_cfg=self.cstr_cfg,
                        fp_version=self.fp_version,
                        fp_seq=self.fp_seq,
                        plant=plant,
                        plant_start_time=self.plant_start_time,
                        data=data,
                        prep_data=prep_data,
                        model_init=plant_model[plant],
                    )
                    pp.run()

                # print("Post Process is finished.")
