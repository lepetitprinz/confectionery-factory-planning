from common.name import Key
from common.sql import Query
from dao.io import DataIO
from unit.initUnit import Init
from unit.loadUnit import DataLoad
from init.preprocess import Preprocess
from unit.modelUnit import OptSeq
from Post.processBak import Process


class Pipeline(object):
    def __init__(self, cfg: dict, base_path: dict, fp_seq: str, fp_num='01'):
        self.io = DataIO()
        self.query = Query()
        self.key = Key()
        self.cfg = cfg

        # Path instance attribute
        self.path = {}
        self.base_path = base_path

        # Plant information instance attribute
        self.fp_seq = fp_seq
        self.fp_num = fp_num
        self.version = None
        self.fp_version = ''

        # Time instance attribute
        self.calendar = None
        self.plant_start_day = None

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
        self.version = init.version
        self.calendar = init.calendar
        self.fp_version = init.fp_version
        self.plant_start_day = init.plant_start_day
        print("Initialization is finished.\n")

        # =================================================================== #
        # 2. Load dataset
        # =================================================================== #
        # Instantiate load class
        load = DataLoad(
            io=self.io,
            query=self.query,
            version=self.version,
        )

        data = None
        if self.cfg['step']['cls_load']:
            data = load.load()

            # Save the master & demand information
            if self.cfg['exec']['save_step_yn']:
                self.io.save_object(data=data, path=self.path['load_data'], data_type='binary')

        # =================================================================== #
        # 4. Data preprocessing
        # =================================================================== #
        prep_data = None
        if self.cfg['step']['cls_prep']:
            print("Step: Data Preprocessing\n")

            if not self.cfg['step']['cls_load']:
                data = self.io.load_object(path=self.path['load_data'], data_type='binary')

            # Instantiate data preprocessing class
            prep = Preprocess(cstr_cfg=self.cfg['cstr'], version=self.version)

            # Preprocess demand / resource / constraint data
            prep_data = prep.preprocess(data=data)

            # Save the preprocessed demand`
            if self.cfg['exec']['save_step_yn']:
                self.io.save_object(data=prep_data, path=self.path['prep_data'], data_type='binary')

            print("Data Preprocessing is finished.\n")

        # =================================================================== #
        # Model
        # =================================================================== #
        plant_model = {}
        if self.cfg['step']['cls_model']:
            print("Step: Modeling & Optimization\n")

            if not self.cfg['step']['cls_prep']:
                prep_data = self.io.load_object(path=self.path['prep_data'], data_type='binary')

            # Model optimization by each plant
            for plant in prep_data[self.key.dmd][self.key.dmd_list]:
                print(f" - Set the OtpSeq model: {plant}")
                # Instantiate OptSeq class
                opt_seq = OptSeq(
                    cfg=self.cfg,
                    plant=plant,
                    plant_data=prep_data,
                    version=self.version,
                )

                # Initialize the each model of plant
                model, rm_act_list = opt_seq.init(
                    plant=plant,
                    dmd_list=prep_data[self.key.dmd][self.key.dmd_list][plant],
                    res_grp_dict=prep_data[self.key.res][self.key.res_grp][plant]
                )

                # Make activity to mode hash map
                act_mode_name_map = opt_seq.make_act_mode_map(model=model)

                plant_model[plant] = {
                    'model': model,
                    'act_mode_name': act_mode_name_map,
                    'rm_act_list': rm_act_list
                }

                # Check and fix the model setting
                model = opt_seq.check_and_fix_model_setting(model=model)

                # Optimization
                print('\n============================================')
                print(f" - Optimize the OtpSeq model: {plant}")
                print('============================================')
                opt_seq.optimize(model=model)

                # Save original result
                opt_seq.save_org_result()

            if self.cfg['exec']['save_step_yn']:
                self.io.save_object(data=plant_model, path=self.path['model'], data_type='binary')

            print("Modeling & Optimization is finished.\n")

        # =================================================================== #
        # Post Process
        # =================================================================== #
        if self.cfg['step']['cls_pp']:
            print("Step: Post Process\n")
            # Load data / preprocessed data / model information
            if not self.cfg['step']['cls_load']:
                data = self.io.load_object(path=self.path['load_data'], data_type='binary')
            if not self.cfg['step']['cls_prep']:
                prep_data = self.io.load_object(path=self.path['prep_data'], data_type='binary')
            if not self.cfg['step']['cls_model']:
                plant_model = self.io.load_object(path=self.path['model'], data_type='binary')

            # Post Process after optimization
            for plant in prep_data[self.key.dmd][self.key.dmd_list]:
                print(f"\nPost process: plant {plant}")
                # Todo: Temporal test
                if plant == 'K130':
                    if len(plant_model[plant]['model'].act) > 0:
                        pp = Process(
                            io=self.io,
                            cfg=self.cfg,
                            query=self.query,
                            version=self.version,
                            plant=plant,
                            plant_start_time=self.plant_start_day,
                            data=data,
                            prep_data=prep_data,
                            model_init=plant_model[plant],
                            calendar=self.calendar
                        )
                        pp.run()

            # Close DB session
            self.io.session.close()

            print("Post Process is finished.")
