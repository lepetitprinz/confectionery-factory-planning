
class Query(object):
    #################################
    # Master & Common Code Dataset
    #################################
    # Factory planning engine version data
    @staticmethod
    def sql_fp_eng_version():
        sql = """
            SELECT FP_VRSN_ID AS FP_VERSION
                 , FP_VRSN_SEQ AS FP_SEQ
                 , FROM_YYMMDD AS START_DAY
              FROM M4E_I401010
        """
        return sql

    # Item master data (engine)
    @staticmethod
    def sql_item_master(**kwargs):
        sql = f"""
            SELECT ENG_ITEM_CD AS ITEM_CD
                 , ITEM_NM
                 , ITEM_ATTR03_CD
                 , ITEM_ATTR04_CD
                 , ITEM_TYPE_CD
                 , ITEM_ATTR29_CD AS FLAVOR
                 , PKG_CTGRI_SUB_CD AS PKG
                 , ITEM_ATTR17_CD AS WEIGHT
                 , ITEM_ATTR19_CD AS WEIGHT_UOM
              FROM M4E_I401080
             WHERE FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
        """
        return sql

    # BOM route data
    @staticmethod
    def sql_bom_route(**kwargs):
        sql = f"""
            SELECT PLANT_CD
                 , ITEM_CD
                 , ITEM_HALB_CD
                 , IN_RATE / OUT_RATE AS QTY_RATE
                 , MFG_LT AS LEAD_TIME
                 , MFG_LT_UOM AS TIME_UOM
              FROM M4E_I401300
             WHERE FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
        """
        return sql

    # Resource group (code/name)
    @staticmethod
    def sql_res_grp_nm():
        sql = """
            SELECT PLANT_CD
                 , RES_GRP_CD
                 , RES_GRP_NM
              FROM M4S_I305100
             WHERE USE_YN = 'Y'
        """
        return sql

    # Calendar data
    @staticmethod
    def sql_calendar():
        sql = f"""
            SELECT YYMMDD
                 , YY
                 , YYMM
                 , WEEK
                 , START_WEEK_DAY
              FROM M4S_I002030
        """
        return sql

    #################################
    # Factory Planning Dataset
    #################################
    # Demand data
    @staticmethod
    def sql_demand(**kwargs):
        sql = f"""
            SELECT DMD_ID
                 , DP_KEY
                 , PLANT_CD
                 , DMD.ITEM_CD
                 , RES_GRP_CD
                 , DUE_DATE
                 , QTY
              FROM (
                    SELECT FP_KEY              AS DMD_ID
                         , DP_KEY
                         , PLANT_CD
                         , ENG_ITEM_CD         AS ITEM_CD
                         , RES_CD              AS RES_GRP_CD
                         , TIME_INDEX          AS DUE_DATE
                         , CEILING(REQ_FP_QTY) AS QTY
                      FROM M4E_I401060
                     WHERE FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
                       AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
                       AND REQ_FP_QTY > 0
                       AND USE_YN = 'Y'
                   ) DMD
             INNER JOIN (
                         SELECT ITEM_CD
                              , ITEM_ATTR01_CD
                           FROM VIEW_I002040
                          WHERE USE_YN = 'Y'
                            AND DEL_YN = 'N'
                        ) ITEM
                ON DMD.ITEM_CD = ITEM.ITEM_CD
             WHERE ITEM.ITEM_ATTR01_CD = 'P1'  -- Exception
        """
        return sql

    # Resource & Halb capacity data
    @staticmethod
    def sql_res_grp(**kwargs):
        sql = f"""
            SELECT PLANT_CD
                 , RES_GRP_CD
                 , RES_CD
                 , RES_NM
                 , RES_CAPA_VAL AS CAPACITY
                 , RES_TYPE_CD
                 , CAPA_UNIT_CD
                 , HALB_CAPA_VAL AS MOLD_CAPA
                 , HALB_CAPA_UOM_CD AS MOLD_UOM
              FROM M4E_I401100
             WHERE FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
        """
        return sql

    # Resource rate of capacity usage data
    @staticmethod
    def sql_item_res_dur(**kwargs):
        sql = f"""
            SELECT PLANT_CD
                 , RES_CD
                 , ROUTE_CD AS ITEM_CD
                 , CEILING(60 * CAPA_USE_RATE) AS DURATION
                 , MIN_LOT_VAL AS MIN_LOT_SIZE
                 , MULTI_LOT_VAL AS MULTI_LOT_SIZE
                 , HALB_RES_CD AS MOLD_RES_CD
                 , CONSM_RATE AS MOLD_USE_RATE
              FROM M4E_I401120
             WHERE CAPA_USE_RATE IS NOT NULL
               AND FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
               AND CAPA_USE_RATE > 0
        """
        return sql

    # Factory planning sequence list
    @staticmethod
    def sql_fp_seq_list(**kwargs):
        sql = f"""
            SELECT FP_VRSN_SEQ
              FROM M4E_O402130
             WHERE FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
             GROUP BY FP_VRSN_SEQ     
        """
        return sql

    #################################
    # Constraint
    #################################
    # Resource available time
    @staticmethod
    def sql_res_avail_time(**kwargs):
        sql = f"""
            SELECT PLANT_CD
                 , RES_GRP_CD
                 , RES_CD
                 , ISNULL(CAPA01_D_VAL, 0) AS CAPACITY1_D
                 , ISNULL(CAPA01_N_VAL, 0) AS CAPACITY1_N
                 , ISNULL(CAPA02_D_VAL, 0) AS CAPACITY2_D
                 , ISNULL(CAPA02_N_VAL, 0) AS CAPACITY2_N
                 , ISNULL(CAPA03_D_VAL, 0) AS CAPACITY3_D
                 , ISNULL(CAPA03_N_VAL, 0) AS CAPACITY3_N
                 , ISNULL(CAPA04_D_VAL, 0) AS CAPACITY4_D
                 , ISNULL(CAPA04_N_VAL, 0) AS CAPACITY4_N
                 , ISNULL(CAPA05_D_VAL, 0) AS CAPACITY5_D
                 , ISNULL(CAPA05_N_VAL, 0) AS CAPACITY5_N
                 , ISNULL(CAPA06_D_VAL, 0) AS CAPACITY6_D
                 , ISNULL(CAPA06_N_VAL, 0) AS CAPACITY6_N
                 , ISNULL(CAPA07_D_VAL, 0) AS CAPACITY7_D
                 , ISNULL(CAPA07_N_VAL, 0) AS CAPACITY7_N
              FROM M4E_I401140
             WHERE FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
               AND RES_CTIG_CD <> 'RES_CTGI_11'
               AND RES_GRP_CD IS NOT NULL
        """
        return sql

    # Job change data
    @staticmethod
    def sql_job_change(**kwargs):
        sql = f"""
            SELECT PLANT_CD
                 , RES_GRP_CD
                 , FROM_RES_CD
                 , TO_RES_CD
                 , JOB_CHANGE_TYPE AS JC_TYPE
                 , WORKING_TIME AS JC_TIME
                 , UNIT_CD AS JC_UNIT
              FROM M4E_I401271
             WHERE FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
        """
        return sql

    # Human resource usage data
    @staticmethod
    def sql_res_human_usage(**kwargs):
        sql = f"""
            SELECT PLANT_CD
                 , RES_GRP_CD
                 , ITEM_ATTR04_CD
                 , PKG_CTGRI_SUB_CD AS PKG
                 , ATTR01_VAL AS FLOOR_CD
                 , MP_M_VAL AS M_VAL
                 , MP_W_VAL AS W_VAL
              FROM M4E_I401291
             WHERE FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
               AND LEFT(YYMMDD, 4) = '{kwargs['yy']}'
               AND WEEK = '{kwargs['week']}'
        """
        return sql

    # Human resource capacity data
    @staticmethod
    def sql_res_human_capacity(**kwargs):
        sql = f"""
            SELECT PLANT_CD
                 , LEFT(PLAN_YYMM, 4) AS YY
                 , PLAN_WEEK AS WEEK
                 , PLANT_F_VAL AS FLOOR_CD
                 , MP_M_VAL AS M_VAL
                 , MP_W_VAL AS W_VAL
              FROM M4E_I401290
             WHERE FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
                -- AND LEFT(PLAN_YYMM, 4) = '{kwargs['yy']}'
                -- AND PLAN_WEEK = '{kwargs['week']}'
        """
        return sql

    # Human resource capacity data (weekly)
    @staticmethod
    def sql_res_human_capacity_weekly(**kwargs):
        sql = f"""
            SELECT PLANT_CD
                 , LEFT(PLAN_YYMM, 4) AS YY
                 , PLAN_WEEK AS WEEK
                 , PLANT_F_VAL AS FLOOR_CD
                 , MP_M_VAL AS M_VAL
                 , MP_W_VAL AS W_VAL
              FROM M4E_I401290
             WHERE FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
        """
        return sql

    # simultaneous production data (constraint)
    @staticmethod
    def sql_sim_prod_cstr(**kwargs):
        sql = f"""
            SELECT PLANT_CD
                 , RES_GRP_CD
                 , ITEM_ATTR03_CD
                 , PKG_CTGRI_SUB_CD_1 AS PKG1
                 , PKG_CTGRI_SUB_CD_2 AS PKG2
                 , IS_SIMULTANEOUS AS SIM_TYPE
              FROM M4E_I401280
             WHERE ITEM_ATTR01_CD = 'P1'
               AND FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
        """
        return sql

    # Mold capacity data (constraint)
    @staticmethod
    def sql_mold_capacity(**kwargs):
        sql = f"""
            SELECT PLANT_CD
                 , RES_CD
                 , ISNULL(CAPA01_D_VAL, 0) AS CAPACITY1_D
                 , ISNULL(CAPA01_N_VAL, 0) AS CAPACITY1_N
                 , ISNULL(CAPA02_D_VAL, 0) AS CAPACITY2_D
                 , ISNULL(CAPA02_N_VAL, 0) AS CAPACITY2_N
                 , ISNULL(CAPA03_D_VAL, 0) AS CAPACITY3_D
                 , ISNULL(CAPA03_N_VAL, 0) AS CAPACITY3_N
                 , ISNULL(CAPA04_D_VAL, 0) AS CAPACITY4_D
                 , ISNULL(CAPA04_N_VAL, 0) AS CAPACITY4_N
                 , ISNULL(CAPA05_D_VAL, 0) AS CAPACITY5_D
                 , ISNULL(CAPA05_N_VAL, 0) AS CAPACITY5_N
                 , ISNULL(CAPA06_D_VAL, 0) AS CAPACITY6_D
                 , ISNULL(CAPA06_N_VAL, 0) AS CAPACITY6_N
                 , ISNULL(CAPA07_D_VAL, 0) AS CAPACITY7_D
                 , ISNULL(CAPA07_N_VAL, 0) AS CAPACITY7_N
              FROM M4E_I401140
             WHERE RES_CTIG_CD = 'RES_CTGI_11'
               AND FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_vrsn_seq']}'
        """
        return sql

    #################################
    # Delete SQL
    #################################
    # Delete required production quantity result
    @staticmethod
    def del_dmd_result(**kwargs):
        sql = f"""
            DELETE 
              FROM M4E_O402010
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_seq']}'
               AND PLANT_CD = '{kwargs['plant_cd']}'
        """
        return sql

    # Delete resource (day/night)
    @staticmethod
    def del_res_day_night_qty(**kwargs):
        sql = f"""
            DELETE 
              FROM M4E_O402130
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_seq']}'
               AND PLANT_CD = '{kwargs['plant_cd']}'
        """
        return sql

    @staticmethod
    def del_res_day_night_dmd_qty(**kwargs):
        sql = f"""
            Delete
              FROM M4E_O402131
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_seq']}'
               AND PLANT_CD = '{kwargs['plant_cd']}'
        """
        return sql

    @staticmethod
    def del_res_status_result(**kwargs):
        sql = f"""
            DELETE 
              FROM M4E_O402050
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_seq']}'
               AND PLANT_CD = '{kwargs['plant_cd']}'
        """
        return sql

    @staticmethod
    def del_gantt_result(**kwargs):
        sql = f"""
            DELETE 
              FROM M4E_O402140
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_seq']}'
               AND PLANT_CD = '{kwargs['plant_cd']}'
        """
        return sql

    @staticmethod
    def del_human_capa_profile(**kwargs):
        sql = f"""
            DELETE
              FROM M4E_O402150
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_seq']}'
               AND PLANT_CD = '{kwargs['plant_cd']}'
        """
        return sql

    @staticmethod
    def del_human_capa_profile_dtl(**kwargs):
        sql = f"""
            DELETE
              FROM M4E_O402151
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_seq']}'
               AND PLANT_CD = '{kwargs['plant_cd']}'
        """
        return sql
