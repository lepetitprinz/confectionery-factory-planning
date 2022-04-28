
class SqlConfig(object):
    #################################
    # Master & Common Code Dataset
    #################################
    @staticmethod
    def sql_item_master(**kwargs):
        sql = f"""
            SELECT ENG_ITEM_CD AS ITEM_CD
                 , ITEM_NM
                 , ITEM_ATTR03_CD
                 , ITEM_ATTR29_CD AS FLAVOR
                 , PKG_CTGRI_SUB_CD AS PKG
              FROM M4E_I401080
             WHERE ITEM_TYPE_CD IN ('FERT', 'HAWA')
               AND FP_VRSN_ID = '{kwargs['fp_version']}'

        """
        return sql

    @staticmethod
    def sql_res_grp_nm():
        sql = """
            SELECT PLANT_CD
                 , RES_GRP_CD
                 , RES_GRP_NM
              FROM M4S_I305100
             WHERE USE_YN = 'Y'
               AND PLANT_CD IN ('K130')
        """
        return sql

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
                     WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
                       AND REQ_FP_QTY > 0
                       AND PLANT_CD IN ('K130')
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
                 , START_TIME_INDEX
                 , END_TIME_INDEX
              FROM M4E_I401100
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND PLANT_CD IN ('K130')
        """
        return sql

    @staticmethod
    def sql_item_res_dur(**kwargs):
        sql = f"""
            SELECT PLANT_CD
                 , RES_CD
                 , ROUTE_CD AS ITEM_CD
                 , ROUND(60 * CAPA_USE_RATE, 0) AS DURATION
              FROM M4E_I401120
             WHERE CAPA_USE_RATE IS NOT NULL
               AND FP_VRSN_ID = '{kwargs['fp_version']}'
               AND CAPA_USE_RATE > 0
               AND PLANT_CD IN ('K130')
        """
        return sql

    @staticmethod
    def sql_fp_seq_list(**kwargs):
        sql = f"""
            SELECT FP_VRSN_SEQ
              FROM M4E_O402130
             WHERE FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
             GROUP BY FP_VRSN_SEQ     
        """
        return sql

    #################################
    # Constraint
    #################################
    @staticmethod
    def sql_res_avail_time(**kwargs):
        sql = f"""
        SELECT PLANT_CD
             , RES_CD
             , CAPA01_VAL AS CAPACITY1
             , CAPA02_VAL AS CAPACITY2
             , CAPA03_VAL AS CAPACITY3
             , CAPA04_VAL AS CAPACITY4
             , CAPA05_VAL AS CAPACITY5
          FROM M4E_I401140
         WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
           AND PLANT_CD IN ('K130')
        """
        return sql

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
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND PLANT_CD IN ('K130')  
        """
        return sql

    @staticmethod
    def sql_res_human_usage(**kwargs):
        sql = """
            SELECT PLANT_CD
                 , RES_GRP_CD
                 , ITEM_ATTR04_CD
                 , PKG_CTGRI_SUB_CD AS PKG
                 , ATTR01_VAL AS FLOOR
                 , MP_M_VAL
                 , MP_W_VAL
              FROM M4S_I402020
        """
        return sql

    @staticmethod
    def sql_res_human_capacity():
        sql = """
            SELECT PLAN_YYMMDD
                 , PLAN_WEEK
                 , PLANT_CD
                 , PLANT_F_VAL AS FLOOR
                 , MP_M_VAL
                 , MP_W_VAL
              FROM M4S_I402023
        """
        return sql

    #################################
    # Delete SQL
    #################################
    @staticmethod
    def del_dmd_result(**kwargs):
        sql = f"""
            DELETE 
              FROM M4E_O402010
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_seq']}'
        """
        return sql

    @staticmethod
    def del_res_day_night_qty(**kwargs):
        sql = f"""
            DELETE 
              FROM M4E_O402130
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_seq']}'
        """
        return sql

    @staticmethod
    def del_res_status_result(**kwargs):
        sql = f"""
            DELETE 
              FROM M4E_O402050
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_seq']}'
        """
        return sql

    @staticmethod
    def del_gantt_result(**kwargs):
        sql = f"""
            DELETE 
              FROM M4E_O402140
             WHERE FP_VRSN_ID = '{kwargs['fp_version']}'
               AND FP_VRSN_SEQ = '{kwargs['fp_seq']}'
        """
        return sql