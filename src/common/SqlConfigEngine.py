
class SqlConfigEngine(object):
    #################################
    # Master & Common Code Dataset
    #################################
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

    @staticmethod
    def sql_item_master(**kwargs):
        sql = f"""
            SELECT ITEM.ITEM_ATTR03_CD AS ITEM_ATTR03_CD
                 , ENG_ITEM_CD AS ITEM_CD
                 , FP_ITEM.ITEM_NM AS ITEM_NM
                 , FP_ITEM.ITEM_TYPE_CD AS ITEM_TYPE_CD
              FROM (
                    SELECT *
                      FROM M4E_I401080
                     WHERE ITEM_TYPE_CD IN ('FERT', 'HAWA')
                       AND FP_VRSN_ID = '{kwargs['fp_version']}'
                  ) FP_ITEM
               LEFT OUTER JOIN M4S_I002040 ITEM
                ON FP_ITEM.ITEM_CD = ITEM.ITEM_CD
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
        """
        return sql

    #################################
    # Factory Planning Dataset
    #################################
    @staticmethod
    def sql_demand(**kwargs):
        sql = f"""
            SELECT FP_KEY AS DMD_ID
                 , DP_KEY
                 , PLANT_CD
                 , ENG_ITEM_CD AS ITEM_CD
                 , RES_CD AS RES_GRP_CD
                 , TIME_INDEX AS DUE_DATE
                 , CEILING(REQ_FP_QTY) AS QTY
              FROM M4E_I401060
             WHERE 1=1
               AND FP_VRSN_ID = '{kwargs['fp_version']}'
             --  AND RES_CD IN ('X247', 'X267')
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
             WHERE 1=1
               AND FP_VRSN_ID = '{kwargs['fp_version']}'
        """
        return sql

    @staticmethod
    def sql_item_res_duration(**kwargs):
        sql = f"""
            SELECT PLANT_CD
                 , RES_CD
                 , ROUTE_CD AS ITEM_CD
                 , ROUND(60 * CAPA_USE_RATE, 0) AS DURATION
              FROM M4E_I401120
             WHERE CAPA_USE_RATE IS NOT NULL
               AND FP_VRSN_ID = '{kwargs['fp_version']}'
        """
        return sql

    @staticmethod
    def sql_fp_seq_list(**kwargs):
        sql = f"""
            SELECT FP_VRSN_SEQ
              FROM M4E_O402122
             WHERE FP_VRSN_ID = '{kwargs['fp_vrsn_id']}'
             GROUP BY FP_VRSN_SEQ     
        """
        return sql

