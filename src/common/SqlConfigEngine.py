
class SqlConfigEngine(object):
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
            SELECT FP_VRSN_ID
                 , ENG_ITEM_CD AS ITEM_CD
                 , ITEM_NM
                 , ITEM_TYPE_CD
              FROM M4E_I401080
             WHERE 1=1
               AND FP_VRSN_ID = '{kwargs['fp_version']}'
        """
        return sql

    @staticmethod
    def sql_demand(**kwargs):
        sql = f"""
            SELECT FP_VRSN_ID
                 , FP_KEY AS DMD_ID
                 , DP_KEY
                 , PLANT_CD
                 , ENG_ITEM_CD AS ITEM_CD
                 , RES_CD AS RES_GRP_CD
                 , TIME_INDEX AS DUE_DATE
                 , CEILING(REQ_FP_QTY) AS QTY
--                 , 1 as QTY
              FROM M4E_I401060
             WHERE 1=1
               AND FP_VRSN_ID = '{kwargs['fp_version']}'
        """
        return sql

    @staticmethod
    def sql_res_grp(**kwargs):
        sql = f"""
            SELECT FP_VRSN_ID
                 , PLANT_CD
                 , RES_GRP_CD
                 , RES_CD
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
            SELECT FP_VRSN_ID
                 , PLANT_CD
                 , RES_CD AS RES_GRP_CD
                 , ROUTE_CD AS ITEM_CD
                 , ROUND(60 * CAPA_USE_RATE, 0) AS DURATION
              FROM M4E_I401120
             WHERE CAPA_USE_RATE IS NOT NULL
               AND FP_VRSN_ID = '{kwargs['fp_version']}'
        """
        return sql
