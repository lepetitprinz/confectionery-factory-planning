
class SqlConfig(object):
    @staticmethod
    def sql_calendar():
        sql = """
            SELECT YYMMDD
                 , YY
                 , YYMM
                 , WEEK
                 , START_WEEK_DAY
              FROM M4S_I002030
        """
        return sql

    @staticmethod
    def sql_item_master():
        sql = """
            SELECT -- ENG_ITEM_CD AS ITEM_CD
                   SUBSTRING(ENG_ITEM_CD, 0, CHARINDEX('@', ENG_ITEM_CD, 0)) AS ITEM_CD
                 , ITEM_NM
                 , ITEM_TYPE_CD
              FROM M4E_I401080
             WHERE 1=1
            -- AND FP_VRSN_ID = ''
        """
        return sql

    @staticmethod
    def sql_demand():
        sql = """
            SELECT FP_VRSN_ID
                 , FP_KEY AS DMD_ID
                 , DP_KEY
                 , PLANT_CD
                 , ENG_ITEM_CD AS ITEM_CD
                 -- , SUBSTRING(ITEM_CD, 0, CHARINDEX('@', ENG_ITEM_CD, 0)) AS ITEM_CD
                 , RES_CD AS RES_GRP_CD
                 , TIME_INDEX AS DUE_DATE
                 , CEILING(REQ_FP_QTY) AS QTY
              FROM M4E_I401060
             WHERE 1=1
            -- AND FP_VRSN_ID = ''
        """
        return sql

    @staticmethod
    def sql_res_grp():
        sql = """
            SELECT FP_VRSN_ID
                 --, PLANT_CD
                 , 'K120' AS PLANT_CD
                 --, RES_GRP_CD
                 , SUBSTRING(RES_CD, 1, CHARINDEX('@', RES_CD, 0)-1) AS RES_GRP_CD  -- temp
                 , RES_CD
                 , RES_CAPA_VAL AS CAPACITY
                 , RES_TYPE_CD
                 , CAPA_UNIT_CD
                 , START_TIME_INDEX
                 , END_TIME_INDEX
            FROM M4E_I401100
        """
        return sql

    @staticmethod
    def sql_item_res_duration():
        sql = """
            SELECT FP_VRSN_ID
                 --, PLANT_CD
                 , 'K120' AS PLANT_CD
                 --, RES_CD AS RES_GRP_CD
                 , SUBSTRING(RES_CD, 1, CHARINDEX('@', RES_CD, 0)-1) AS RES_GRP_CD
                 --, ROUTE_CD AS ITEM_CD
                 , SUBSTRING(ROUTE_CD, CHARINDEX('@', ROUTE_CD, 0)+1, CHARINDEX('@', ROUTE_CD, CHARINDEX('@', ROUTE_CD, 0))) AS ITEM_CD
                 , ROUND(60 * CAPA_USE_RATE, 0) AS DURATION
              FROM M4E_I401120
             WHERE CAPA_USE_RATE IS NOT NULL
        """
        return sql
