
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
    def sql_demand():
        sql = """
            SELECT FP_VRSN_ID
                 , FP_KEY AS DMD_ID
                 , DP_KEY
                 , PLANT_CD 
                 , ITEM_CD
                 , RES_CD AS RES_GRP_CD
                 --, REQ_FP_YYMMDD
                 , '20220508' AS DUE_DATE
                 , REQ_FP_QTY AS QTY
                 -- , 100 AS QTY
              FROM M4S_I405020
              WHERE 1=1
        """
        return sql

    @staticmethod
    def sql_bom_route():
        sql = """
            SELECT PLANT_CD
                 , PROD_ITEM_CD AS PARENT_ITEM
                 , CNSM_ITEM_CD AS CHILD_ITEM
                 , BOM_LV AS BOM_LVL
                 , 1 AS RATE
                 -- , ROUND(PROD_VAL / CNSM_VAL, 3) AS RATE
              FROM M4S_I305200
             WHERE USE_YN = 'Y'
              AND LEFT(PROD_ITEM_CD, 1) = '5'
              AND CNSM_ITEM_CD IN (
                                   SELECT ITEM_CD
                                     FROM M4S_I002040
                                    WHERE ITEM_TYPE_CD = 'ROH1'
                                  )
        """
        return sql

    @staticmethod
    def sql_item_res_grp():
        sql = """
           SELECT PLANT_CD
                , ITEM_CD
                , PROD_VER_CD  AS RES_GRP_CD
             FROM M4S_I305400
        """
        return sql

    @staticmethod
    def sql_res_grp():
        sql = """
            SELECT PLANT_CD
                 , RES_GRP_CD
                 , RES_CD
                 , RES_CAPA_VAL AS CAPACITY
                 , RES_TYPE_CD
                 , CAPA_UNIT_CD
              FROM M4S_I305090
             WHERE USE_YN = 'Y'
        """
        return sql

    @staticmethod
    def sql_item_res_duration():
        sql = """
            SELECT PLANT_CD
                 , ITEM_CD
                 , ROUTE_CD
                 , RES_CD AS RES_GRP_CD
                 , ROUND(60 * CAPA_USE_RATE, 0) AS DURATION -- production time per 1 box                  , 'SEC' AS TIME_UOM
              FROM M4S_I305110
             WHERE CAPA_USE_RATE IS NOT NULL
               AND USE_YN = 'Y'
        """
        return sql