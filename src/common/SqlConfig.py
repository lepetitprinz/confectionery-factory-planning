
class SqlConfig(object):
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
    def sql_res_item():
        sql = """
           SELECT PLANT_CD
                , ITEM_CD
                , PROD_VER_CD  AS RES_CD
             FROM M4S_I305400
        """
        return sql

    @staticmethod
    def sql_res_cnt_capa():
        sql = """
            SELECT PLANT_CD
                 , RES_GRP_CD AS RES_CD
                 , COUNT(RES_CD) AS RES_CNT
                 , AVG(RES_CAPA_VAL) AS RES_CAPA
              FROM M4S_I305090
             GROUP BY PLANT_CD
                    , RES_GRP_CD
        """
        return sql

    @staticmethod
    def sql_demand():
        sql = """
            SELECT TOP 1 IF_VRSN_ID
                 , FP_VRSN_ID
                 , FP_KEY AS DEMAND_ID
                 , DP_KEY
                 , ITEM_CD
                 , PLANT_CD 
                 , RES_CD AS WC_CD
                 --, REQ_FP_YYMMDD
                 , '20221231' AS DUEDATE
                 --, REQ_FP_QTY AS QTY
                 , 1 AS QTY
              FROM M4S_I405020
        """
        return sql

    @staticmethod
    def sql_operation():
        sql = """
            SELECT PLANT_CD
                 , ITEM_CD
                 , ROUTE_CD AS OPERATION_NO
                 , RES_CD AS WC_CD
                 , ROUND(100 * (CAPA_USE_RATE), 0) AS SCHD_TIME
                 , 'MIN' AS TIME_UOM
              FROM M4S_I305110
             WHERE CAPA_USE_RATE IS NOT NULL
               AND USE_YN = 'Y'
        """
        return sql