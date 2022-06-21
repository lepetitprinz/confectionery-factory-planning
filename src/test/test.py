from dao.io import DataIO
from common.sql import Query

io = DataIO()
query = Query()

data = io.load_from_db(sql=query.sql_calendar())

print("")

