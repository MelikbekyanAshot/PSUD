import sqlite3 as sql


def f(value):
    return float(value.replace(',', '.')) ** 2


con = sql.connect('PSUD.db')
con.create_function('f', 1, f)
cur = con.cursor()
res = cur.execute("""select f(profit) from orders""")
print(res.fetchone())
cur.close()
con.close()
