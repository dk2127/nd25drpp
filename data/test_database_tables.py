# -*- coding: utf-8 -*-
"""
Created on 2/16/2023 7:48pm
This program validates if the database is readable.
@author: dk2127
"""

# Create the SQLite engine
import os
import pandas as pd
from sqlalchemy import create_engine

print("path exists", os.path.exists('/home/workspace/data/DisasterResponse.db'))

database_url = 'sqlite:///DisasterResponse.db'
engine = create_engine(database_url)

# +
try:
    with engine.connect() as connection:
        result = connection.execute("SELECT COUNT(*) FROM DisasterResponse")
        print("Row count=" , result.fetchall())
##    result2 = connection.execute("SELECT message, COUNT(*) as KNT FROM DisasterResponse where related = 2 group by message")
## print("\n Rows with related=2\n" , result2.fetchall())
        
except Exception as e:
    print(f"An error occurred: {e}")
else:
    print("Connection to database established successfully.")
# -

df = pd.read_sql("SELECT message, COUNT(*) as KNT FROM DisasterResponse where related = 2 group by message order by 2 desc", engine)

df.index


