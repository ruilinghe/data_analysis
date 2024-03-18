import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

df = pd.read_excel('./data.xlsx')
# print(df.columns)
new_columns = []
for col in df.columns:
    col_str = str(col)
    try:
        new_col = datetime.strptime(col_str, '%Y-%m-%d %H:%M:%S.%f')
        new_columns.append(new_col.strftime('%Y-%m-%d %H:%M:%S'))
    except ValueError:
        new_columns.append(col)

df.columns = new_columns
# print(df.columns)
# Melt the dataframe to convert it from wide format to long format
df_long = pd.melt(df, id_vars=['name'],  value_name='visitor',var_name='record_time')
df_long['record_time'] = pd.to_datetime(df_long['record_time'])


# Database connect parameters
conn = psycopg2.connect(
    dbname='mdap',
    user='postgres',
    password='123456',
    host='localhost'
)

engine = create_engine('postgresql+psycopg2://postgres:123456@localhost:5432/mdap')

# write data
df_long.to_sql('raw_data', engine, if_exists='replace', index=False) # table name

data = {
    'facility_id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'name': ['KATANGA CANYON', 'GLOOMY WOOD', 'THE SMILER', 'SHARKBAIT REEF', 'SPINBALL WHIZZER', 'MUTINY BAY', 'DARK FOREST', 'THE TOWERS', 'THE BLADE'],
    'maximum_capacity': [16, 10, 24, 18, 12, 10, 20, 14, 12],
    'runtime': [4, 3, 2, 8, 2, 3, 2, 5, 2]
}


df = pd.DataFrame(data)

df.to_sql('fixed_data', engine, if_exists='replace', index=False) # table name