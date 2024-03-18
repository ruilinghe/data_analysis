import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import time
# 数据库连接参数
dbname = 'mdap'
user = 'postgres'
password = '123456'
host = 'localhost'

# 连接到数据库
conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)

def create_table():
    with conn.cursor() as cur:
        # 删除现有的表（如果存在）
        cur.execute("DROP TABLE IF EXISTS time_wait_data;")
        # 创建新表
        cur.execute("""
            CREATE TABLE time_wait_data (
                data_id SERIAL PRIMARY KEY,
                facility_id INT,
                name VARCHAR(255),
                wait_time INTEGER,
                current_queue INTEGER,
                record_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()

def simulate_minute_by_minute():
    with conn.cursor() as cur:
        # 获取固定数据：每个设施的最大容纳人数和运行时间
        cur.execute("SELECT facility_id, name, maximum_capacity, runtime FROM fixed_data;")
        fixed_info = {name: {'id': facility_id ,'capacity': capacity, 'runtime': runtime} for facility_id, name, capacity, runtime in cur.fetchall()}

        
        for day in [25, 26, 27, 28, 29]:
            # 定义时间范围
            start_hour = datetime(2023, 12, day, 10, 0)  # 起始时间
            end_hour = datetime(2023, 12, day, 18, 0)    # 结束时间，根据需要调整
            current_hour = start_hour

            while current_hour < end_hour:
                # 根据当前小时构造列名
                hour_column = current_hour.strftime('%Y-%m-%d %H:%M:%S')
                
                # 获取每个设施在当前小时开始时的等待人数
                cur.execute(f"SELECT name, visitor FROM raw_data WHERE record_time = \'{hour_column}\' ;")
                queue_data = cur.fetchall()

                # 遍历下一个小时的每一分钟
                for minute in range(60):
                    for name, queue_start in queue_data:
                        # 获取设施的容纳人数和运行时间
                        capacity = fixed_info[name]['capacity']
                        runtime = fixed_info[name]['runtime']
                        fid = fixed_info[name]['id']
                        decrease_per_minute = capacity / (runtime * 60)  # 每分钟减少的人数
                    
                        # current_time = current_hour + timedelta(minutes=minute)
                        # 更新等待人数
                        new_queue = max(0, queue_start - decrease_per_minute * minute)
                        
                        wait_time = new_queue / capacity *runtime
                        # 插入每分钟的数据到新表
                        cur.execute("""
                            INSERT INTO time_wait_data (facility_id, name, wait_time, current_queue) 
                            VALUES (%s, %s, %s, %s);
                        """, (fid, name, wait_time, new_queue))
                    conn.commit()
                    time.sleep(0.1)
                        

                # 移动到下一个小时的开始
                current_hour += timedelta(hours=1)
                

if __name__ == "__main__":
    create_table()
    simulate_minute_by_minute()