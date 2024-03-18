import pandas as pd
import numpy as np
import holidays
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import psycopg2
import matplotlib.pyplot as plt

def load_and_preprocess_data(data_path):
    # Load the data
    data = pd.read_excel(data_path)
    new_columns = []
    for col in data.columns:
        col_str = str(col)
        try:
            new_col = datetime.strptime(col_str, '%Y-%m-%d %H:%M:%S.%f')
            new_columns.append(new_col.strftime('%Y-%m-%d %H:%M:%S'))
        except ValueError:
            new_columns.append(col)

    data.columns = new_columns
    data.columns = data.columns.map(str)  # Ensure all column names are strings for subsequent operations

    # Melt the dataframe to long format
    melted_data = pd.melt(data, id_vars=['name'], var_name='datetime', value_name='people_count')
    melted_data['datetime'] = pd.to_datetime(melted_data['datetime'], errors='coerce')
    
    return melted_data

def split_data(melted_data, validation_date):
    # 过滤出训练集
    train_data = melted_data[melted_data['datetime'].dt.date < validation_date]
    # 过滤出验证集
    validation_data = melted_data[melted_data['datetime'].dt.date == validation_date]
    
    return train_data, validation_data



def add_time_features(df):
    uk_holidays = holidays.UnitedKingdom()
    df['is_holiday'] = df['datetime'].dt.date.apply(lambda x: x in uk_holidays).astype(int)
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    return df

def calculate_historical_averages(dataset):
    # Calculate the mean people count grouped by facility name and hour
    averages = dataset.groupby(['name', 'hour']).mean().reset_index()
    return averages

def predict_using_averages(dataset, averages):
    # Merge the historical averages with the dataset based on facility name and hour
    predictions = pd.merge(dataset, averages, on=['name', 'hour'], how='left')
    return predictions


def visualize_results(train_data, validation_data, validation_predictions, test_data, test_predictions, sample_facility_name):
    plt.figure(figsize=(15, 7))

    # 绘制训练数据
    if not train_data.empty:
        plt.plot(train_data['datetime'], train_data['people_count'], label='Training', color='green')

    # 绘制验证数据及其预测
    if not validation_data.empty:
        plt.plot(validation_data['datetime'], validation_data['people_count'], label='Validation Actual', marker='o', linestyle='-', color='blue')
        plt.plot(validation_data['datetime'], validation_predictions['predicted'], label='Validation Predicted', linestyle='--', marker='x', color='red')

    # 绘制测试数据及其预测（注意测试集可能没有实际的人流量数据）
    if not test_data.empty:
        # 如果测试数据有实际的人流量数据，则绘制它（这可能不适用，因为测试数据可能没有'actual'列）
        if 'actual' in test_data:
            plt.plot(test_data['datetime'], test_data['actual'], label='Test Actual', marker='o', linestyle='-', color='purple')
        # 绘制测试数据的预测结果
        plt.plot(test_data['datetime'], test_predictions['predicted'], label='Test Predicted', linestyle='--', marker='>', color='orange')

    plt.title(f'People Counts for {sample_facility_name}: Training, Validation and Test')
    plt.xlabel('Time')
    plt.ylabel('People Count')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def write_prediction_to_db(predictions_df):
    conn = psycopg2.connect(dbname='mdap', user='postgres', password='123456', host='localhost')
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS predict_data;")
    # 创建新表（如果尚不存在）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predict_data (
        name VARCHAR(255),
        record_time timestamp,
        visitor bigint,
        PRIMARY KEY (name, record_time)
    );
    """)
    conn.commit()

    # 插入或更新预测结果
    for index, row in predictions_df.iterrows():
        cur.execute("""
        INSERT INTO predict_data (name, record_time, visitor)
        VALUES (%s, %s, %s)
        ON CONFLICT (name, record_time) 
        DO UPDATE SET visitor = EXCLUDED.visitor;
        """, (row['name'], row['datetime'], row['predicted']))
    
    conn.commit()
    cur.close()
    conn.close()

def create_testdata(train_data, test_date) :
    facility_names = train_data['name'].unique()
    test_hours = range(10,19) 
    test_data_list = []

    for name in facility_names:
        for hour in test_hours:
            # 创建每个小时的记录
            record_time = test_date + timedelta(hours=hour)
            test_data_list.append({'name': name, 'datetime': record_time, 'people_count': 0})  # 可以将people_count设为0或保持为空
    
    test_data = pd.DataFrame(test_data_list)
    # 确保datetime是正确格式
    test_data['datetime'] = pd.to_datetime(test_data['datetime'])

    # 添加时间特征
    test_data = add_time_features(test_data)
    return test_data

if __name__ == "__main__":
    data_path = 'data.xlsx'
    validation_date = datetime(2023, 12, 29).date()
    test_date = datetime(2023, 12, 30)
    melted_data = load_and_preprocess_data(data_path)

    # 拆分数据为训练集、验证集和测试集
    melted_data = add_time_features(melted_data)
    train_data, validation_data = split_data(melted_data, validation_date)

    # 计算历史平均值并进行预测
    historical_averages = calculate_historical_averages(train_data)
    validation_predictions = predict_using_averages(validation_data, historical_averages)

    test_data = create_testdata(train_data, test_date)
    print(test_data)
    historical_averages = calculate_historical_averages(melted_data)
    test_predictions = predict_using_averages(test_data, historical_averages)
    print(validation_predictions)
    print(test_predictions)
    # 评估验证集预测的性能
    validation_predictions.rename(columns={'people_count_x': 'actual', 'people_count_y': 'predicted'}, inplace=True)
    test_predictions.rename(columns={'people_count_x': 'actual', 'people_count_y': 'predicted'}, inplace=True)
    
    mse = mean_squared_error(validation_predictions['actual'], validation_predictions['predicted'])
    mape = mean_absolute_percentage_error(validation_predictions['actual'], validation_predictions['predicted'])
    print(f'Validation MSE: {mse}, MAPE: {mape}')

    # 将测试集预测结果存入数据库
    predictions = pd.concat([validation_predictions, test_predictions], ignore_index=True)
    write_prediction_to_db(predictions)

    # 可视化验证集的结果
    sample_facility_name = train_data['name'].unique()[2]  # 或者选择其他设施
    visualize_results(
        train_data[train_data['name'] == sample_facility_name], 
        validation_data[validation_data['name'] == sample_facility_name],
        validation_predictions[validation_predictions['name'] == sample_facility_name],
        test_data[test_data['name'] == sample_facility_name],
        test_predictions[test_predictions['name'] == sample_facility_name],
        sample_facility_name
    )

