from kafka import KafkaProducer
import pandas as pd
import json
import time

producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

df = pd.read_csv('transactions.csv')

for _, row in df.iterrows():
    data = row.to_dict()
    producer.send('fraud_stream', data)
    time.sleep(0.5)
