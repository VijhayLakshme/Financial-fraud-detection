from kafka import KafkaConsumer
import json
import pandas as pd
from fraud_detection import load_and_preprocess_data, train_models

consumer = KafkaConsumer('fraud_stream',
                         bootstrap_servers='localhost:9092',
                         auto_offset_reset='latest',
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')))

model = train_models(*load_and_preprocess_data())[0]['RandomForest']

for msg in consumer:
    data = msg.value
    df = pd.DataFrame([data])
    # Preprocess manually if needed
    prediction = model.predict(df)[0]
    print(f"Transaction: {data['transaction_id']} --> {'FRAUD' if prediction else 'LEGIT'}")
