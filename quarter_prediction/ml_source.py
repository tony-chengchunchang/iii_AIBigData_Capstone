from kafka import KafkaProducer
import time

with open('../data_sources/test_line.csv', 'r') as f:
    data = f.readlines()

data = data[1:]
revised = [e[:-1] for e in data]

producer = KafkaProducer(bootstrap_servers='localhost:9092')
topic = 'ml_input'

for msg in revised:
    producer.send(topic, value=bytes(msg, 'utf-8'))
    # time.sleep(1)
producer.close()

