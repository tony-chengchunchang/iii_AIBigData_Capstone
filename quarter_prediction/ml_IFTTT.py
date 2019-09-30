from kafka import KafkaConsumer
import requests

def decoder(value):
    return value.decode('utf-8')
    
topic = 'ml_output'
servers = 'localhost:9092'

consumer = KafkaConsumer(topic, bootstrap_servers=servers, value_deserializer=decoder)

for msg in consumer:
    value = msg.value.split(',')
    pred = float(value[0])
    prob = round(float(value[1]),2)
    
    v1 = 'Penrite更好的機油 隊長您好'
    if pred == 1.0:
        v2 = '[預測]本節得分將 多於 對手'
        v3 = '--請繼續保持狀態--'
    else:
        v2 = '[預測]本節得分將 少於 對手'
        v3 = '--請調整陣容配置--'
        
    # Replace this to your own IFTTT webhook
    # https://help.ifttt.com/hc/en-us/categories/115001566148-Getting-Started
    url =('https://maker.ifttt.com/trigger/myEvent/with/key/myKey' +
               '?value1=' + v1 +
               '&value2=' + v2 +
               '&value3=' + v3)
    r = requests.get(url)
    if r.text[:5] == 'Congr':
        print('已傳送 ' + msg.value + ' 到 Line')