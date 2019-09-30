from kafka import KafkaConsumer
import requests

def decoder(value):
    return value.decode('utf-8')
    
topic = 'dl_output'
servers = 'localhost:9092'

consumer = KafkaConsumer(topic, bootstrap_servers=servers, value_deserializer=decoder)

for msg in consumer:
    # print(msg.value)

    value = msg.value.split(',')
    team = value[0]
    num = value[1]
    score = value[2]
    reb = value[3]
    ast = value[4]
    
    v1 = '{} {}號球員即將上場'.format(team, num)
    v2 = '平均數據: 得分{}, 籃板{}, 助攻{}'.format(score,reb,ast)
    
    # Replace this to your own IFTTT webhook
    # https://help.ifttt.com/hc/en-us/categories/115001566148-Getting-Started
    url =('https://maker.ifttt.com/trigger/myEventName/with/key/myKey' +
               '?value1=' + v1 +
               '&value2=' + v2 +
               '&value3=')
    r = requests.get(url)
    if r.text[:5] == 'Congr':
        print('已傳送 ' + msg.value + ' 到 Line')


