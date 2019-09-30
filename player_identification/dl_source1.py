from kafka import KafkaProducer
import cv2

topic = 'dl_input'
broker_list = 'myServIp:port'

img = cv2.imread('/home/cloudera/project/data_sources/5.jpg')
ret, buf = cv2.imencode('.jpg', img)


producer = KafkaProducer(bootstrap_servers=broker_list)

producer.send(topic, value=buf.tobytes())

producer.close()   