from kafka import KafkaProducer
import cv2

topic = 'dl_input'
broker_list = 'localhost:9092'

img = cv2.imread('5.jpg')
ret, buf = cv2.imencode('.jpg', img)


producer = KafkaProducer(bootstrap_servers=broker_list)

producer.send(topic, value=buf.tobytes())

producer.close()   