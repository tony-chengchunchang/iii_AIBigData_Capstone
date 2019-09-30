# spark-submit --master spark://quickstart.cloudera:7077 --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.4.3 ml_k_to_ss.py


from pyspark.sql import Row, SparkSession
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, OneHotEncoderEstimator
from kafka import KafkaProducer
import json

def col_mapping(record):
    return Row(enemyStyle=int(record[0]),teamNumber=int(record[1]),enemyNumber=int(record[2]),fastBreak=int(record[3]),
                restrictedNumber=int(record[4]),perimeterNumber=int(record[5]),threeNumber=int(record[6]),freethrowNumber=int(record[7]),
                assist=int(record[8]),steal=int(record[9]),block=int(record[10]),foul=int(record[11]),
                turnover=int(record[12]),enemyPeriodScore=int(record[13]),winPeriod=int(record[14])
                )

def predict(rdd):
    if(rdd.count() == 0):
        return rdd
    
    df = rdd.toDF()
    
    ohe = OneHotEncoderEstimator.load('/home/cloudera/project/data_sources/ml_encoder')
    df_encoded = ohe.fit(df).transform(df)
    
    assembler = VectorAssembler.load('/home/cloudera/project/data_sources/ml_assembler')
    df_X = assembler.transform(df_encoded)
    
    model = RandomForestClassificationModel.load('/home/cloudera/project/data_sources/ml_model')
    temp = model.transform(df_X)
    
    pred = temp.select('prediction').rdd.map(lambda x:x[0])
    prob = temp.select('probability').rdd.map(lambda x:x[0])
    
    return pred.zip(prob)


def output_rdd(rdd):
    rdd.foreachPartition(output_partition)

def output_partition(partition):
    producer = KafkaProducer(bootstrap_servers='192.168.222.133:9092')
    topic = 'ml_output'
    
    for e in partition:
        pred = e[0]
        if int(pred) == 0:
            prob = e[1][0]
        else:
            prob = e[1][1]
            
        msg = '{},{}'.format(pred,prob)
        producer.send(topic, value=bytes(msg, 'utf-8'))
    
    producer.close()



if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    ssc = StreamingContext(sc,1)
    
    raw_stream = KafkaUtils.createStream(ssc, '192.168.222.133:2182', 'ml', {'ml_input':3})
    values = raw_stream.map(lambda x:x[1])
    split = values.map(lambda x:x.split(','))
    rows = split.map(col_mapping)
    res = rows.transform(predict)
    res.foreachRDD(output_rdd)
    
    ssc.start()
    ssc.awaitTermination()
    
    
    
    