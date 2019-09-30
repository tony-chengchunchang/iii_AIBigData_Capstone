from pyspark.sql import Row, SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, OneHotEncoderEstimator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    data = spark.read.csv('train_data.csv', header=True, inferSchema=True)
    
    ohe = OneHotEncoderEstimator(inputCols=['enemyStyle'], outputCols=['enemyStyle'+'_classVec'])
    data_encoded = ohe.fit(data).transform(data)
    ohe_path = 'quarter_prediction/ohe'
    ohe.save(ohe_path)
    
    cols = data_encoded.columns
    cols.remove('enemyStyle')
    cols.remove('winPeriod')
    
    assembler = VectorAssembler(inputCols=cols, outputCol='features')
    assembler_path = 'quarter_prediction/assembler'
    assembler.save(assembler_path)
    
    train = assembler.transform(data_encoded)
    
    rf = RandomForestClassifier(maxDepth=6, numTrees=14, impurity='entropy', labelCol='winPeriod')
    model = rf.fit(train)
    
    model_path = 'quarter_prediction/ml_model'
    model.save(model_path)
    
    # res = model.transform(train)
    # evaluator = BinaryClassificationEvaluator(labelCol='winPeriod')
    # evaluator.evaluate(res)