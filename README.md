# iii_AIBigData_Capstone

This is the final project of all my study at Institute for Information Industry. (Taiwan)

## Objective
Use AI & Big Data tools to improve the performances of the teams of the community basketball league.

## Subjects
- Quarter Prediction:
Use the first 8 minutes stats to predict the result at the end of the quarter.
- Player Identification:
Use the image recognition solution to identify and return the average stats of each opponent player when they are checking in to the game.

## Environment Setup
- Spark 2.4.3
- Kafka 0.10.2.1
- Scala 2.11
- jdk-8u221 


## Data Pipeline
Data source --> Kafka --> Spark Streaming --> IFTTT & Line APP

## Run the Job
- Before Starting:
    - Create Kafka Topics of input and output for both subjects.
    - Create and make the connection of a new IFTTT webhook url. (https://help.ifttt.com/hc/en-us/categories/115001566148-Getting-Started)
   
- Quarter Prediction:
     - Build the model for quarter prediction: ```python ml_model.py```
    
    - Start a Spark Streaming job:
    ```
    spark-submit --master mySparkCore --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.4.3 ml_k_to_ss.py
    ```
    
    - Start the IFTTT service:```python ml_IFTTT.py```
    
    - Input data source:```python ml_source.py```
    
    - The result should be sent to every group that associated with the IFTTT service.
    
- Player Identification:
    - Compress the folders of openpose, digi-detector, model, color to ```dependencies.zip```
    
    - Start a Spark Streaming job:
    ```
    spark-submit --master mySparkCore --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.4.3 --py-files dependencies.zip --files player_info_new.csv,config dl_k_to_ss.py
    ```
    
    - Start the IFTTT service:```python dl_IFTTT.py```
    
    - Input data source:```python dl_source.py```
    
    - The result should be sent to every group that associated with the IFTTT service.
    
    
    
    
    
