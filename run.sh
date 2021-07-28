#!/bin/bash
source /etc/profile
export LIB_JVM=/usr/jdk64/jdk1.8.0_112/jre/lib/amd64/server/
export LIB_HDFS=/usr/hdp/3.1.5.0-152/usr/lib/
echo $LIB_HDFS
export LD_LIBRARY_PATH=$LIB_HDFS:$LIB_JVM
echo $LD_LIBRARY_PATH
export CLASSPATH="$(hadoop classpath --glob)"
export PYSPARK_PYTHON=/opt/app/anaconda3/bin/python3

ls -l
spark-submit \
  --master yarn \
  --num-executors 1 \
  --executor-cores 2 \
  --executor-memory 2g \
  --driver-memory 8g \
  --jars hdfs:///your_path_on_hdfs/spark-tensorflow-connector_2.11-1.11.0.jar \
  --conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
  --conf spark.executorEnv.PYSPARK_PYTHON=/opt/app/anaconda3/bin/python3 \
  --files  vocab.txt \
  --py-files  news_quality.py \
  news_quality.py \
  --config-path "bert_config_256.json" \
  --checkpoint-path "hdfs:///your_bert_checkpoint_path/bert_model.ckpt" \
  --train-data-file "content_quality_train.csv" \
  --max-len 256 \
  --batch-size 16 \
  --epochs 1 \
  --learning-rate 0.00005 \
  --test-ratio 0.3 \
  --model-save-path hdfs:///user/your_model_save_path/SaveModel/v1 \
  --model-desc-save-path hdfs:///user/your_model_save_path/SaveModel/v1/model_desc.json
