# 导入所需的包
import pandas as pd
import numpy as np
import time
import logging
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
import random
import unicodedata
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import json
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
模型需求：内容中心 内容质量评分
@time：2021/01/20
@description：运用BERT 预训练模型进行内容质量评分
@author: szlemail
"""

# 常量定义

TOKENS_TENSOR_NAME = "tokens"
IDS_TENSOR_NAME = "ids"


def parse_args():
    """解析参数."""
    parser = argparse.ArgumentParser(usage="it's usage tip.",
                                     description="predict news type")
    parser.add_argument("--dict-path", default="vocab.txt", help="tokenizer字典")
    parser.add_argument("--config-path", default="bert_config.json",
                        help="模型配置文件")
    parser.add_argument("--checkpoint-path", default="bert_model.ckpt",
                        help="BERT预训练checkpoint路径")
    parser.add_argument("--train-data-file",
                        default="content_quality_train.csv",
                        help="训练样本文件")
    parser.add_argument("--max-len", default=256, type=int,
                        help="最大文本字数")
    parser.add_argument("--batch-size", default=16, type=int,
                        help="模型训练每个批次样本数量")
    parser.add_argument("--epochs", default=3, type=int,
                        help="样本训练轮数")
    parser.add_argument("--model-save-path", type=str, help="模型存储路径")
    parser.add_argument("--model-desc-save-path", type=str, help="模型描述文件存储路径")
    parser.add_argument("--learning-rate", default=0.001, type=float,
                        help="模型学习率")
    parser.add_argument("--test-ratio", default=0.3, type=float,
                        help="模型学习率")

    return parser.parse_args()


def get_logger(file_path=None):
    # 创建一个logger
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(ch)

    # 再创建一个handler 输出到文件，如果需要的话
    if file_path is not None:
        timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
        fh = logging.FileHandler(file_path + 'log_' + timestamp + '.txt')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def get_tokenizer(dict_path):
    """
    获取BERT 编码器
    """
    token_dict = {}
    with open(dict_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = Tokenizer(token_dict)
    return tokenizer


class DataGenerator:
    """
    数据批量生成和编码，节约内存
    """

    def __init__(self, data, tokenizer, max_len=256, batch_size=16,
                 is_shuffle=True, is_predict=False):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.is_predict = is_predict
        self.is_shuffle = is_shuffle
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            indexes = list(range(len(self.data)))
            if self.is_shuffle:
                np.random.shuffle(indexes)
            tokens, ids, labels = [], [], []
            for i in indexes:
                value = self.data[i]
                _token, _id = self.tokenizer.encode(first=value[0],
                                                    second=value[1],
                                                    max_len=self.max_len)
                tokens.append(_token)
                ids.append(_id)
                labels.append(value[3])
                if len(tokens) == self.batch_size or i == indexes[-1]:
                    tokens = np.array(tokens)
                    ids = np.array(ids)
                    labels = np.array(labels)
                    yield [tokens, ids], labels
                    tokens, ids, labels = [], [], []
            if self.is_predict:
                break


class Evaluator(Callback):
    """
    训练评估类，评估指标，调整学习率
    """

    def __init__(self, val_data, tokenizer, max_len=256, learning_rate=5e-5,
                 min_learning_rate=1e-5):
        Callback.__init__(self)
        self.score = []
        self.best = 0.
        self.early_stopping = 0
        self.val_data = val_data
        self.predict = []
        self.lr = 0
        self.passed = 0
        self.max_len = max_len
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.tokenizer = tokenizer

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            self.lr = (self.passed + 1.) / self.params[
                'steps'] * self.learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            self.lr = (2 - (self.passed + 1.) / self.params['steps']) * (
                    self.learning_rate - self.min_learning_rate)
            self.lr += self.min_learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        acc, f1 = self.evaluate()
        if f1 > self.best:
            self.best = f1
            self.early_stopping = 0
        else:
            self.early_stopping += 1
        logger.info('lr: %.6f, epoch: %d, acc: %.4f, f1: %.4f,best: %.4f\n' % (
            self.lr, epoch, acc, f1, self.best))

    def evaluate(self):
        self.predict = []
        prob = []
        label_val = []
        for i in range(len(self.val_data)):
            element = self.val_data[i]
            tokens, ids = self.tokenizer.encode(first=element[0],
                                                second=element[1],
                                                max_len=self.max_len)
            _prob = self.model.predict([np.array([tokens]), np.array([ids])])
            self.predict.append(np.int64(_prob[0][0] > 0.2))
            prob.append(_prob[0][0])
            label_val.append(element[3])

        acc = accuracy_score(label_val, self.predict)
        f1 = f1_score(label_val, self.predict)
        return acc, f1


# 创建模型类
def get_model(config_path, checkpoint_path):
    bert_model = load_trained_model_from_checkpoint(config_path,
                                                    checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    tokens = Input(shape=(None,), name=TOKENS_TENSOR_NAME)
    ids = Input(shape=(None,), name=IDS_TENSOR_NAME)

    bert_out = bert_model([tokens, ids])

    token_vector = Lambda(lambda x: x[:, 0])(bert_out)
    output = Dense(1, activation='sigmoid',
                   kernel_regularizer=l1_l2(l1=0.005, l2=0.001))(token_vector)

    model = Model([tokens, ids], output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model


# 定义模型描述类
class ModelDescription(object):
    """分类模型描述文件

        描述分类模型的入参出参等字段

        参数:
            dim: 数据维度
            map_key: 模型入参字段和输入字段对应字段
            data_type: 模型入参字段名
            handler: 处理器
            fill_value: 字段不足时填充值

        返回:
            model: 模型描述文件字典
        """

    model = {}

    def __init__(self):
        self.model['model_desc'] = {}
        self.model['model_desc']['signature_name'] = ""
        self.model['model_desc']['inputs'] = {}
        self.model['model_desc']['outputs'] = []
        pass

    def build_context_field(self, dim, map_key, tensor_name, data_type="int",
                            handler="tokenizer", fill_value=0):
        field = {'dim': dim, 'map_key': map_key, 'tensor_name': tensor_name,
                 'data_type': data_type, 'handler': handler,
                 'fill_value': fill_value}
        return field

    def build_tokens(self, dim, input_fields):
        return self.build_context_field(dim, input_fields,
                                        TOKENS_TENSOR_NAME,
                                        handler="bertTokens")

    def build_ids(self, dim, input_fields):
        return self.build_context_field(dim, input_fields,
                                        IDS_TENSOR_NAME,
                                        handler="bertIds")

    def set_context(self, dim, input_fields):
        tokens = self.build_tokens(dim, input_fields)
        ids = self.build_ids(dim, input_fields)
        self.model['model_desc']['inputs']['context'] = [tokens, ids]

    def add_out_put(self, map_key, tensor_name, tag_name):
        output = {"map_key": map_key, "tensor_name": tensor_name,
                  "data_type": "int", "handler": "tags",
                  "tag_name": tag_name, "fill_value": "0", "dim": -1}
        self.model['model_desc']['outputs'] = self.model['model_desc'][
                                                  'outputs'] + [output]

    def to_json(self):
        return json.dumps(self.model, ensure_ascii=False)


def save_file(spark, path: str, data: str):
    """保存文件至hdfs.
    参数:
        path(str): hdfs上的路径
        data(str): 数据
    """
    sc = spark.sparkContext
    fs_class = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    conf_class = sc._gateway.jvm.org.apache.hadoop.conf.Configuration
    fs = fs_class.get(conf_class())
    path_class = sc._gateway.jvm.org.apache.hadoop.fs.Path
    output = fs.create(path_class(path))
    output.write(data.encode())
    output.flush()
    output.close()


def save_model(spark, model, args):
    # 保存模型到HDFS
    model.save(args.model_save_path)

    # 以下是为了保存模型描述文件
    news_model = ModelDescription()
    news_model.set_context(dim=args.max_len, input_fields="title,content")
    # 保存描述文件到HDFS
    data = news_model.to_json()
    save_file(spark, args.model_desc_save_path, data)


# 主函数入口
if __name__ == '__main__':
    logger = get_logger()
    args = parse_args()
    spark = SparkSession \
        .builder \
        .appName("内容质量评分") \
        .config("spark.sql.broadcastTimeout", "3000") \
        .master("yarn") \
        .enableHiveSupport() \
        .getOrCreate()
    tf.random.set_seed(1)
    tokenizer = get_tokenizer(args.dict_path)
    data = pd.read_csv(args.train_data_file).fillna("")
    split_index = int(len(data) * (1 - args.test_ratio))
    train_data = data[:split_index].values.tolist()
    test_data = data[split_index:].values.tolist()
    train_data_set = DataGenerator(train_data,
                                   tokenizer,
                                   max_len=args.max_len,
                                   batch_size=args.batch_size,
                                   is_shuffle=False
                                   )
    evaluator = Evaluator(val_data=test_data,
                          tokenizer=tokenizer,
                          max_len=args.max_len,
                          learning_rate=args.learning_rate,
                          min_learning_rate=args.learning_rate / 5.0
                          )
    model = get_model(args.config_path, args.checkpoint_path)
    model.fit_generator(train_data_set.__iter__(),
                        steps_per_epoch=len(train_data_set),
                        epochs=args.epochs,
                        callbacks=[evaluator]
                        )
    model.save(args.model_save_path)
    save_model(spark, model, args)
