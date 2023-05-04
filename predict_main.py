# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
from absl import app
from absl import flags
from absl import logging
import math, time
from termcolor import colored
import tensorflow as tf

from src import bert_example, tagging_converter
from src.utils import predict_utils
from src.utils import utils

from src.curLine_file import curLine

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', '/home/data/temp/zzx/lasertagger-chinese/corpus/rephrase_corpus/test.txt',
    'Path to the input file containing examples for which to compute '
    'predictions.')
flags.DEFINE_enum(
    'input_format', 'wikisplit', ['wikisplit'],
    'Format which indicates how to parse the input_file.')
flags.DEFINE_string(
    'output_file',
    '/home/data/temp/zzx/lasertagger-chinese/output/models/cefect/pred.tsv',
    'Path to the TSV file where the predictions are written to.')
flags.DEFINE_string(
    'label_map_file', '/home/data/temp/zzx/lasertagger-chinese/corpus/rephrase_corpus/output/label_map.txt',
    'Path to the label map file. Either a JSON file ending with ".json", that '
    'maps each possible tag to an ID, or a text file that has one tag per '
    'line.')
flags.DEFINE_string('vocab_file', '/home/data/temp/zzx/lasertagger-chinese/bert_base/RoBERTa-tiny-clue/vocab.txt',
                    'Path to the BERT vocabulary file.')
flags.DEFINE_integer('max_seq_length', 40, 'Maximum sequence length.')
flags.DEFINE_bool(
    'do_lower_case', False,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_bool('enable_swap_tag', True, 'Whether to enable the SWAP tag.')
flags.DEFINE_string('saved_model',
                    '/home/data/temp/zzx/lasertagger-chinese/output/cefect/export/1591764354',
                    'Path to an exported TF model.')


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('input_format')
    flags.mark_flag_as_required('output_file')
    flags.mark_flag_as_required('label_map_file')
    flags.mark_flag_as_required('vocab_file')
    flags.mark_flag_as_required('saved_model')

    label_map = utils.read_label_map(FLAGS.label_map_file)
    converter = tagging_converter.TaggingConverter(
        tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
        FLAGS.enable_swap_tag)
    builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                              FLAGS.max_seq_length,
                                              FLAGS.do_lower_case, converter)
    predictor = predict_utils.LaserTaggerPredictor(
        tf.contrib.predictor.from_saved_model(FLAGS.saved_model), builder,
        label_map)
    print(colored("%s input file:%s" % (curLine(), FLAGS.input_file), "green"))
    sources_list = []
    category_list = []
    with tf.io.gfile.GFile(FLAGS.input_file) as f:
        for line in f:
            sources, target = line.rstrip('\n').replace('\ufeff', '').split('[seq]')
            sources_list.append([sources + target])
            category_list.append(target)
    number = len(sources_list)  # 总样本数
    predict_batch_size = min(64, number)
    batch_num = math.ceil(float(number) / predict_batch_size)

    start_time = time.time()
    num_predicted = 0
    prediction_list = list()
    for batch_id in range(batch_num):
        sources_batch = sources_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
        prediction_batch = predictor.predict_batch(sources_batch=sources_batch)
        assert len(prediction_batch) == len(sources_batch)
        num_predicted += len(prediction_batch)
        for id, prediction in enumerate(prediction_batch):
            prediction_list.append(prediction)
    pd.DataFrame({'sources': sources_list, 'prediction': prediction_list, 'category': category_list}).to_csv(FLAGS.output_file)
    cost_time = (time.time() - start_time) / 60.0
    logging.info(
        f'{curLine()} {num_predicted} predictions saved to:{FLAGS.output_file}, cost {cost_time} min, ave {cost_time / num_predicted} min.')


if __name__ == '__main__':
    app.run(main)
