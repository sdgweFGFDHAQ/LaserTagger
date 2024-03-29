# 最傻的办法在shell中运行conda指定环境
export python=python
# 然后：${python} xxx.py

# set gpu id to use
export CUDA_VISIBLE_DEVICES=""
export Root_Dir="/home/data/temp/zzx/lasertagger-chinese"
export WIKISPLIT_DIR="${Root_Dir}/corpus/rephrase_corpus"
export BERT_BASE_DIR="${Root_Dir}/bert_base/RoBERTa-tiny-clue"
export OUTPUT_DIR="${Root_Dir}/output"
export max_seq_length=20
export CONFIG_FILE="${Root_Dir}/configs/lasertagger_config.json"
export EXPERIMENT=cefect

### 4. Prediction

TIMESTAMP=$(ls "${OUTPUT_DIR}/models/${EXPERIMENT}/" |
  grep -v "temp-" | sort -r | head -1)

# 保存的pb模型路径
SAVED_MODEL_DIR=${OUTPUT_DIR}/models/${EXPERIMENT}/${TIMESTAMP}
#输出路径
PREDICTION_FILE=${OUTPUT_DIR}/models/pred.csv

${python} predict_main.py \
  --input_file=${WIKISPLIT_DIR}/test.txt \
  --input_format=wikisplit \
  --output_file=${PREDICTION_FILE} \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --max_seq_length=${max_seq_length} \
  --saved_model=${SAVED_MODEL_DIR}
