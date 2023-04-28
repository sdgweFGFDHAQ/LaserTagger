# 最傻的办法在shell中运行conda指定环境
export python=python
# 然后：${python} xxx.py

export Root_Dir="/home/data/temp/zzx/lasertagger-chinese"
export OUTPUT_DIR="${Root_Dir}/output"
EXPERIMENT=cefect

export CONFIG_FILE="${Root_Dir}/configs/lasertagger_config.json"

#导出那个模型为pd模型？pd，模型轻量更快，适合预测
export EXPORT_FILE=${OUTPUT_DIR}/models/model.ckpt-8000

# Export the model.
${python} run_lasertagger.py \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --init_checkpoint=${EXPORT_FILE}\
  --model_config_file=${CONFIG_FILE} \
  --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
  --do_train=false \
  --do_eval=false \
  --do_export=true \
  --export_path=${OUTPUT_DIR}/models/${EXPERIMENT}
