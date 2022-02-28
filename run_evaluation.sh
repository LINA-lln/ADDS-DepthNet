#!/bin/bash


txt_file=('oxford_day_451' 'oxford_night_411')

for ((i=0;i<${#txt_file[*]};i++))
do
  eval_split=${txt_file[$i]}
  txt_name=${txt_file[$i]}$'_60.txt'
  for j in $(seq 19 19)
  do
    load_weights_folder='pretrained_model'
    echo ${eval_split} $load_weights_folder
    mean_errors=`python evaluate_depth.py --load_weights_folder $load_weights_folder --eval_split $eval_split`
    echo $mean_errors
    echo $mean_errors >> $txt_name

  done
#   break
done