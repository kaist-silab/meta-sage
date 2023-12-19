#!/bin/bash
 
dir_path='../0_test_dataset/X/X_900_to_1000'

for file in $(find $dir_path -name "*.vrp" -type f)
do
	echo $file
        #CUDA_VISIBLE_DEVICES=1 python test.py --eas_batch_size 1 --ep 1 --iter 201 --filename $file --bias 1 --aug 8

done
