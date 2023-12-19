#!/bin/bash

dir_path='../0_test_dataset/X/X_100_to_200'

for file in $(find $dir_path -name "*.vrp" -type f)
do
	echo $file
	CUDA_VISIBLE_DEVICES=0 python test.py --eas_batch_size 1 --ep 1 --iter 201 --filename $file --bias 0 --aug 8

done
