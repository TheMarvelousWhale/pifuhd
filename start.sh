#!/bin/env bash

## pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

set -e

echo "removing previous results"


cd ../lightweight-human-pose-estimation.pytorch
git fetch

for file in $(ls ../pifuhd/sample_images/ | grep -E 'jpeg|jpg|png')
do
    echo "processing file $file..."
    python preprocessing.py ../pifuhd/sample_images/"$file"
done

cd ../pifuhd
git fetch
python -m apps.simple_test --use_rect -r 256 -i ./sample_images


set -e
cd ./results/pifuhd_final/recon/
for file in $(ls | grep obj)
do
    mv "$file" ./../../../../previous_results/"$(date "+%F-%T")"_"$file"
done
cd ./../../..

rm ./results/pifuhd_final/recon/*
rm ./sample_images/*