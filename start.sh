#!/bin/env bash

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

mv ./results/pifuhd_final/recon/*.obj ../previous_results/
rm ./results/pifuhd_final/recon/*
rm ./sample_images/*