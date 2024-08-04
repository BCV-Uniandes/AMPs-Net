#!/bin/bash

# Function to display usage message
usage() {
    echo "Usage: $0 -m INFERENCE_MODEL -b BS -d DEVICE -n NUM_METADATA -f FILE_INFERENCE"
    exit 1
}

# Parse command line arguments
while getopts m:b:d:n:f: flag
do
    case "${flag}" in
        m) INFERENCE_MODEL=${OPTARG};;
        b) BS=${OPTARG};;
        d) DEVICE=${OPTARG};;
        n) NUM_METADATA=${OPTARG};;
        f) FILE_INFERENCE=${OPTARG};;
        *) usage;;
    esac
done

# Check if all mandatory arguments are provided
if [ -z "$INFERENCE_MODEL" ] || [ -z "$BS" ] || [ -z "$DEVICE" ] || [ -z "$NUM_METADATA" ] || [ -z "$FILE_INFERENCE" ]; then
    usage
fi

# Run inference script
python inference.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --batch_size $BS --file_infe $FILE_INFERENCE --metadata --num_metadata $NUM_METADATA --binary --save $INFERENCE_MODEL

# Run filter peptides descriptors script
python filter_peptides_descriptors.py --file_name $FILE_INFERENCE