INFERENCE_MODEL=''
BS=4
DEVICE=7
NUM_METADATA=8
FILE_INFERENCE='Example.csv'
python inference.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --batch_size $BS --file_infe $FILE_INFERENCE --metadata --num_metadata 8 --binary
python filter_peptides_descriptors.py --file_name $FILE_INFERENCE
