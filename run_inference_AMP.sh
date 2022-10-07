INFERENCE_MODEL='TEST_BIN_AMPs'
BS=4
DEVICE=7
NUM_METADATA=8
FILE_INFERENCE='Example.csv'
python inference.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --batch_size $BS --save $INFERENCE_MODEL --file_infe $FILE_INFERENCE --metadata --num_metadata 8 --multilabel --nclasses 4

