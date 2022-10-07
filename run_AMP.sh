EXPERIMENT='TEST_BIN_AMPs'
BS=112
DEVICE=3
EPOCHS=300
python main.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --cross_val 1 --batch_size $BS --binary --save $EXPERIMENT --balanced_loader --metadata --num_metadata 8 --epochs $EPOCHS 
python main.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --cross_val 2 --batch_size $BS --binary --save $EXPERIMENT --balanced_loader --metadata --num_metadata 8 --epochs $EPOCHS 
python main.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --cross_val 3 --batch_size $BS --binary --save $EXPERIMENT --balanced_loader --metadata --num_metadata 8 --epochs $EPOCHS 
python main.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --cross_val 4 --batch_size $BS --binary --save $EXPERIMENT --balanced_loader --metadata --num_metadata 8 --epochs $EPOCHS 
python validation.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --batch_size $BS --binary --save $EXPERIMENT --metadata --num_metadata 8
python train.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --batch_size $BS --binary --save $EXPERIMENT --metadata --num_metadata 8
python test.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --batch_size $BS --binary --save $EXPERIMENT --metadata --num_metadata 8
python test_ensamble.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --batch_size $BS --binary --save $EXPERIMENT --metadata --num_metadata 8
