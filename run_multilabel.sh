EXPERIMENT='TEST_MULTILABEL_AMP'
BS=64
DEVICE=4
EPOCHS=300
python main2.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --cross_val 1 --batch_size $BS --save $EXPERIMENT --multilabel --nclasses 4 --lr 5e-5 --balanced_loader --metadata --epochs $EPOCHS
python main2.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --cross_val 2 --batch_size $BS --save $EXPERIMENT --multilabel --nclasses 4 --lr 5e-5 --balanced_loader --metadata --epochs $EPOCHS
python main2.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --cross_val 3 --batch_size $BS --save $EXPERIMENT --multilabel --nclasses 4 --lr 5e-5 --balanced_loader --metadata --epochs $EPOCHS
python main2.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --cross_val 4 --batch_size $BS --save $EXPERIMENT --multilabel --nclasses 4 --lr 5e-5 --balanced_loader --metadata --epochs $EPOCHS
python validation.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --batch_size $BS --save $EXPERIMENT --multilabel --nclasses 4
python train.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --batch_size $BS --save $EXPERIMENT --multilabel --nclasses 4
python test.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --batch_size $BS --save $EXPERIMENT --multilabel --nclasses 4
python test_ensamble.py --device $DEVICE --use_gpu --conv_encode_edge --block res+ --gcn_aggr softmax --learn_t --t 1.0 --dropout 0.2 --batch_size $BS --save $EXPERIMENT --multilabel --nclasses 4
