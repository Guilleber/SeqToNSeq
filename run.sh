python3 train.py --use-cuda 1 --model-name DAE_dec2 --num-epochs 15
python3 translate.py --model models/DAE_dec2_e15.model --use-cuda 1 --input-file ./data/test.src --output-file ./results/DAE_dec2_e15_0.test --decoder 0
python3 translate.py --model models/DAE_dec2_e15.model --use-cuda 1 --input-file ./data/test.src --output-file ./results/DAE_dec2_e15_1.test --decoder 1
