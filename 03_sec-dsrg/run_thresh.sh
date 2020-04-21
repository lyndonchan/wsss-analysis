python demo.py --method SEC --dataset ADP-morph --seed VGG16 --eval_setname tuning -i -v -t 0.3
python demo.py --method SEC --dataset ADP-morph --seed X1.7 --eval_setname tuning -i -v -t 0.3
python demo.py --method DSRG --dataset ADP-morph --seed VGG16 --eval_setname tuning -i -v
python demo.py --method DSRG --dataset ADP-morph --seed X1.7 --eval_setname tuning -i -v
python demo.py --method SEC --dataset ADP-morph --seed VGG16 --eval_setname segtest -i -v
python demo.py --method SEC --dataset ADP-morph --seed X1.7 --eval_setname segtest -i -v
python demo.py --method DSRG --dataset ADP-morph --seed VGG16 --eval_setname segtest -i -v
python demo.py --method DSRG --dataset ADP-morph --seed X1.7 --eval_setname segtest -i -v