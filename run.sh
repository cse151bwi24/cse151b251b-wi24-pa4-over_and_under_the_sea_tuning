mkdir assets
python main.py --n-epochs 10 --do-train --learning-rate 0.00005
# python main.py --n-epochs 10 --do-train --learning-rate 0.00005 >> results/baseline/default-lr5e5.txt
# python main.py --n-epochs 10 --do-train --learning-rate 0.00001 >> results/baseline/default-lr1e5.txt
# python main.py --n-epochs 10 --do-train --learning-rate 0.0001 >> results/baseline/default-lr1e4.txt
# python main.py --n-epochs 10 --do-train --learning-rate 0.001 >> results/baseline/default-lr1e3.txt
# python main.py --n-epochs 10 --do-train --learning-rate 0.00005 --scheduler cosine >> results/baseline/default-lr5e5-sched.txt
# python main.py --n-epochs 10 --do-train --learning-rate 0.00005 --scheduler cosine --drop-rate 0.5 >> results/baseline/default-lr5e5-sched-dr0.5.txt
# python main.py --n-epochs 10 --do-train --learning-rate 0.00005 --scheduler cosine --drop-rate 0.3 >> results/baseline/default-lr5e5-sched-dr0.3.txt
# python main.py --n-epochs 10 --do-train --learning-rate 0.00005 --scheduler cosine --drop-rate 0.2 >> results/baseline/default-lr5e5-sched-dr0.2.txt
# python main.py --n-epochs 10 --do-train --learning-rate 0.00005 --scheduler cosine --hidden-dim 20 >> results/baseline/default-lr5e5-sched-hidden20.txt
# python main.py --n-epochs 10 --do-train --learning-rate 0.00005 --scheduler cosine --hidden-dim 30 >> results/baseline/default-lr5e5-sched-hidden30.txt
# python main.py --n-epochs 10 --do-train --learning-rate 0.00005 --scheduler cosine --hidden-dim 40 >> results/baseline/default-lr5e5-sched-hidden40.txt



# python main.py --n-epochs 10 --do-train --task custom --reinit_n_layers 3
# python main.py --n-epochs 10 --do-train --task supcon --batch-size 64
