# the commend to run on expense server
for seed in 1111;
do
    sbatch expanse_setup \
    "python -u svhn.py
    --loss_type MSE
    --rescale_factor 1
    --lr 0.005
    --name svhn_sqen"
done
