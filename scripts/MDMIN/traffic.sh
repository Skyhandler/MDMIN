if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MDMIN

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 2 \
      --n_heads 1 \
      --d_model 34 \
      --d_ff 372 \
      --dropout 0.105\
      --fc_dropout 0.12\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 \
      --batch_size 64 \
      --layers_num 3 \
      --hidden_channels 91 \
      --momentum 0.8200000000000001 \
      --out_channels 24 \
      --stride1 2 \
      --stride2 5 \
      --stride3 7 \
      --learning_rate 0.00676 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done