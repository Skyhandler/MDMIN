if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MDMIN

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2021
for pred_len in 96
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
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 2 \
      --d_model 32 \
      --d_ff 334 \
      --dropout 0.179\
      --fc_dropout 0.146\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --layers_num 1 \
      --hidden_channels 19 \
      --momentum 0.89 \
      --out_channels 32 \
      --stride1 2 \
      --stride2 3 \
      --stride3 5 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4 \
      --itr 1 \
      --batch_size 116 \
      --momentum 0.84 \
      --learning_rate 0.00249 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 192
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
      --enc_in 7 \
      --e_layers 2 \
      --n_heads 4 \
      --d_model 80 \
      --d_ff 360 \
      --dropout 0.10300000000000001\
      --fc_dropout 0.108\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --layers_num 1 \
      --hidden_channels 86 \
      --momentum 0.89 \
      --out_channels 126 \
      --stride1 2 \
      --stride2 4 \
      --stride3 7 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4 \
      --itr 1 \
      --batch_size 118 \
      --momentum 0.84 \
      --learning_rate 0.00272 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 336
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
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 2 \
      --d_model 28 \
      --d_ff 482 \
      --dropout 0.187\
      --fc_dropout 0.17300000000000001\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --layers_num 1 \
      --hidden_channels 82 \
      --momentum  0.96 \
      --out_channels 107 \
      --stride1 3 \
      --stride2 4 \
      --stride3 5 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4 \
      --itr 1 \
      --batch_size 134 \
      --momentum 0.84 \
      --learning_rate 0.002770 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 720
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
      --enc_in 7 \
      --e_layers 2 \
      --n_heads 2 \
      --d_model 56 \
      --d_ff 314 \
      --dropout 0.258\
      --fc_dropout 0.365\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --layers_num 1 \
      --hidden_channels 27 \
      --momentum 0.89 \
      --out_channels 127 \
      --stride1 2 \
      --stride2 4 \
      --stride3 6 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4 \
      --itr 1 \
      --batch_size 105 \
      --momentum 0.81 \
      --learning_rate 0.00103 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done