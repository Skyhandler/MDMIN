if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MDMIN

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

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
      --d_model 28 \
      --d_ff 471 \
      --dropout 0.402   \
      --fc_dropout 0.19\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 \
      --batch_size 63 \
      --layers_num 2 \
      --hidden_channels 82 \
      --momentum 0.92 \
      --out_channels 45 \
      --stride1 3 \
      --stride2 3 \
      --stride3 6 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 8\
      --lradj 'TST'\
      --pct_start 0.4\
      --learning_rate 0.00058 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --e_layers 3 \
      --n_heads 2 \
      --d_model 24 \
      --d_ff 380 \
      --dropout 0.417  \
      --fc_dropout 0.15 \
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 \
      --batch_size 210 \
      --layers_num 1 \
      --hidden_channels 122 \
      --momentum 0.98 \
      --out_channels 126 \
      --stride1 3 \
      --stride2 5 \
      --stride3 6 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 8\
      --lradj 'TST'\
      --pct_start 0.4\
      --learning_rate 0.00462 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --e_layers 2 \
      --n_heads 1 \
      --d_model 26 \
      --d_ff 445 \
      --dropout 0.327   \
      --fc_dropout 0.448 \
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 \
      --batch_size 38 \
      --layers_num 3 \
      --hidden_channels 127 \
      --momentum 0.88 \
      --out_channels 42 \
      --stride1 3 \
      --stride2 4 \
      --stride3 7 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 8\
      --lradj 'TST'\
      --pct_start 0.4\
      --learning_rate 0.00116 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --e_layers 4 \
      --n_heads 2 \
      --d_model 28 \
      --d_ff 402 \
      --dropout 0.34700000000000003   \
      --fc_dropout 0.293\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 \
      --batch_size 89 \
      --layers_num 2 \
      --hidden_channels 56 \
      --momentum 0.89 \
      --out_channels 47 \
      --stride1 2 \
      --stride2 4 \
      --stride3 6 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 8\
      --lradj 'TST'\
      --pct_start 0.4\
      --learning_rate 0.00619 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done