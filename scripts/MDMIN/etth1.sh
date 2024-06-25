if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MDMIN

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

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
      --e_layers 2 \
      --n_heads 2 \
      --d_model 32 \
      --d_ff 80 \
      --dropout 0.193   \
      --fc_dropout 0.332\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 \
      --batch_size 201 \
      --layers_num 2 \
      --hidden_channels 98 \
      --momentum 0.84 \
      --out_channels 114 \
      --stride1 2 \
      --stride2 4 \
      --stride3 6 \
      --learning_rate 0.000740 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --e_layers 1 \
      --n_heads 1 \
      --d_model 10 \
      --d_ff 195 \
      --dropout 0.227 \
      --fc_dropout 0.157\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 \
      --batch_size 181 \
      --layers_num 2 \
      --hidden_channels 86 \
      --momentum 0.86 \
      --out_channels 68 \
      --stride1 2 \
      --stride2 4 \
      --stride3 5 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 8\
      --lradj 'TST'\
      --pct_start 0.4\
      --learning_rate 0.009470 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --n_heads 2 \
      --d_model 32 \
      --d_ff 80 \
      --dropout 0.193   \
      --fc_dropout 0.332\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 \
      --batch_size 201 \
      --layers_num 2 \
      --hidden_channels 98 \
      --momentum 0.84 \
      --out_channels 114 \
      --stride1 2 \
      --stride2 4 \
      --stride3 6 \
      --learning_rate 0.000740 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --d_model 24 \
      --d_ff 120 \
      --dropout 0.20800000000000002   \
      --fc_dropout 0.379\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 \
      --batch_size 139 \
      --layers_num 2 \
      --hidden_channels 16 \
      --momentum 0.9500000000000001 \
      --out_channels 84 \
      --stride1 2 \
      --stride2 3 \
      --stride3 5 \
      --learning_rate 0.006120 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done