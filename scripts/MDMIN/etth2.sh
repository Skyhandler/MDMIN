if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MDMIN

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

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
        --features  M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers 3 \
        --n_heads 1 \
        --d_model 16 \
        --d_ff 108 \
        --dropout 0.166   \
        --fc_dropout 0.17200000000000001\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --itr 1 \
        --batch_size 202 \
        --layers_num 1 \
        --hidden_channels 125 \
        --momentum 0.9 \
        --out_channels 17 \
        --stride1 3 \
        --stride2 5 \
        --stride3 7 \
        --learning_rate 0.00136 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
        --features  M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers 3 \
        --n_heads 3 \
        --d_model 54 \
        --d_ff 129 \
        --dropout 0.404   \
        --fc_dropout 0.427\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --itr 1 \
        --batch_size 245 \
        --layers_num 1 \
        --hidden_channels 88 \
        --momentum 0.9400000000000001 \
        --out_channels 25 \
        --stride1 4 \
        --stride2 4 \
        --stride3 5 \
        --learning_rate 0.001700>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
        --features  M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers 2 \
        --n_heads 3 \
        --d_model 36 \
        --d_ff 462 \
        --dropout 0.246   \
        --fc_dropout 0.164\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --itr 1 \
        --batch_size 256 \
        --layers_num 4 \
        --hidden_channels 16 \
        --momentum 0.8200000000000001 \
        --out_channels 120 \
        --stride1 3 \
        --stride2 5 \
        --stride3 7 \
        --learning_rate 0.00051 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
        --features  M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers 2 \
        --n_heads 4 \
        --d_model 40 \
        --d_ff 343 \
        --dropout 0.276   \
        --fc_dropout 0.254 \
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --itr 1 \
        --batch_size 242 \
        --layers_num 1 \
        --hidden_channels 77 \
        --momentum 0.84 \
        --out_channels 53 \
        --stride1 3 \
        --stride2 5 \
        --stride3 6 \
        --learning_rate 0.0002 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done