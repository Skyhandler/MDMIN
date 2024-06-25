if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MDMIN

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

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
      --enc_in 21 \
      --e_layers 4 \
      --n_heads 3 \
      --d_model 30 \
      --d_ff 162 \
      --dropout 0.16\
      --fc_dropout 0.341\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --layers_num 3 \
      --hidden_channels 23 \
      --momentum 0.88 \
      --out_channels 66 \
      --stride1 2 \
      --stride2 4 \
      --stride3 5 \
      --itr 1\
      --batch_size 103 \
      --learning_rate 0.00553 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --enc_in 21 \
      --e_layers 4 \
      --n_heads 3 \
      --d_model 30 \
      --d_ff 162 \
      --dropout 0.16\
      --fc_dropout 0.341\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --layers_num 3 \
      --hidden_channels 23 \
      --momentum 0.88 \
      --out_channels 66 \
      --stride1 2 \
      --stride2 4 \
      --stride3 5 \
      --itr 1\
      --batch_size 103 \
      --learning_rate 0.00553 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --enc_in 21 \
      --e_layers 4 \
      --n_heads 3 \
      --d_model 30 \
      --d_ff 162 \
      --dropout 0.16\
      --fc_dropout 0.341\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --layers_num 3 \
      --hidden_channels 23 \
      --momentum 0.88 \
      --out_channels 66 \
      --stride1 2 \
      --stride2 4 \
      --stride3 5 \
      --itr 1\
      --batch_size 103 \
      --learning_rate 0.00553 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --enc_in 21 \
      --e_layers 1 \
      --n_heads 1 \
      --d_model 12 \
      --d_ff 509 \
      --dropout 0.176\
      --fc_dropout 0.10300000000000001\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --layers_num 1 \
      --hidden_channels 88 \
      --momentum 0.99 \
      --out_channels 69 \
      --stride1 2 \
      --stride2 5 \
      --stride3 5 \
      --itr 1\
      --batch_size 189 \
      --learning_rate 0.00223 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done