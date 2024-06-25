# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MDMIN

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 12 \
  --e_layers 3 \
  --n_heads 2 \
  --d_model 72 \
  --d_ff 139 \
  --dropout 0.281\
  --fc_dropout 0.293\
  --head_dropout 0\
  --layers_num 4 \
  --hidden_channels 34 \
  --momentum 0.87 \
  --out_channels 40 \
  --stride1 3 \
  --stride2 3 \
  --stride3 5 \
  --learning_rate 0.007050 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'96.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 8 \
  --e_layers 3 \
  --n_heads 2 \
  --d_model 56 \
  --d_ff 248 \
  --dropout 0.307\
  --fc_dropout 0.399\
  --layers_num 3 \
  --hidden_channels 39 \
  --momentum 0.85 \
  --out_channels 30 \
  --stride1 2 \
  --stride2 3 \
  --stride3 5 \
  --learning_rate 0.00780 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'192.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 10  \
  --e_layers 3 \
  --n_heads 3 \
  --d_model 114 \
  --d_ff 119 \
  --dropout 0.158\
  --fc_dropout 0.331\
  --layers_num 1 \
  --hidden_channels 21 \
  --momentum 0.8200000000000001 \
  --out_channels 52 \
  --stride1 3 \
  --stride2 3 \
  --stride3 5 \
  --learning_rate 0.00936 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'336.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 29 \
  --e_layers 2 \
  --n_heads 2 \
  --d_model 76 \
  --d_ff 399 \
  --dropout 0.289\
  --fc_dropout 0.336\
  --layers_num 4 \
  --hidden_channels 45 \
  --momentum 0.84 \
  --out_channels 35 \
  --stride1 4 \
  --stride2 4 \
  --stride3 5 \
  --learning_rate 0.00479 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'720.log
