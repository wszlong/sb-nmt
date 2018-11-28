
export CUDA_VISIBLE_DEVICES=3

nohup python run.py  \
		--worker_gpu=1 \
		--gpu_mem_fraction=0.9 \
		--hparams='batch_size=2048'  \
		--data_dir=/data/lzhou/t2t/interactive-decoding/c2e/data-trans  \
		--vocab_src_size=30720  \
		--vocab_tgt_size=30720  \
		--vocab_src_name=vocab.bpe.zh \
		--vocab_tgt_name=vocab.bpe.en \
		--hparams_set=transformer_params_big  \
		--train_steps=200000  \
		--keep_checkpoint_max=2  \
		--output_dir=./train-sb > log.train-sb &


