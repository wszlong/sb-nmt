
export CUDA_VISIBLE_DEVICES=3

python run.py \
		--gpu_mem_fraction=0.98 \
		--hparams='' \
		--data_dir=/data/lzhou/t2t/interactive-decoding/c2e/data-trans \
		--hparams_set=transformer_params_big \
		--output_dir=/data/lzhou/t2t/interactive-decoding/c2e/train-tanh \
		--vocab_src_size=30720  \
		--vocab_tgt_size=30720  \
		--vocab_src_name=vocab.bpe.zh \
		--vocab_tgt_name=vocab.bpe.en \
		--train_steps=0 \
		--decode_beam_size=4 \
		--decode_alpha=0.6 \
		--decode_batch_size=50  \
		--decode_from_file=../data/03.seg.bpe \
		--decode_to_file=./output/dev.sbsg.tmp.out 

