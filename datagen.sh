

python run.py \
		--generate_data=True \
		--data_dir=./data-sb \
		--tmp_dir=/data/lzhou/t2t/interactive-decoding/c2e/data-trans \
		--train_src_name=2m.bpe.unk.zh \
		--train_tgt_name=2m.bpe.unk.en \
		--vocab_src_size=30720 \
		--vocab_tgt_size=30720 \
		--vocab_src_name=vocab.bpe.zh \
		--vocab_tgt_name=vocab.bpe.en \
		--num_shards=50 

