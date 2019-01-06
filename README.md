
Synchronous Bidirectional Nueral Machine Translaiton
===

This is the official codebase for the following paper, implemented in tensorflow:

Long Zhou, Jiajun Zhang, Chengqing Zong. **Synchronous Bidirectional Nueral Machine Translaiton.** In Transactions of ACL 2019.



Requirements
---
1. python2.7
2. tensorflow >=1.4
3. cuda >=8.0

Usage
---
1. Construct pseudo training data using [Transformer](https://github.com/wszlong/transformer) as introduced in the paper. 
2. Preprocessing. run `./datagen.sh`.
3. Training. run `./train.sh`.
4. Inference. run `./test.sh`.

## Citation
If you found this code useful in your research, please cite:
<pre><code>@article{Zhou:2019:TACL,
  author    = {Zhou, Long and Zhang, Jiajun and Zong, Chengqing},
  title     = {Synchronous Bidirectional Nueral Machine Translaiton},
  journal   = {Transactions of the Association for Computational Linguistics},
  year      = {2019},
}
</code></pre>

Contact
---
If you have questions, suggestions and bug reports, please email wszlong@gmail.com.

