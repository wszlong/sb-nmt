
Code for SB-NMT
===
This is the implementation of the TACL paper: Synchronous Bidirectional Nueral Machine Translaiton. 

If you use the code, please cite our paper:

<pre><code>@article{Zhou:2019:TACL,
  author    = {Zhou, Long and Zhang, Jiajun and Zong, Chengqing},
  title     = {Synchronous Bidirectional Nueral Machine Translaiton},
  journal   = {Transactions of the Association for Computational Linguistics},
  year      = {2019},
}
</code></pre>

Requirements
---
1. python2.7
2. tensorflow >=1.4
3. cuda >=8.0

Usage
---
1. Construct pseudo training data.
2. Preprocessing. run './datagen.sh'.
3. Training. run './train.sh'.
4. Inference. run './test.sh'.

Contact
---
If you have questions, suggestions and bug reports, please email wszlong@gmail.com.

