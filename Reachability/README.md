### Code for Section 4.4
`python generalDAG_train.py --mode [full-scratchpad/no-scratchpad] --vertex-num [any int below 40]` will do the job.

A folder `generalDAG_output` will be created.

If your environment does not support `torch.compile`, set `torch_compile=False` in `TrainingArguments` 
