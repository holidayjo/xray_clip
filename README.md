Work to do.
* creating images features in cache to boost training speed.
* adaptor design upate. (basic nn architecture approach for now.)
  * Adding skip-connection.
* Checking how the results change with different prompt
* Zero-shot by applying different recent CLIP model
* Previous better test results for overall accuracy. (JW)
* Metric for (1) only one class, (2) 2 classes or more.

DONEs
* code check (inference phase) - Done 
* training curve check (with loss and val set results) - Done
* In main.ipynb,what does load_clip_model actually load? - Done


Meeting on 20260907
- train on mimic dataset --> test on chest 14 dataset (our old dataset)
- after that we can going to the llm
- svip q2 q3 target

Meeting on 20260914
- prompt change: "a photo of ...", 부정관사 확인.
- cos similarity
- (in training) understanding contrastive learning with shapes of each tensor
- (in inference) understaning constrastive learning with shapes of each tensor and if it finally outputs the probability.
- a better few labels are okay