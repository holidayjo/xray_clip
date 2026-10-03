Work to do.
* (20260627_to_do) To try.
  - Extract from claude or gemini that combining impression and findings and outputs the appropriate text for training. And then apply the zero shot training and apply the few shot learning and then test on the testset.
* (20260927_note) Few-shot learning. 
  - We can apply few-shot learning concept on LG tumor classification. It is a perfect application area.
  - Zero-shot has a limitation. It is hard to overcome the ViT performance.
  - CheXzero is not a complete zero-shot. There are papers addressing this issue. It is now a common issue.
  - We found that there are class name explicitly mentioned in the reports. And we also found that the more class named mentioned in the paper, the better performance.
  - We can now apply a few-shot learning. 
  - Is this few-shot learning affected by class imbalacement? Then we should solve this issue.
* (20260927_note) Foundation model
  - Foundation model problem - we train the model using the same domainn dataset for train, validation, and test. 
  - For example, we train the model on mimic cxr dataset and validate and test on the same dataset.
  - However, can we apply the model on other hospital dataset?
  - Different from ChestX-ray14 dataset only results, we train the model using mimic-cxr dataset, and use a few-shot learning and compare the result from the existing results.

  
* creating images features in cache to boost training speed. - Done
* adaptor design upate. (basic nn architecture approach for now.) - Done - Not working.
  * Adding skip-connection.
* Checking how the results change with different prompt - Not working.
* Zero-shot by applying different recent CLIP model - Not working.
* Previous better test results for overall accuracy. (JW) - 
* Metric for (1) only one class, (2) 2 classes or more. - Not working.

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