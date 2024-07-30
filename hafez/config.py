class Setting:
    pre_trained_model = "facebook/wav2vec2-large-xlsr-53"
    repo_id="mohammed/quran"
    save_dir="quran"
    dataset_dir = "D:\\ML\\Dataset\\"
    train_data_path = "whisper_train.csv"
    test_data_path = "whisper_test.csv"
    sampling_rate=16000
    feature_size=1
    padding_value=0.0
    do_normalize=True
    return_attention_mask=True
    batch_size=8
    num_proc=4
    do_normalize_eval = True
    
    # model parameters
    attention_dropout=0.1
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    
    # training arguments
    group_by_length=True,
    per_device_train_batch_size=2,  # Minimize batch size
    gradient_accumulation_steps=16,  # Increase to maintain effective batch size
    evaluation_strategy="steps",
    num_train_epochs=15,
    fp16=True,
    save_steps=250,
    save_total_limit=2,
    eval_steps=250,
    logging_steps=250,
    learning_rate=5e-6,  # Adjusted learning rate
    warmup_steps=500,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    
    # Trainer arguments
    esp=5