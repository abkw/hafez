from transformers import (Wav2Vec2ForCTC,
                          Wav2Vec2Processor,
                          TrainingArguments,
                          Trainer,
                          EarlyStoppingCallback
                          )
import numpy as np
import evaluate
from helpers import get_processor, setting
from data_collector import DataCollatorCTCWithPadding as DCTC


class QuranTrainer:
    def __init__(self, train_data, test_data) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.processor = get_processor()
        self.vocab_size = len(self.processor.tokenizer)
        self.get_model()
        print(f"Vocab size: {self.vocab_size}")

    def get_model(self):
        self.model = Wav2Vec2ForCTC.from_pretrained(
            setting.pre_trained_model,
            attention_dropout=setting.attention_dropout,
            hidden_dropout=setting.hidden_dropout,
            feat_proj_dropout=setting.feat_proj_dropout,
            mask_time_prob=setting.mask_time_prob,
            layerdrop=setting.layerdrop,
            gradient_checkpointing=setting.gradient_checkpointing,
            ctc_loss_reduction=setting.ctc_loss_reduction,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=self.vocab_size
        )
        self.model.freeze_feature_extractor()

    def get_train_args(self):
        training_args = TrainingArguments(
            setting.repo_id,
            group_by_length=setting.group_by_length,
            per_device_train_batch_size=setting.per_device_train_batch_size,
            gradient_accumulation_steps=setting.gradient_accumulation_steps,
            evaluation_strategy=setting.evaluation_strategy,
            num_train_epochs=setting.num_train_epochs,
            fp16=setting.fp16,
            save_steps=setting.save_steps,
            save_total_limit=setting.save_total_limit,
            eval_steps=setting.eval_steps,
            logging_steps=setting.logging_steps,
            learning_rate=setting.learning_rate,
            warmup_steps=setting.warmup_steps,
            weight_decay=setting.weight_decay,
            load_best_model_at_end=setting.load_best_model_at_end,
            metric_for_best_model=setting.metric_for_best_model,
            greater_is_better=setting.greater_is_better,
            push_to_hub=setting.push_to_hub,
            )
        
        return training_args
    
    def compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids,
                                                group_tokens=False
                                                )
        metric = evaluate.load("wer")
        wer = metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def get_trainer(self):
        esp = EarlyStoppingCallback(early_stopping_patience=setting.esp)
        data_collator = DCTC(processor=get_processor, padding=True)
        
        return Trainer(
            model=self.model,
            data_collator=data_collator,
            args=self.get_train_args(),
            compute_metrics=self.compute_metrics,
            train_dataset=self.train_data,
            eval_dataset=self.test_data,
            tokenizer=self.processor.feature_extractor,
            callbacks=[esp]
        )