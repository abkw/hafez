import os
from transformers import (Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor)
import torch
from config import Setting

setting = Setting()

def get_processor():
    tokenizer = Wav2Vec2CTCTokenizer("vocab.json",
                                    unk_token="[UNK]",
                                    pad_token="[PAD]",
                                    word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(setting.feature_size,
                                                sampling_rate=setting.sampling_rate,
                                                padding_value=setting.padding_value,
                                                do_normalize=setting.do_normalize,
                                                return_attention_mask=setting.return_attention_mask
                                                )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                tokenizer=tokenizer)
    processor.save_pretrained(os.getenv(os.getenv("SAVE_DIR")))

    return processor

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    processor = get_processor()
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids

    return batch

def move_to_device(batch):
    # this is an optional function to move training data to gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch