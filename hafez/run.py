from preprocessor import QuranPreprocessor
import logging
import torch
from helpers import (prepare_dataset,
                     move_to_device,
                     setting)
from train import QuranTrainer


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s'
                    )
logger = logging.getLogger(__name__)


data = QuranPreprocessor(dataset_dir=setting.dataset_dir,
                         train_file=setting.train_data_path,
                         test_file=setting.test_data_path
                         )
data.load_quran_dataset()
logger.info("Quran dataset has been loaded!")
data.remove_diacritics()
logger.info("Diacritics have been removed!")


data.train_dataset = data.train_dataset.map(
    data.speech_file_to_array_fn
)
logger.info("Train data has been mapped to array!")

data.test_dataset = data.test_dataset.map(
    data.speech_file_to_array_fn
)
logger.info("Test data has been mapped to array!")

data.train_dataset = data.train_dataset.map(prepare_dataset,
                                                    remove_columns=data.train_dataset.column_names,
                                                    batch_size=setting.batch_size,
                                                    num_proc=setting.num_proc,
                                                    batched=True
                                                    )
logger.info("Training data has been prepared!")

data.test_dataset = data.test_dataset.map(prepare_dataset,
                                                    remove_columns=data.test_dataset.column_names,
                                                    batch_size=setting.batch_size,
                                                    num_proc=setting.num_proc,
                                                    batched=True
                                                    )
logger.info("Testing data has been prepared!")

# if you have a gpu
# data.train_dataset = data.train_dataset.map(move_to_device, batched=True)
# data.test_dataset = data.test_dataset.map(move_to_device, batched=True)

trainer = QuranTrainer(data.train_dataset, data.test_dataset)
logger.info("Trainer object created!")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer.model = trainer.model.to(device)
torch.cuda.empty_cache()
trainer.train()