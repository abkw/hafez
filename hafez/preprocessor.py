from datasets import load_dataset
import re
import torchaudio
import tnkeeh as tn

class QuranPreprocessor:
    # a class for preprocessing quran data
    def __init__(self, dataset_dir, train_file, test_file):
        self.dataset_dir = dataset_dir
        self.train_file = train_file
        self.test_file = test_file
    
    def load_quran_dataset(self):
        # load the dataset
        data_files = {"train": self.dataset_dir + self.train_file,
                      "test": self.dataset_dir + self.test_file}
        dataset = load_dataset(self.dataset_dir, data_files=data_files)
        self.train_dataset = dataset["train"]
        self.test_dataset = dataset["test"]
    
    def remove_diacritics(self):
        # prepare the cleaner to clean with Tnkeeh
        cleaner = tn.Tnkeeh(remove_diacritics=True, remove_english=True)
        
        self.train_dataset = cleaner.clean_hf_dataset(self.train_dataset,
                                                       'transcription'
                                                       )
        self.test_dataset = cleaner.clean_hf_dataset(self.test_dataset,
                                                      'transcription'
                                                      )
        
        # remove unwanted characters
        self.train_dataset = self.train_dataset.map(self.remove_special_characters)
        self.test_dataset = self.test_dataset.map(self.remove_special_characters)
    
    def remove_special_characters(self, batch):
    # creating a dictionary with all diacritics
        dict = {
        'ِ': '',
        'ُ': '',
        'ٓ': '',
        'ٰ': '',
        'ْ': '',
        'ٌ': '',
        'ٍ': '',
        'ً': '',
        'ّ': '',
        'َ': '',
        '~': '',
        ',': '',
        'ـ': '',
        '—': '',
        '.': '',
        '!': '',
        '-': '',
        ';': '',
        ':': '',
        '\'': '',
        '"': '',
        '☭': '',
        '«': '',
        '»': '',
        '؛': '',
        'ـ': '',
        '_': '',
        '،': '',
        '“': '',
        '%': '',
        '‘': '',
        '”': '',
        '�': '',
        '_': '',
        ',': '',
        '?': '',
        '#': '',
        '‘': '',
        '.': '',
        '؛': '',
        'get': '',
        '؟': '',
        '\'ۖ ': '',
        '\'': ''
        }
        # Create a regular expression  from the dictionary keys
        regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
        # For each match, look-up corresponding value in dictionary
        batch["sentence"] = regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], batch["transcription"])

        return batch

    def speech_file_to_array_fn(self, batch):
        resamplers = {
            48000: torchaudio.transforms.Resample(48000, 16000),
            44100: torchaudio.transforms.Resample(44100, 16000),
            32000: torchaudio.transforms.Resample(32000, 16000),
            16000: torchaudio.transforms.Resample(16000, 16000),
        }
        speech_array, sampling_rate = torchaudio.load(batch["file_name"])
        batch["speech"] = resamplers[sampling_rate](speech_array).squeeze().numpy()
        batch["sampling_rate"] = 16_000
        batch["target_text"] = batch["sentence"]

        return batch