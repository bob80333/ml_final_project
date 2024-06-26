from pathlib import Path

import soundfile as sf  # fast crossplatform audio loading
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class CVSS_T(Dataset):
    def __init__(self, main_folder, segment_length=2**14, random_segment=True):
        # cvss_folder: Path to the folder containing the CVSS audio files, e.g. 'data/cvss/train'
        # cv_folder: Path to the folder containing the original commonvoice audio files, e.g. 'data/cv/clips'
        # since cvss is already split into train, dev, test, I will reuse the same split.
        # the original CV data has no split, but the clips have the same name, so I will just replace the path
        self.english_folder = main_folder + "/english"
        self.german_folder = main_folder + "/german"
        # CVSS came with .wav files, CV data was converted to wav for faster loading (mp3 decoding is kinda slow)
        self.english_audio = [x.absolute() for x in Path(self.english_folder).rglob('*.wav')]
        # since we're using pairs and just replacing the folder we only need 1 list

        self.random_segment = random_segment
        self.segment_length = segment_length
        
    def __len__(self):
        return len(self.english_audio)
    
    def __getitem__(self, idx):
        english_file = self.english_audio[idx]
        german_file = str(english_file).replace(self.english_folder, self.german_folder)
        
        
        
        english_audio = sf.read(english_file)[0]
        german_audio = sf.read(german_file)[0]
        
        if self.random_segment:
            start_percent = torch.rand((1,))[0]
            
            english_start = int(start_percent * (len(english_audio) - self.segment_length))
            english_audio = english_audio[english_start:english_start+self.segment_length]
            
            german_start = int(start_percent * (len(german_audio) - self.segment_length))
            german_audio = german_audio[german_start:german_start+self.segment_length]
            
        else:
            english_audio = english_audio[:self.segment_length]
            german_audio = german_audio[:self.segment_length]
            
        english_audio = torch.tensor(english_audio).unsqueeze(0).float()
        german_audio = torch.tensor(german_audio).unsqueeze(0).float()
        
        # pad if not long enough
        
        english_audio = F.pad(english_audio, (0, max(self.segment_length - english_audio.shape[1], 0)))
        german_audio = F.pad(german_audio, (0, max(self.segment_length - german_audio.shape[1], 0)))
                    
   
        return english_audio, german_audio, torch.LongTensor([idx])
        
        
if __name__ == "__main__":
    dataset = CVSS_T('data/cvss_t_de_en_v1.0/train')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for english_audio, german_audio, idx in dataloader:
        # should be of same shape, [8, 1, 16384]
        # 8 is batch size, 1 is # of channels in audio, and 16384 is the segment length (2**14)
        print(english_audio.shape, german_audio.shape)
        break
