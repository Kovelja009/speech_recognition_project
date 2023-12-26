import torch
import json
import os
import numpy as np
import librosa
from scipy.spatial.distance import cosine
from model import SoundModel
from dtw_utils import dtw

class SR_System():
    def __init__(self, model_path='models/best.pth', ref_path='./data', commands_path='commands.json', use_normalization=False, ref_student='42-20', n_mfcc=24, sr=22050, hop_length=512, sound_duration=3.1, threshold=0.5):
        self.use_normalization = use_normalization
        self.ref_student = ref_student
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.hop_length = hop_length
        self.sound_duration = sound_duration
        self.num_frames = int(np.ceil(self.sound_duration * self.sr / self.hop_length))
        self.threshold = threshold
        self.ref_sound_dict = {}
        self._load_commands(commands_path)
        self._load_model(model_path)
        self.is_new_sound = False
        self._load_ref_sounds(ref_path)
        

    def _load_model(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = SoundModel(num_classes=len(self.commands_dict))
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()

    def _load_commands(self, commands_path):
        with open(commands_path, 'r') as file:
            self.commands_dict = json.load(file)

    def _load_ref_sounds(self, ref_path):
        for filename in os.listdir(ref_path):        
            if filename.endswith('.wav'):
                codes = filename.split('.')[0].split('-')
                command = codes[0]
                index_year = codes[1] + '-' + codes[2]

                # check if it is my voice
                ref_sound = index_year == self.ref_student

                # loading reference sound
                if ref_sound:                    
                    # load sound file
                    file_path = os.path.join(ref_path, filename)
                    audio_data, _ = librosa.load(file_path)

                    preprocessed_sound = self.preprocess(audio_data, loading_first_time=True)

                    # add to dictionary of reference sounds
                    if command not in self.ref_sound_dict:
                        self.ref_sound_dict[command] = []

                    self.ref_sound_dict[command].append(preprocessed_sound)

    def _idx_to_cmd(self, idx):
        for cmd, cmd_idx in self.commands_dict.items():
            if cmd_idx == idx:
                return cmd
        return None


    def predict(self, audio_data):
        preprocessed_sound = self.preprocess(audio_data)
        command = None

        if self.is_new_sound: # using dtw because our model is not trained on new sound
            command = self.dtw_predict(preprocessed_sound)
        else: # using model
            prediction = self.model(preprocessed_sound)
            command = self._idx_to_cmd(torch.argmax(prediction, dim=1))

        return command

    def add_new_sound(self, audio_data, label):
        self.is_new_sound = True
        self.commands_dict[label] = len(self.commands_dict)
        self.ref_sound_dict[label] = [self.preprocess(audio_data)]

    def dtw_predict(self, processed_sound):
        min_cost = 100000
        min_command = None

        for command, lst in self.ref_sound_dict.items():
            cum_cost = 0
            for ref_sound in lst:
                cost, _ = dtw(processed_sound, ref_sound, lambda x, y: cosine(x, y))
                cum_cost += cost
            cum_cost /= len(lst)
            print(f'--Cost for {command}: {cum_cost}--')
            if cum_cost < min_cost:
                min_cost = cum_cost
                min_command = command

        print(f'MIN Cost: {min_cost} for command: {min_command}')

        print(self.threshold)
        if min_cost > self.threshold:
            return 'Unknown sound'
        else:
            return min_command


    def preprocess(self, audio_data, loading_first_time=False):
        # preprocessing
        mfccs = librosa.feature.mfcc(y=audio_data, n_fft=2048, hop_length=self.hop_length, n_mfcc=self.n_mfcc)
        
        # Pad or truncate MFCCs to a fixed length (e.g., self.num_frames)
        if mfccs.shape[1] > self.num_frames:
            mfccs = mfccs[:, :self.num_frames]
        elif mfccs.shape[1] < self.num_frames:
            mfccs = np.pad(mfccs, ((0, 0), (0, self.num_frames - mfccs.shape[1])), mode='constant')

        if self.use_normalization:
            # Calculate mean and standard deviation along the frame axis
            mean = np.mean(mfccs, axis=1, keepdims=True)
            std = np.std(mfccs, axis=1, keepdims=True)

            # Small value to avoid division by zero or near-zero
            epsilon = 1e-6

            # Avoid normalization when std is close to zero
            std_safe = np.where(std < epsilon, epsilon, std)

            # Apply normalization
            mfccs = (mfccs - mean) / std_safe

        # means we are running model inference
        if self.is_new_sound == False and not loading_first_time == True:
            mfccs = torch.tensor(mfccs, dtype=torch.float32)
            # add batch dimension
            mfccs = mfccs.unsqueeze(0)
            mfccs = mfccs.to(self.device)

        return mfccs


if __name__ == '__main__':
    sr_system = SR_System()

    print(len(sr_system.ref_sound_dict['krug']))
    print(len(sr_system.ref_sound_dict['kvadrat']))




#     # using data for labels that model was trained on
#     gt = 'krug'
#     audio_data, _ = librosa.load(f'./test_data/{gt}-42-20-3.wav')
#     command = sr_system.predict(audio_data)
#     print(f'Predicted: {command}, True: {gt}')

#     gt = 'kvadrat'
#     audio_data, _ = librosa.load(f'./test_data/{gt}-42-20-1.wav')
#     command = sr_system.predict(audio_data)
#     print(f'Predicted: {command}, True: {gt}')



# ####### now we are in dtw mode because we added new sound #######
#     # adding new sound
#     gt = 'povecaj'
#     audio_data, _ = librosa.load(f'./additional_command/{gt}-42-20-1.wav')
#     sr_system.add_new_sound(audio_data, 'povecaj')

#     print(f'All commands: {sr_system.commands_dict}')
#     print('-----------Now we are in dtw mode-----------')
    
#     # testing new sound
#     command = sr_system.predict(audio_data)
#     print(f'Predicted: {command}, True: {gt}')

#     #testing old sounds
#     gt = 'krug'
#     audio_data, _ = librosa.load(f'./test_data/{gt}-42-20-1.wav')
#     command = sr_system.predict(audio_data)
#     print(f'Predicted: {command}, True: {gt}')


    



