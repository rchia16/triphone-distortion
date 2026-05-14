import os, shutup, sys, collections, json, math, random, time, copy, pickle, mat73, h5py, pyxdf
import string, pywt, warnings, gc, psutil, omegaconf, mne
from joblib import Parallel, delayed
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import BartTokenizer, BertTokenizer, BertConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Iterable
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Any, Dict
from sklearn.utils import compute_class_weight
import matplotlib.pyplot as plt
from numpy import linalg
from imblearn.over_sampling import ADASYN, SMOTE,BorderlineSMOTE,SVMSMOTE,RandomOverSampler
import scipy
import scipy.io
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import euclidean, chebyshev, cosine, correlation
from scipy.stats import entropy
from sklearn.discriminant_analysis import _cov
import scipy.linalg
if sys.platform != 'win32':
    import fcntl
import sklearn
import seaborn as sns
import hashlib
import yaml, json
import pathlib 
import omegaconf
from contextlib import contextmanager
# from tqdm_joblib import tqdm_joblib
shutup.please()
os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return collections.OrderedDict(
            {
                "file": ann["image"],
                "dialogue": ann["dialogue"],
                "image": sample["image"],
            }
        )
def get_tokenizer(model_name):
    if "bert" in model_name:

        tokenizer = BertTokenizer.from_pretrained(model_name)
        tokenizer.bos_token_id = 101
        tokenizer.eos_token_id = 102
    elif "bart" in model_name:
        tokenizer = BartTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("using tokenizer {}, pad_token_id {}, eos_token_id {}, bos_token_id {}".format(
        model_name, tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id))

    return tokenizer
def load_JSON_sessions(session_name, json_folder):
    json_path = os.path.join(json_folder, f"{session_name}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        print(f"Loaded JSON for session '{session_name}' from '{json_path}'")
        session_annotation_result = []
        # loop through the json_data
        for i in range(len(json_data)):
            trial_annotations = json_data[i]['annotations'][0]['result']
            trial_annotation_result= {
                'trial_id':i,
                'word_segment': {}, 
                'syllable_segment': [{}],
                'ding_segment': {},
                'file_path': json_data[i]['data']['s3_path'],
                'original_length': json_data[i]['annotations'][0]['result'][0]['original_length'],
            }
            for annotation in trial_annotations:
                # skip if the annotation don't have value:'start' and 'end'
                if 'value' not in annotation:
                    continue
                elif 'start' not in annotation['value'] or 'end' not in annotation['value']:
                    continue
                # print('annotation\n',annotation)
                '''
                    {'value': 
                    {'text': ['Other']}, 
                    'id': 'mulg5ptUGB', 
                    'from_name': 'transcription', 'to_name': 'audio', 'type': 'textarea', 'origin': 'manual'}
                '''
                annotation_label = annotation['value']['labels'][0]
                annot = {
                        'start': annotation['value']['start'],
                        'end': annotation['value']['end'],
                        'label': annotation['value']['labels'][0],
                    }

                if annotation_label == 'Word':
                    trial_annotation_result['word_segment'] = annot
                elif annotation_label == 'Ding':
                    trial_annotation_result['ding_segment']=annot
                else:# all the syllable segment
                    trial_annotation_result['syllable_segment'].append(annot)
                    
            session_annotation_result.append(trial_annotation_result)
        # print the first 10
        # for i in range(10):
        #     print(session_annotation_result[i])
        # print('debugging JOSN loader')
        # exit(0)
        return session_annotation_result
    else:
        print(f"Warning: JSON file for session '{session_name}' not found at '{json_path}'")
        return None
def _get_event_type_by_marker(marker):
    event_dict = {
        'start_session': '10',
        'end_session': '11',
        'ding_on': '12',
        'ding_off': '17',
        'start_trial': '13',
        'end_trial': '14',
        'cross_on': '15',
        'cross_off': '16',
        'relax_on': '18',
        'relax_off': '19',
        'Speech_Test': '99',
        'Speech_Jumping': '100',
        'Speech_Running': '101',
        'Speech_Swimming': '102',
        'Speech_Going': '103',
        'Speech_Happy': '104',
        'Speech_Sad': '105',
        'Speech_Fun': '106',
        'Speech_Horrible': '107',
        'Speech_College': '108',
        'Speech_Home': '109',
        'Speech_Battlefield': '110',
        'Speech_Here': '111',
        'Speech_Mother': '112',
        'Speech_Cowboy': '113',
        'Speech_Professor': '114',
        'Speech_Me': '115',
        'Speech_One': '116',
        'Speech_Three': '117',
        'Speech_Eleven': '118',
        'Speech_Million': '119',
        'Speech_Spoon': '120',
        'Speech_Alfa': '121',
        'Speech_Python': '122',
        'Speech_Telephone': '123', 
        'Cue_Test': '98',        
        'Cue_Jumping': '200',
        'Cue_Running': '201',
        'Cue_Swimming': '202',
        'Cue_Going': '203',
        'Cue_Happy': '204',
        'Cue_Sad': '205',
        'Cue_Fun': '206',
        'Cue_Horrible': '207',
        'Cue_College': '208',
        'Cue_Home': '209',
        'Cue_Battlefield': '210',
        'Cue_Here': '211',
        'Cue_Mother': '212',
        'Cue_Cowboy': '213',
        'Cue_Professor': '214',
        'Cue_Me': '215',
        'Cue_One': '216',
        'Cue_Three': '217',
        'Cue_Eleven': '218',
        'Cue_Million': '219',
        'Cue_Spoon': '220',
        'Cue_Alfa': '221',
        'Cue_Python': '222',
        'Cue_Telephone': '223',
        }
    marker_dict = {v: k for k, v in event_dict.items()}
    
    return marker_dict[marker]
def _get_number_of_utterances_by_marker(marker):
    
    
    
    pass
def _get_audio_info(session_name, json_folder, trial_id):
    json_data = load_JSON_sessions(session_name, json_folder)
    if json_data is not None:
        audio_info = json_data['audio_info']
        return audio_info
    else:
        return None
def _load_eeglab_raw(file_path):
    raw = mne.io.read_raw_eeglab(file_path, preload=True,verbose=False)  
    print('loaded ',file_path)
    return raw
def combine_curry_sessions(
    subject_data_folder,
    relevant_events,
    start=0,end=1.0,
    output_format='epoch',
    postfix='_cleaned.set',
    epoched_data= False,
    combine_sessions=False, 
    syllable_mapping=False,
    default_drop_channels = ['10', '11', '84', '85', '110', '111','VEO', 'HEO', 'EKG', 'EMG', 'Trigger'],
    n_jobs=16,
    debug=False
    ):
    json_dir_path ='/projects/SilSpeech/Spoken_EEG/Subject_syllables'
    if debug:
        print('relevant_events',relevant_events)    
    session_info = {
        'session_shape':[],
        'session_name':[],
        'session_id':[],
        'events':[],
        'num_audio':[],
    }
    subject_id = subject_data_folder.split('/')[-1]
    #list all folders starts with sess, and is folder
    set_folder= subject_data_folder
    if debug:
        print('subject_data_folder',subject_data_folder)
    if postfix is None: # no preprocess is required, using the raw session data
        cdt_files = [f for f in os.listdir(set_folder) if f.endswith('.set')]
        cdt_files.sort(key=lambda x: int(x[4:-4]))
        cdt_file_ids = [f[4:-4] for f in cdt_files]    
        set_files = [os.path.join(subject_data_folder,p) for p in cdt_files]
        all_sessions = [f.split('/')[-1] for f in set_files]
    else: # path finding is different. but will find the session folders
        # print(set_folder)
        preprocessed_eeg_folder = [] 
        for f in os.listdir(set_folder):
            # print(f)
            if os.path.isdir(os.path.join(set_folder,f)) and f.startswith('sess'):
                preprocessed_eeg_folder.append(f)
        # print(preprocessed_eeg_folder)
        preprocessed_eeg_folder.sort(key=lambda x: int(x[4:]))
        # print('preprocessed_eeg_folder',preprocessed_eeg_folder)
        set_files = []
        all_sessions = []
        for f in preprocessed_eeg_folder:
            path = os.path.join(set_folder,f)
            # if not exists, then raise
            if not os.path.exists(path):
                raise ValueError('path not exists when looing for ',path)
            set_files.append(os.path.join(set_folder,f,f+postfix))
            all_sessions.append(f)
        # print('set_files',set_files)
        if debug:
            print('all_sessions',all_sessions)
    ####################################################
    # Start processing the data to raw wave
    epoch_data = []
    to_drop = default_drop_channels
    event_map = {}
    event_map_reverse = {}
    relevant_events = [str(e) for e in relevant_events]
    for i, e in enumerate(relevant_events):
        event_map[e] = i+1
        event_map_reverse[i+1] = e
    
    ####################################################
    # Preload all raw data in parallel
    raw_data_list = Parallel(n_jobs=n_jobs)(delayed(_load_eeglab_raw)(f) for f in set_files)


    ####################################################
    # Loop through the sessions
    speech_event_count =0
    cue_event_count = 0
    for i,set_file in enumerate(set_files):
        ##############################
        # Preload the audio annotation data
        if syllable_mapping == True:
            audio_annotation_json_data = load_JSON_sessions(f"{subject_id}-session{str(i+1)}", os.path.join(json_dir_path, subject_id))
            print('json_data',len(audio_annotation_json_data))
            session_info['num_audio'].append(len(audio_annotation_json_data))
            # calculate the time of the first ding.
            # we align the audio data to the start of the cross_on event. 
            # first put all annotation to a list. 
            # get info mation like: trial id (strat from 0), ding event onset, word type, word onset, syllable list and the onset of each syllable
        else:
            audio_annotation_json_data = None
            # start_list = []
            # syllable_list = []
            # print('json_counter',json_counter)
            # for annotation in json_data[json_counter]['annotations']:
            #     for result in annotation['result']:
            #         if result['origin'] == 'manual':
            #             try:
            #                 start_list.append(result['value']['start'])
            #                 syllable_list.append(result['value']['labels'][0])
            #             except:
            #                 pass                
            # for index, syllable in enumerate(syllable_list):
            #     event_dict_syllable= {}
            #     broad_onset = float(event_dict['onset'])-float(start_list[0])
            #     event_dict_syllable['kind'] = f"Syllable_{index}"
            #     event_dict_syllable['word'] = syllable
            #     event_dict_syllable['onset'] = broad_onset+start_list[index]
            #     event_dict_syllable['audio_name'] = json_data[json_counter]['data']['s3_path']
            #     event_dict_syllable['session_id'] = i
            #     event_dict_syllable['session_name'] = all_sessions[i]
            #     event_dict_syllable['subject_name'] = subject_id
            #     event_dict_syllable['condition'] = 'speech'
            #     event_dict_syllable['description'] = -1
            #     eeg_events.append(event_dict_syllable)
            #     # json_counter+=1
        session_trial_count = 0
        session_trial_intervals = [] # contains tuples of start (13) and end (14) of each trial
        if not epoched_data: # only non-epoched data is supported at the moment             
            full_path = os.path.join(set_folder, set_file)
            # raw = mne.io.read_raw_eeglab(full_path, preload=True,verbose=False)  
            raw = raw_data_list[i] 
            events, event_id = mne.events_from_annotations(raw,verbose=False)
            ############################################
            ### Make the event dataframe
            eeg_events=list()
            marker_count = {
            }
            for annot in raw.annotations:                    
                marker = annot['description']
                onset = annot['onset']
                marker_count[marker] = marker_count.get(marker,0)+1
                if marker == 'boundary':
                    continue
                try: 
                    marker = int(marker)
                except:
                    print('marker',marker,'is not a number but not "boundary", found a critical error, exiting program')
                    exit(0)
                
                # Prepare a dictionary for describing the event
                event_dict = {}
                event_dict['semanticGroup'] = None
                event_dict['num_utterance'] = None #
                event_dict['description'] = marker
                event_dict['onset'] = onset
                try:
                    maker_event = _get_event_type_by_marker(str(marker))
                except:
                    # print('marker',marker,'is not in the event list')
                    # print(annot)
                    continue
                # find start of trial event
                if maker_event == 'start_trial':
                    session_trial_count+=1
                    # we can put the audio annotation here. 

                # Cue or action. 
                if 'speech' in maker_event.lower() or 'cue' in maker_event.lower():                    
                    event_dict['kind'] = 'word'
                    word = maker_event.split('_')[1]
                    event_dict['word'] = word
                    if 'speech' in maker_event.lower():
                        event_dict['condition'] = 'speech'
                        speech_event_count+=1
                        # word-semantic group mapping
                        if word in ['Jumping','Running','Swimming','Going']:
                            event_dict['semanticGroup'] = 'speech_Motion'
                        elif word in ['Happy','Sad','Fun','Horrible']:
                            event_dict['semanticGroup'] = 'speech_Emotion'
                        elif word in ['College','Home','Battlefield','Here']:
                            event_dict['semanticGroup'] = 'speech_Location'
                        elif word in ['Mother','Cowboy','Professor','Me']:
                            event_dict['semanticGroup'] = 'speech_Person'
                        elif word in ['One','Three','Eleven','Million']:
                            event_dict['semanticGroup'] = 'speech_Number'
                        elif word in ['Spoon','Alfa','Python','Telephone']:
                            event_dict['semanticGroup'] = 'speech_Object'
                        # word - number of utterance mapping
                        if word in ['Sad','Fun','Home','Me','One','Three','Spoon']:
                            event_dict['num_utterance'] = 1
                        elif word in ['Horrible','College','Battlefield','Professor','Eleven','Million','Telephone']:
                            event_dict['num_utterance'] = 3
                        elif word in ['Jumping','Running','Swimming','Going','Happy','Here','Mother','Cowboy','Alfa','Python']:
                            event_dict['num_utterance'] = 2
                    elif 'cue' in maker_event.lower():
                        event_dict['condition'] = 'read'
                        cue_event_count+=1
                        # print('>>>>>>>>>>>>>>>>> found cue',cue_event_count,'marker maker_event',maker_event,annot)
                        if word in ['Jumping','Running','Swimming','Going']:
                            event_dict['semanticGroup'] = 'read_Motion'
                        elif word in ['Happy','Sad','Fun','Horrible']:
                            event_dict['semanticGroup'] = 'read_Emotion'
                        elif word in ['College','Home','Battlefield','Here']:
                            event_dict['semanticGroup'] = 'read_Location'
                        elif word in ['Mother','Cowboy','Professor','Me']:
                            event_dict['semanticGroup'] = 'read_Person'
                        elif word in ['One','Three','Eleven','Million']:
                            event_dict['semanticGroup'] = 'read_Number'
                        elif word in ['Spoon','Alfa','Python','Telephone']:
                            event_dict['semanticGroup'] = 'read_Object'           

                        # word - number of utterance mapping
                        if word in ['Sad','Fun','Home','Me','One','Three','Spoon']:
                            event_dict['num_utterance'] = 1
                        elif word in ['Horrible','College','Battlefield','Professor','Eleven','Million','Telephone']:
                            event_dict['num_utterance'] = 3
                        elif word in ['Jumping','Running','Swimming','Going','Happy','Here','Mother','Cowboy','Alfa','Python']:
                            event_dict['num_utterance'] = 2

                else:                    
                    event_dict['kind'] = 'exp'
                    event_dict['word'] = None
                    event_dict['condition'] = None
                    if maker_event == 'cross_on':
                        event_dict['condition'] = 'eye'
                    if maker_event == 'relax_on':
                        event_dict['condition'] = 'relax'
                    if maker_event == 'ding_on':    
                        event_dict['condition'] = 'ding'
                event_dict['session_id'] = i
                event_dict['session_name'] = all_sessions[i]
                event_dict['subject_name'] = subject_id
                event_dict['audio_name'] = None
                eeg_events.append(event_dict)

            # make dataframe 
            eeg_events_df = pd.DataFrame(eeg_events)
            # print('eeg_events_df\n',eeg_events_df.head())
            ############################################
            ### Finish making the event dataframe
            ############################################
            ### Do a mapping in mne event id to ensure session consistency
            reverse_mapping = {v: k for k, v in event_id.items()}
            reformed_events = []
            for j,e in enumerate(events):
                if str(reverse_mapping[e[2]]) not in relevant_events:
                    pass
                else:
                    reformed_events.append(e)
            for j,e in enumerate(reformed_events):
                e_label = reverse_mapping[e[2]]
                e[2] = event_map[e_label]
            reformed_events = np.array(reformed_events)
            
            filtered_event_id = {key: event_map[key] for key in relevant_events if key in event_map}
            quick_drop = True
            for drop_channel in to_drop:
                if drop_channel not in raw.ch_names:
                    quick_drop = False
                    break
            if quick_drop:
                print('quick drop channels',to_drop)
                raw.drop_channels(to_drop)
            else:
                for drop_channel in to_drop:
                    if drop_channel in raw.ch_names:
                        print('drop channel',drop_channel)
                        raw.drop_channels(drop_channel)

            if output_format == 'epoch':
                epochs = mne.Epochs(raw, reformed_events, filtered_event_id, tmin=start, tmax=end, baseline=None, preload=True,verbose=False)
            elif output_format == 'raw':
                epochs = raw # the whole session.
        else: # the dataset is already epoched by MATLAB script. 
            raise NotImplementedError
        print('finished preprocessing in {} format {}: {}'.format(output_format,set_file,epochs.get_data().shape))
        epoch_data.append(epochs)   
        session_info['session_shape'].append(epochs.get_data().shape)
        session_info['session_name'].append(all_sessions[i])
        session_info['session_id'].append(all_sessions[i].split('.')[0][4:]) 
        session_info['events'].append(eeg_events_df)  
        
        """HUGE UPDATE HERE, SUBJECT12-SESSION13, the 302th EEG trial is missing, so the audio annotation need to skil this trial to match EEG data number. 302th data in python list index is 301.
        """
        


        if debug:
            if i==1:
                break  
    
    ####################################################
    # Combine the data if needed
    if combine_sessions:
        if output_format == 'epoch':
            all_epochs = mne.concatenate_epochs(epoch_data)
        elif output_format == 'raw':
            all_epochs = mne.concatenate_raws(epoch_data)
        else:
            raise ValueError('output_format not supported')        
        if debug:
            print('after concatenate',all_epochs.get_data().shape)
    else:
        all_epochs = epoch_data
        if debug:
            print('no concatenate, number of session:',len(all_epochs))    
            # print size for each session
            for i in range(len(all_epochs)):
                print('session',i,all_epochs[i].get_data().shape)
            """
            session 0 (122, 2110766)
            session 1 (122, 2380900)
            """

            
    return all_epochs,session_info
def _load_cdt_raw(file_path):
    raw = mne.io.read_raw_curry(file_path, preload=True,verbose=False)
    print('loaded ',file_path)
    return raw
def combine_cdt_sessions(
    subject_data_folder,
    relevant_events,
    start=0,end=1.0,
    output_format='epoch',
    epoched_data= False,
    combine_sessions=False, 
    syllable_mapping=False,
    default_drop_channels = ['10', '11', '84', '85', '110', '111','VEO', 'HEO', 'EKG', 'EMG', 'Trigger'],
    n_jobs=16,
    debug=False
    ):
    session_info = {
        'session_shape':[],
        'session_name':[],
        'session_id':[],
        'events':[],
        'num_audio':[],
    }
    subject_id = subject_data_folder.split('/')[-1]
    #list all folders starts with sess, and is folder
    set_folder= subject_data_folder
    if debug:
        print('subject_data_folder',subject_data_folder)
    print('start loading session from cdt')
    cdt_files = [f for f in os.listdir(set_folder) if f.endswith('.cdt')]
    cdt_files.sort(key=lambda x: int(x[4:-4]))
    cdt_file_ids = [f[4:-4] for f in cdt_files]    
    set_files = [os.path.join(subject_data_folder,p) for p in cdt_files]
    all_sessions = [f.split('/')[-1] for f in set_files]
    # print(set_files)
    # print('all_sessions',all_sessions)

    ####################################################
    # Start processing the data to raw wave
    epoch_data = []
    to_drop = default_drop_channels
    event_map = {}
    event_map_reverse = {}
    relevant_events = [str(e) for e in relevant_events]
    for i, e in enumerate(relevant_events):
        event_map[e] = i+1
        event_map_reverse[i+1] = e
    
    ####################################################
    # Preload all raw data in parallel
    raw_data_list = Parallel(n_jobs=n_jobs)(delayed(_load_cdt_raw)(f) for f in set_files)

    ####################################################
    # Loop through the sessions
    speech_event_count =0
    cue_event_count = 0
    for i,set_file in enumerate(set_files):
        ##############################
        # Preload the audio annotation data
        if syllable_mapping == True:
            raise NotImplementedError("syllable_mapping is not supported for cdt files") 
        else:
            audio_annotation_json_data = None
        session_trial_count = 0
        session_trial_intervals = [] # contains tuples of start (13) and end (14) of each trial
        if not epoched_data: # only non-epoched data is supported at the moment             
            full_path = os.path.join(set_folder, set_file)
            # print('full_path',full_path)
            # raw = mne.io.read_raw_eeglab(full_path, preload=True,verbose=False)  
            raw = raw_data_list[i] #mne.io.read_raw_curry(full_path, preload=True,verbose=False)
            
            events, event_id = mne.events_from_annotations(raw,verbose=False)            
            ############################################
            ### Make the event dataframe
            eeg_events=list()
            marker_count = {
            }
            # print('raw.annotations',raw.annotations)
            # exit(0)
            for annot in raw.annotations:                    
                marker = annot['description']
                onset = annot['onset']
                marker_count[marker] = marker_count.get(marker,0)+1
                if marker == 'boundary':
                    continue
                try: 
                    marker = int(marker)
                except:
                    print('marker',marker,'is not a number but not "boundary", found a critical error, exiting program')
                    exit(0)                
                # Prepare a dictionary for describing the event
                event_dict = {}
                event_dict['semanticGroup'] = None
                event_dict['num_utterance'] = None #
                event_dict['description'] = marker
                event_dict['onset'] = onset
                try:
                    maker_event = _get_event_type_by_marker(str(marker))
                except:
                    # print('marker',marker,'is not in the event list')
                    # print(annot)
                    continue
                # find start of trial event
                if maker_event == 'start_trial':
                    session_trial_count+=1
                    # we can put the audio annotation here. 

                # Cue or action. 
                if 'speech' in maker_event.lower() or 'cue' in maker_event.lower():                    
                    event_dict['kind'] = 'word'
                    word = maker_event.split('_')[1]
                    event_dict['word'] = word
                    if 'speech' in maker_event.lower():
                        event_dict['condition'] = 'speech'
                        speech_event_count+=1
                        # word-semantic group mapping
                        if word in ['Jumping','Running','Swimming','Going']:
                            event_dict['semanticGroup'] = 'speech_Motion'
                        elif word in ['Happy','Sad','Fun','Horrible']:
                            event_dict['semanticGroup'] = 'speech_Emotion'
                        elif word in ['College','Home','Battlefield','Here']:
                            event_dict['semanticGroup'] = 'speech_Location'
                        elif word in ['Mother','Cowboy','Professor','Me']:
                            event_dict['semanticGroup'] = 'speech_Person'
                        elif word in ['One','Three','Eleven','Million']:
                            event_dict['semanticGroup'] = 'speech_Number'
                        elif word in ['Spoon','Alfa','Python','Telephone']:
                            event_dict['semanticGroup'] = 'speech_Object'
                        # word - number of utterance mapping
                        if word in ['Sad','Fun','Home','Me','One','Three','Spoon']:
                            event_dict['num_utterance'] = 1
                        elif word in ['Horrible','College','Battlefield','Professor','Eleven','Million','Telephone']:
                            event_dict['num_utterance'] = 3
                        elif word in ['Jumping','Running','Swimming','Going','Happy','Here','Mother','Cowboy','Alfa','Python']:
                            event_dict['num_utterance'] = 2
                    elif 'cue' in maker_event.lower():
                        event_dict['condition'] = 'read'
                        cue_event_count+=1
                        # print('>>>>>>>>>>>>>>>>> found cue',cue_event_count,'marker maker_event',maker_event,annot)
                        if word in ['Jumping','Running','Swimming','Going']:
                            event_dict['semanticGroup'] = 'read_Motion'
                        elif word in ['Happy','Sad','Fun','Horrible']:
                            event_dict['semanticGroup'] = 'read_Emotion'
                        elif word in ['College','Home','Battlefield','Here']:
                            event_dict['semanticGroup'] = 'read_Location'
                        elif word in ['Mother','Cowboy','Professor','Me']:
                            event_dict['semanticGroup'] = 'read_Person'
                        elif word in ['One','Three','Eleven','Million']:
                            event_dict['semanticGroup'] = 'read_Number'
                        elif word in ['Spoon','Alfa','Python','Telephone']:
                            event_dict['semanticGroup'] = 'read_Object'           

                        # word - number of utterance mapping
                        if word in ['Sad','Fun','Home','Me','One','Three','Spoon']:
                            event_dict['num_utterance'] = 1
                        elif word in ['Horrible','College','Battlefield','Professor','Eleven','Million','Telephone']:
                            event_dict['num_utterance'] = 3
                        elif word in ['Jumping','Running','Swimming','Going','Happy','Here','Mother','Cowboy','Alfa','Python']:
                            event_dict['num_utterance'] = 2

                else:                    
                    event_dict['kind'] = 'exp'
                    event_dict['word'] = None
                    event_dict['condition'] = None
                    if maker_event == 'cross_on':
                        event_dict['condition'] = 'eye'
                    if maker_event == 'relax_on':
                        event_dict['condition'] = 'relax'
                    if maker_event == 'ding_on':    
                        event_dict['condition'] = 'ding'
                event_dict['session_id'] = i
                event_dict['session_name'] = all_sessions[i]
                event_dict['subject_name'] = subject_id
                event_dict['audio_name'] = None
                eeg_events.append(event_dict)

            # make dataframe 
            eeg_events_df = pd.DataFrame(eeg_events)
            ############################################
            ### Finish making the event dataframe
            ############################################
            ### Do a mapping in mne event id to ensure session consistency
            reverse_mapping = {v: k for k, v in event_id.items()}
            reformed_events = []
            for j,e in enumerate(events):
                if str(reverse_mapping[e[2]]) not in relevant_events:
                    pass
                else:
                    reformed_events.append(e)
            for j,e in enumerate(reformed_events):
                e_label = reverse_mapping[e[2]]
                e[2] = event_map[e_label]
            reformed_events = np.array(reformed_events)
            filtered_event_id = {key: event_map[key] for key in relevant_events if key in event_map}
            # print('prepare to drop channels',raw.ch_names)
            # check if all the channels in to_drop are in raw.ch_names
            quick_drop = True
            for drop_channel in to_drop:
                if drop_channel not in raw.ch_names:
                    quick_drop = False
                    break
            if quick_drop:
                print('quick drop channels',to_drop)
                raw.drop_channels(to_drop)
            else:
                for drop_channel in to_drop:
                    if drop_channel in raw.ch_names:
                        print('drop channel',drop_channel)
                        raw.drop_channels(drop_channel)
            if output_format == 'epoch':
                epochs = mne.Epochs(raw, reformed_events, filtered_event_id, tmin=start, tmax=end, baseline=None, preload=True,verbose=False)
            elif output_format == 'raw':
                epochs = raw # the whole session.
        else: # the dataset is already epoched by MATLAB script. 
            raise NotImplementedError
        print('finished preprocessing in {} format {}: {}'.format(output_format,set_file,epochs.get_data().shape))
        epoch_data.append(epochs)   
        session_info['session_shape'].append(epochs.get_data().shape)
        session_info['session_name'].append(all_sessions[i])
        session_info['session_id'].append(all_sessions[i].split('.')[0][4:]) 
        session_info['events'].append(eeg_events_df)  
        
        """HUGE UPDATE HERE, SUBJECT12-SESSION13, the 302th EEG trial is missing, so the audio annotation need to skil this trial to match EEG data number. 302th data in python list index is 301.
        """
        if debug:
            if i==1:
                break  
    
    ####################################################
    # Combine the data if needed
    if combine_sessions:
        if output_format == 'epoch':
            all_epochs = mne.concatenate_epochs(epoch_data)
        elif output_format == 'raw':
            all_epochs = mne.concatenate_raws(epoch_data)
        else:
            raise ValueError('output_format not supported')        
        if debug:
            print('after concatenate',all_epochs.get_data().shape)
    else:
        all_epochs = epoch_data
        if debug:
            print('no concatenate, number of session:',len(all_epochs))    
            # print size for each session
            for i in range(len(all_epochs)):
                print('session',i,all_epochs[i].get_data().shape)
            """
            session 0 (122, 2110766)
            session 1 (122, 2380900)
            """

            
    return all_epochs,session_info
def safe_np_load(file):
    print("Loading numpy file:", file)
    with np.load(file, allow_pickle=True) as data:
        return dict(data)
def hash_params_simple(params):
    # Sort keys for consistent hashing
    sorted_params = dict(sorted(params.items()))
    param_str = json.dumps(sorted_params, sort_keys=True, default=str)
    return hashlib.md5(param_str.encode()).hexdigest()
def _save_npy_trial(trial, output_path, tmin, tmax, resample_fs):
    # trial_ = trial.copy()
    raw_trial = trial.get_data()
    raw_trial = raw_trial.squeeze()
    raw_trial = raw_trial[:,np.newaxis,:int((tmax-tmin)*resample_fs)]
    # save the data to npy file
    np.save(output_path, raw_trial)
    # grant the permission
    # os.chmod(output_path, 0o777)
    # print('saved npy file to', output_path)

################### Preprocessing Tools 
# in this function, no ICA is considered
def _preproc_core_epoch(
    epoch, 
    preproc_params):
    print("###########################################################")
    print('Start Process Core')
    # assign the parameters
    n_jobs = preproc_params.get('n_jobs', 16)
    resample_fs = preproc_params.get('resample_fs', epoch.info['sfreq'])
    fmin = preproc_params.get('fmin', None)
    fmax = preproc_params.get('fmax', None)
    avg_ref = preproc_params.get('avg_ref', False)
    filter_method = preproc_params.get('filter_method', 'fir')
    baseline_tmin = preproc_params.get('baseline_tmin', None)
    baseline_tmax = preproc_params.get('baseline_tmax', None)
    reject_epoch =  preproc_params.get('reject_epochs', False)
    reject_epoch_method = preproc_params.get('reject_epochs_method', None)
    apply_whitening = preproc_params.get('apply_whitening', False)
    whitening_dim = preproc_params.get('whitening_dim', 20)
    apply_ica = preproc_params.get('apply_ica', False)
    ica_n_components = preproc_params.get('ica_n_components', 20)
    ica_method = preproc_params.get('ica_method', 'fastica')
    ica_decim = preproc_params.get('ica_decim', 3)
    print(
        'epoch shape',epoch.get_data().shape,
        'resample_fs',resample_fs,
        'fmin',fmin,
        'fmax',fmax,
        'filter_method',filter_method,
        'avg_ref',avg_ref,
        'baseline_tmin',baseline_tmin,
        'baseline_tmax',baseline_tmax, 
        'reject_epoch',reject_epoch,
        'reject_epoch_method',reject_epoch_method,
        'apply_whitening',apply_whitening
    )    
    ############################################################
    # apply avg_referencing
    if avg_ref:
        print('apply average reference')
        epoch.set_eeg_reference(ref_channels='average', projection=True, verbose=False)
    else:
        print('no average reference applied')    
    ##########################################################
    # resample 
    if resample_fs != epoch.info['sfreq']:
        print('resample data from {} to {}'.format(epoch.info['sfreq'],resample_fs))
        epoch = epoch.resample(sfreq=resample_fs,n_jobs=n_jobs)
    else:
        print('no resample applied, current fs is {}'.format(epoch.info['sfreq']))    
    ###########################################################
    # Band pass filter 
    if fmin is not None and fmax is not None:
        # check larger than 0
        if fmin > 0 and fmax > 0:
            print('filtering data with fmin {} and fmax {}'.format(fmin,fmax))
            epoch = epoch.filter(
                l_freq=fmin,
                h_freq=fmax,
                method=filter_method, 
                verbose=False, 
                n_jobs=n_jobs
            )
    else:
        print('no filter applied')        

    ############################################################
    # apply baseline correction if needed
    if baseline_tmin is not None and baseline_tmax is not None:
        print('apply baseline correction from {} to {}'.format(baseline_tmin,baseline_tmax))
        epoch.apply_baseline((baseline_tmin, baseline_tmax))
    else:
        print('no baseline correction applied')

    ############################################################
    # apply epoch rejection if needed
    bad_epochs = None
    if len(epoch) <= 1:
        print('only 1 epoch, skip epoch rejection')
        reject_epoch = False
    if reject_epoch:
        print('apply epoch rejection using {}'.format(reject_epoch_method))
        print('Epoch Rejection is not ready yet because after deleting epoch, you need to change the identifiers accordingly')
        epoch, bad_epochs = _reject_bad_epochs(epoch, preproc_params)
    else:
        print('no epoch rejection is applied')
    
    ############################################################
    # apply ICA on Epoch
    if apply_ica:
        raise NotImplementedError('ICA on epoch data is not implemented yet, because the ICA fitting is not stable on epoched data, we recommend applying ICA on raw data before epoching. ')
        epoch, ica = _clean_raw_with_ica(
                epoch,
                n_components=epoch_ica_cfg.get("ica_n_components", "n-1"),
                method=epoch_ica_cfg.get("ica_method", "fastica"),
                random_state=epoch_ica_cfg.get("ica_random_state", 97),
                ica_threshhold=epoch_ica_cfg.get("ica_threshhold", 0.9),
                debug=epoch_ica_cfg.get("debug", False),
                iclabel_remove=epoch_ica_cfg.get("iclabel_remove", None),
                iclabel_keep=epoch_ica_cfg.get("iclabel_keep", None),
                iclabel_thresholds=epoch_ica_cfg.get("iclabel_thresholds", None),
                decim=epoch_ica_cfg.get("ica_decim", None),
                ica_fit_highpass=ica_fit_hp,   # NEW
            )
    else:
        print('no ICA applied on epoch data')
    

    ############################################################
    # apply ICA on Epoch


    ############################################################
    # apply whitening to data if needed
    if apply_whitening:
        print('apply whitening to data')
        # Implement whitening here
        epoch = mvnn_whiten_train_test(epoch, mvnn_dim= whitening_dim)
    else:
        print('no whitening applied')

    # print("bad_epochs", bad_epochs, type(bad_epochs))
    # e.g., [3] <class 'numpy.ndarray'>
    # check_epoch_channel_erp_plot(epoch,title="Debug_CoTraining")
    return epoch, bad_epochs

def _preproc_core_raw(
    raw,    
    preproc_params,
    force_skip_channel_cleaning=False, ):
    print("###########################################################")
    print('Start Process Core on Raw Data')
    n_jobs = preproc_params.get('n_jobs', 16)    
    # drop the channels if needed
    to_drop = preproc_params.get('drop_channels', [])
    if len(to_drop) > 0:
        all_drops =[]
        for drop_channel in to_drop:
            if drop_channel in raw.ch_names:
                all_drops.append(drop_channel)
        print('drop channel',all_drops)
        raw.drop_channels(all_drops)
    avg_ref = preproc_params.get('avg_ref', False)
    l_freq, h_freq = preproc_params.get('l_freq', None), preproc_params.get('h_freq', None)
    filter_method = preproc_params.get('filter_method', 'fir')    
    apply_notch = preproc_params.get('notch_filter', False)
    notch_freqs = preproc_params.get('notch_freqs', [50])
    apply_clean_channels = preproc_params.get('clean_channels', False)
    clean_lo_std = preproc_params.get('clean_lo_std', 0.1)
    clean_hi_std = preproc_params.get('clean_hi_std', 5.0)
    clean_win = preproc_params.get('clean_win_sec', 2.0)
    clean_step =  preproc_params.get('clean_win_step_sec', clean_win)
    resample_fs = preproc_params.get('resample_fs', None)
    apply_ica = preproc_params.get('apply_ica', False)

    print("###########################################################")
    print(f"[Raw Preprocess] Bandpass Filtering {l_freq} - {h_freq} Hz")   
    if l_freq is not None and h_freq is not None:
        # check larger than 0
        if l_freq > 0 and h_freq > 0:
            print('filtering data with {} low {} and high {}'.format(filter_method,l_freq,h_freq))
            raw = raw.filter(l_freq=l_freq,h_freq=h_freq, method='iir', verbose=False, n_jobs=n_jobs)
    
    print("###########################################################")
    print(f"[Raw Preprocess] Apply Notch Filter: {apply_notch} at freqs {notch_freqs}")    

    if apply_notch:
        print('apply notch filter at freqs {}'.format(preproc_params['notch_freqs']))
        raw = raw.notch_filter([50, 100], n_jobs=n_jobs, verbose=False)    
    
    print("###########################################################")
    print(f"[Raw Preprocess] clean channels {apply_clean_channels} with low std {clean_lo_std} and high std {clean_hi_std}, window {clean_win} sec, step {clean_step} sec")
    if apply_clean_channels and not force_skip_channel_cleaning:
        raw = _detect_and_interpolate_bad_channels_v2(raw, lo_std=clean_lo_std, hi_std=clean_hi_std,win_sec=clean_win, step_sec=clean_step,n_jobs=n_jobs) 

    # apply avg_referencing if needed
    print("###########################################################")
    print(f"[Raw Preprocess] Apply average reference: {avg_ref}")    
    if avg_ref:
        print('apply average reference')
        # apply average reference
        raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)

    print("###########################################################")
    print(f"[Raw Preprocess] Resampling data to {resample_fs} Hz")    # down sampling
    
    if resample_fs is not None:
        if resample_fs != raw.info['sfreq']:
            print('resample data from {} to {}'.format(raw.info['sfreq'],resample_fs))
            raw = raw.resample(sfreq=resample_fs,n_jobs=n_jobs)
        else:
            print('no resample applied, current fs is {}'.format(raw.info['sfreq']))
    
    
    # apply ICA 
    print("###########################################################")
    print(f"[Raw Preprocess] Apply ICA: {apply_ica}")
    ica=None
    if apply_ica:
        raw, ica = _clean_raw_with_ica(
            raw, 
            n_components=preproc_params.get('ica_n_components', 20), 
            method=preproc_params.get('ica_method', 'picard'), 
            random_state=preproc_params.get('ica_random_state', 97),
            ica_threshhold=preproc_params.get('ica_threshhold', 0.9),
            debug=preproc_params.get('debug', False)
        )        
    return raw, ica

def _make_channel_names_uppercase(raw):
    new_ch_names = {}
    for ch in raw.ch_names:
        new_ch_names[ch] = ch.upper()
    raw.rename_channels(new_ch_names)
    return raw

def preprocess_raw_wave_V5(
    subject='S05',
    root='/projects/SilSpeech/Spoken_EEG/Subjects_MergeSet',
    cache_folder = '/projects/MMBCI/Charles_MMBCI/Cache/',
    feature_folder_name='debug',
    pp_postfix=None,#'_cleaned.set',
    epoched_data=False,
    tmin=0.2,tmax=0.8,
    fmin=0,fmax=150, 
    resample_fs=250, 
    resample_freq_time=1, 
    avg_ref=False,
    format = 'curry',
    output_format='epoch',
    cache_trials=True, # if True, will cache the trials into npy files
    cache_session=True, # if True, will cache the session into tif files
    combine_sessions = True,
    n_jobs=16, batch_size=16, debug=False, 
    relevant_events=['100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123'],
    **kwargs):
    # cache_trials cannot be True if output_format is raw
    if output_format == 'raw' and cache_trials:
        raise ValueError('output_format is raw, cannot cache trials')
    # cache_session cannot be True if output_format is epoch
    if output_format == 'epoch' and cache_session:
        raise ValueError('output_format is epoch, cannot cache session')
    # cache_trials and cache_session cannot be True at the same time
    if cache_trials and cache_session:
        raise ValueError('cache_trials and cache_session cannot be True at the same time')
    # if output_format is raw, we cannot combine_sessions
    if output_format == 'raw' and combine_sessions:
        raise ValueError('output_format = raw, cannot combine_sessions = True')
    data_path =os.path.join(root,subject) 
    if cache_folder is not None:
        subject_output_folder = os.path.join(cache_folder,subject,'features',feature_folder_name)
        subject_meta_info_path = os.path.join(subject_output_folder,'meta_info.pkl')
    else:
        subject_output_folder =os.path.join(data_path,subject,'features',feature_folder_name)   
        subject_meta_info_path = os.path.join(subject_output_folder,'meta_info.pkl')  
    
    print('[preprocess_raw_wave_V5] subject_output_folder',subject_output_folder)
    if not os.path.exists(subject_output_folder):
        os.makedirs(subject_output_folder,exist_ok=True)
        print('create folder',subject_output_folder)
        # grant the permission
        os.chmod(subject_output_folder, 0o777)
    print('>>>>>>>>>>>debug',debug)
    if debug:    
        print("pp_postfix",pp_postfix)    
    t0 = time.time()
    if format=='curry':
        epochs,meta_info=combine_curry_sessions(
            subject_data_folder=data_path,
            relevant_events=relevant_events,
            start=tmin, end=tmax,
            postfix=pp_postfix,
            epoched_data=epoched_data,
            output_format=output_format,
            combine_sessions=combine_sessions,
            debug=debug)
    elif format =='cdt':
        epochs,meta_info=combine_cdt_sessions(
            subject_data_folder=data_path,
            relevant_events=relevant_events,
            start=tmin, end=tmax,
            epoched_data=epoched_data,
            output_format=output_format,
            combine_sessions=combine_sessions,n_jobs=n_jobs,
            debug=debug)
    if debug:   
        print('meta_info events',len(meta_info['events']))   
    print('data loading time',time.time()-t0)
    if output_format == 'epoch':
        # for the number of session in meta_info, we will create the session folder 
        for i in range(len(meta_info['session_name'])):
            session_folder = os.path.join(subject_output_folder,meta_info['session_name'][i])
            if not os.path.exists(session_folder):
                os.makedirs(session_folder,exist_ok=True)
                # grant the permission
                os.chmod(session_folder, 0o777)
        meta_info['features'] = []
        meta_info['label'] = []
        meta_info['subject_labels'] = []
        meta_info['session_labels'] = []
        meta_info['num_channels'] = None
        meta_info['num_freq'] = None
        meta_info['num_time'] = resample_freq_time
        if debug:
            epochs = epochs[:10]
        if isinstance(epochs,mne.Epochs):
            channel_names = epochs.ch_names
        else:
            channel_names = epochs[0].ch_names
        # channel_names = epochs.ch_names    
        # epochs, _ = _preproc_core_epoch(epochs, resample_fs, fmin, fmax, avg_ref,n_jobs)     
        epochs, _ = _preproc_core_epoch(
            epochs,
            {
                'resample_fs': resample_fs,
                'fmin': fmin,
                'fmax': fmax,
                'avg_ref': avg_ref,
                'n_jobs': n_jobs,
                'filter_method': 'fir',
                'baseline_tmin': None,
                'baseline_tmax': None,
                'reject_epochs': False,
                'reject_epochs_method': None,
                'apply_whitening': False,
                'whitening_dim': 20,
                'apply_ica': False,
                'ica_n_components': 20,
                'ica_method': 'fastica',
                'ica_decim': 3
            }
        )
        #############################
        # Start processing the data to raw wave
        #############################    
        gc.enable()
        if debug:   
            print('preprocess_raw_wave_V5 got EEG data',epochs.get_data().shape)
        times = epochs.get_data().shape[-1] 
        global_trial_index = 0
        global_session_index = 0        
        # iterate through the epochs, not saving the cache yet
        raw_trial_shape = None
        output_path_list = []
        for i in tqdm(range(len(epochs))):
            if raw_trial_shape is None:
                epoch = epochs[i].copy()
                # save the pw into the session folder 
                raw_trial = epoch.get_data()
                raw_trial = raw_trial.squeeze()
                # add 1 frequency dimension
                raw_trial = raw_trial[:,np.newaxis,:int((tmax-tmin)*resample_fs)]
                raw_trial_shape = raw_trial.shape
                if debug:
                    print('raw_trial',raw_trial_shape)
            session_folder = os.path.join(subject_output_folder,meta_info['session_name'][global_session_index])
            output_path = os.path.join(session_folder,'trial_{}.npy'.format(global_trial_index))
            output_path_list.append(output_path)
            # if cache_trials:
            #     np.save(output_path,raw_trial)
            #     # grant the permission
            #     os.chmod(output_path, 0o777)
            if debug:
                print('saved',output_path)
            if meta_info['num_channels'] == None:
                meta_info['num_channels'] = raw_trial_shape[0]
            if meta_info['num_freq'] == None:
                meta_info['num_freq'] = raw_trial_shape[1]
            meta_info['features'].append(output_path)
            meta_info['subject_labels'].append(subject)
            meta_info['session_labels'].append(meta_info['session_id'][global_session_index])
            global_trial_index += 1
            if global_trial_index >= meta_info['session_shape'][global_session_index][0]:
                print(meta_info['session_name'][global_session_index],meta_info['session_shape'][global_session_index],'session done')
                global_session_index += 1
                global_trial_index = 0
        
        # save the cache here in parralle. 
        # if cache_trials:
        #     print('cache trials to npy files')
        #     t0 = time.time()
        #     # Parallel(n_jobs=n_jobs)(delayed(_save_npy_trial)(epochs[i],output_path=output_path_list[i], tmin=tmin, tmax=tmax, resample_fs=resample_fs) for i in range(len(epochs)))
        #     # do it serially for debugging
        #     # for i in tqdm(range(len(epochs))):
        #     #     _save_npy_trial(epochs[i],output_path=output_path_list[i], tmin=tmin, tmax=tmax, resample_fs=resample_fs)

        #     # print('cache trials done time spent:',time.time()-t0)
        #     Parallel(n_jobs=n_jobs)(
        #         delayed(_save_npy_trial)(
        #             epochs[i],
        #             output_path=output_path_list[i], 
        #             tmin=tmin, 
        #             tmax=tmax, 
        #             resample_fs=resample_fs
        #         ) for i in tqdm(range(len(epochs)))
        #     )
            
        #     print('cache trials done time spent:', time.time()-t0)
        if cache_trials:
            print('Cache trials to npy files')
            t0 = time.time()
            
            # Pre-calculate once
            n_samples = int((tmax - tmin) * resample_fs)
            n_trials = len(epochs)
            
            # Get ALL data at once (this is fastest in MNE)
            print(f'Loading all {n_trials} trials into memory...')
            all_data = epochs.get_data()  # Shape: (n_trials, n_channels, n_times)
            
            # Crop time dimension
            all_data = all_data[:, :, :n_samples]
            
            # Add singleton dimension if needed
            all_data = all_data[:, :, np.newaxis, :]  # or whatever shape you need
            
            # Parallel save (now just saving arrays, no MNE objects)
            print(f'Saving {n_trials} files in parallel...')
            
            def save_single_trial(i):
                np.save(output_path_list[i], all_data[i])
            
            Parallel(n_jobs=n_jobs, batch_size='auto')(
                delayed(save_single_trial)(i) 
                for i in tqdm(range(n_trials), desc="Saving")
            )
            
            print(f'Cache trials done, time spent: {time.time()-t0:.2f}s')



        
        meta_info['frequencies'] = []
        meta_info['channel_names'] = channel_names
        meta_info['label'] = epochs.events[:,-1]
        assert len(meta_info['features']) == len(meta_info['label'])
        with open(subject_meta_info_path,'wb') as f:
            pickle.dump(meta_info,f)
            # grant the permission
            os.chmod(subject_meta_info_path, 0o777)
        print('meta_info saved to',subject_meta_info_path)
        if debug:
            print(len(meta_info['features']),len(meta_info['label']))
            print(meta_info['features'][:10],meta_info['label'][:10])
            print(meta_info)
    if output_format == 'raw':    
        if debug:
            print('subject_output_folder',subject_output_folder)
        # need these information to rebuild the dataset
        meta_info['features_h5'] = [] # put the path of the raw data in h5py here. 
        meta_info['features_fif'] = [] # put the path of the raw data in fif here.
        # meta_info['features_h5_combined'] = [] # put the path of the raw data in npy here.
        meta_info['subject_labels'] = []
        meta_info['session_labels'] = []
        meta_info['num_channels'] = None
        meta_info['num_freq'] = None
        meta_info['sampling_rate'] = None
        meta_info['frequency_range'] =None
        channel_names =None    
        # create a h5 file in the subject_output_folder to store all the sessions
        # h5_file_whole_path = os.path.join(subject_output_folder,'eeg-whole.h5')
        # with h5py.File(h5_file_whole_path, 'w', libver="latest") as f:
        # save the session into fif files
        for i in range(len(epochs)):
            ################################################################
            if meta_info['num_channels'] == None:
                meta_info['num_channels'] = epochs[i].get_data().shape[0]
            if channel_names == None:
                channel_names = epochs[i].ch_names      
                meta_info['channel_names'] = channel_names          
            epochs[i], _ = _preproc_core_epoch(epochs[i], resample_fs, fmin, fmax, avg_ref, n_jobs=n_jobs)
            if meta_info['sampling_rate'] == None:
                meta_info['sampling_rate'] = resample_fs
            if meta_info['frequency_range'] == None:
                meta_info['frequency_range'] = [fmin,fmax]
            ################################################################
            gc.enable()
            session_folder = os.path.join(subject_output_folder,meta_info['session_name'][i])
            print('session_folder',session_folder)
            if not os.path.exists(session_folder):
                os.makedirs(session_folder,exist_ok=True) 
                # grant the permission
                os.chmod(session_folder, 0o777) 
            # get the raw data, save as h5py
            outpath_h5 = os.path.join(session_folder,'eeg.h5')
            outpath_fif = os.path.join(session_folder,'eeg-raw.fif')
            meta_info['features_h5'].append(outpath_h5)
            meta_info['features_fif'].append(outpath_fif)
            meta_info['subject_labels'].append(subject)
            meta_info['session_labels'].append(meta_info['session_id'][i])
            raw_data = epochs[i].get_data()
            # convert to numpy array
            raw_data = raw_data.squeeze()                
            # TODO: the chunk size need to be similar to the whole shape, then 
            chunk_size = (raw_data.shape[0],50000)
            print('raw_data',raw_data.shape,raw_data.dtype, 'chunk size',chunk_size)
            if cache_session:
                with h5py.File(outpath_h5, 'w', libver="latest") as f:
                    f.create_dataset('eeg', data=raw_data, chunks=chunk_size)
                    # , compression="gzip"
                    f.swmr_mode = True 
                # grant the permission
                os.chmod(outpath_h5, 0o777)
                epochs[i].save(outpath_fif, overwrite=True)
                print('saved session to',outpath_h5,'and',outpath_fif)
            # also save the event dataframe to a csv file
            session_df =meta_info['events'][i]
            csv_output_path = os.path.join(session_folder,'events.csv')
            session_df.to_csv(csv_output_path,index=False)
            # grant the permission
            os.chmod(csv_output_path, 0o777)
            if debug:
                print(session_df.head())
                print('session saved to',outpath_h5 ,'and',outpath_fif)
                print('csv saved to',csv_output_path)
        # print(meta_info)
        # put all the sessions in one single h5 files.
        with open(subject_meta_info_path,'wb') as f:
            pickle.dump(meta_info,f)
            # grant the permission
            os.chmod(subject_meta_info_path, 0o777)
        print('meta_info saved to',subject_meta_info_path)        
    return meta_info

def _calculate_np_array_size(arr):
    # return in GB
    return arr.nbytes / (1024 ** 3)

################### Tools
# Bandpass filter
def bandpass_filter(data, lowcut, highcut, fs=250, order=5):
    # print('data shape',data.shape)
    
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band') # butter filter parameters
    # Apply the filter to each channel
    res = filtfilt(b, a, data, axis=-1)
    return res   
# Compute band power features
def compute_band_power(dataset, eeg_path, fs=250):
    eeg_data = dataset._load_eeg(eeg_path)
    epsilon = 1e-10
    bands = {"delta": (2, 5), "theta": (5, 8), "alpha": (8, 13), "beta": (13, 30)}
    band_powers = []
    for low, high in bands.values():
        filtered = bandpass_filter(eeg_data, low, high, fs)
        power = np.maximum(np.var(filtered, axis=-1), epsilon) # band power (122,) because 122 channels
        power = np.where(np.isinf(power), epsilon, power)
        band_powers.append(np.log(power))
    res =  np.hstack(band_powers)
    # print('band power shape',res.shape)
    return res
def _normed_cosine_similarity(target_vector, source_vector):
    norm_product = np.linalg.norm(target_vector) * np.linalg.norm(source_vector)
    cosine_dist = 1 - np.dot(target_vector, source_vector) / (norm_product + 1e-10)
    return cosine_dist
def _kl_divergence(target_vector, source_vector):
    kl_divergence = entropy(target_vector + 1e-10, source_vector + 1e-10)
    kl_divergence = np.where(np.isinf(kl_divergence), 1e6, kl_divergence)
    return kl_divergence
# Compute pairwise distances
def compute_distances(distance_mectic, target_vector, source_vectors, n_jobs=16):
    distances = []
    if distance_mectic== 'euclidean':
        func = euclidean
    elif distance_mectic== 'chebyshev':
        func = chebyshev
    elif distance_mectic== 'cosine':
        func = _normed_cosine_similarity 
    elif distance_mectic== 'kl':
        func = _kl_divergence
    elif distance_mectic== 'correlation':
        func = correlation
    else:
        raise ValueError(f'distance metric {distance_mectic} not supported')
    # print('computing distance with',func)   
    results = Parallel(n_jobs=n_jobs)(delayed(func)(target_vector, source_vector) for source_vector in source_vectors)
    results =  np.array(results)
    # print('distance shape',results.shape)
    return results
# Clean the MDM matrix
def clean_mdm_matrix(mdm_matrix):
    if np.any(np.isnan(mdm_matrix)) or np.any(np.isinf(mdm_matrix)):
        print("Warning: MDM matrix contains NaN or inf values. Replacing them.")
        mdm_matrix = np.nan_to_num(mdm_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(mdm_matrix, -1e6, 1e6)
# Cluster subjects using GMM
def cluster_subjects(mdm_matrix, max_clusters=10, cluster_method='gmm'):
    mdm_matrix = clean_mdm_matrix(mdm_matrix)
    best_bic = np.inf
    best_gmm = None
    best_labels = None
    if cluster_method == 'gmm':
        for n in range(3, max_clusters):        
            gmm = sklearn.mixture.GaussianMixture(n_components=n, random_state=42).fit(mdm_matrix)
            bic = gmm.bic(mdm_matrix)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_labels = gmm.predict(mdm_matrix)
    return best_labels, best_gmm

def _canon_label(name: str) -> str:
    """
    Map user / EEGLAB-style labels to mne_icalabel convention.

    Examples:
        'Line Noise'   -> 'line_noise'
        'channel noise'-> 'channel_noise'
        'EOG'          -> 'eye'
        'ECG'          -> 'heart'
    """
    name = str(name).strip().lower()
    # normalise spaces / hyphens
    name = name.replace("-", "_").replace(" ", "_")

    aliases = {
        "muscle_artifact": "muscle",
        "muscle_artifacts": "muscle",
        "muscles": "muscle",
        "eog": "eye",
        "ecg": "heart",
        "line_noise_artifact": "line_noise",
        "linenoise": "line_noise",
        "line-noise": "line_noise",
        "channelnoise": "channel_noise",
        "ch_noise": "channel_noise",
        "chnoise": "channel_noise",
    }
    return aliases.get(name, name)

def _clean_epoch_with_ica(
    raw,
    n_components="n-1",
    method="fastica",
    random_state=97,
    ica_threshhold=0.9,
    debug=False,
    iclabel_remove=None,
    iclabel_keep=None,
    iclabel_thresholds=None,
    decim=None,            # decimation factor for ICA.fit (None or int > 1)
    ica_fit_highpass=None, # NEW: extra HP only for ICA fitting (two-stage ICA)
):
    """
        Run ICA on a Raw object and auto-remove components using ICLabel,
        in a way that mimics EEGLAB-style control.

        Parameters
        ----------
        raw : mne.io.Raw
            Input raw data (will not be modified in-place).
        n_components : int | str | float
            Number of ICA components. If str, can be:
                'n'   -> n_channels
                'n-1' -> n_channels - 1
            If float in (0, 1), it is interpreted as a fraction of channels.
        method : {'fastica', 'picard', 'infomax', 'extended-infomax'}
            ICA algorithm to use. 'picard' is usually fast & robust (if installed).
        random_state : int
            Random seed for ICA.
        ica_threshhold : float
            Global default probability threshold (0–1) if no per-class threshold is
            provided.
        debug : bool
            If True, crop the Raw to ~100 seconds before fitting ICA
            to speed up experiments.
        iclabel_remove : list[str] | None
            Names of ICLabel classes to remove, e.g. (EEGLAB-style):
                ['muscle', 'eye', 'heart', 'line noise', 'channel noise']
            or with underscores: ['line_noise', 'channel_noise'].
        iclabel_keep : list[str] | None
            Classes that should be protected from removal (e.g. ['brain']).
        iclabel_thresholds : None | float | dict[str, float]
            Probability thresholds:
            - None: use ``ica_threshhold`` as global threshold for all remove-classes.
            - float: global probability threshold for all remove-classes.
            - dict: class_name -> probability threshold, EEGLAB-style names allowed
            (spaces/underscores both OK), e.g.:

                {'muscle': 0.8, 'eye': 0.8, 'line_noise': 0.85}
        decim : int | None
            If not None and > 1, pass as ``decim`` argument to ICA.fit()
            to speed up fitting.
        ica_fit_highpass : float | None
            If not None, apply this high-pass filter ONLY to a copy of the
            data used for ICA.fit() (two-stage ICA). The ICA solution is
            then applied to the original, less aggressively filtered data
        Returns
        -------
        raw_clean : mne.io.Raw
            Copy of ``raw`` with ICA cleaned and applied.
        ica : mne.preprocessing.ICA
            Fitted ICA object (with ``ica.exclude`` marking rejected comps).
    """
    import numpy as np
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    # ------------------------------------------------------------------
    # Resolve n_components
    # ------------------------------------------------------------------
    if isinstance(n_components, str):
        if n_components == "n":
            n_components_ = len(raw.ch_names)
        elif n_components == "n-1":
            n_components_ = len(raw.ch_names) - 1
        else:
            raise ValueError(f"n_components {n_components!r} not supported")
    elif isinstance(n_components, (int, np.integer)):
        if n_components <= 0:
            raise ValueError("n_components must be > 0")
        n_components_ = int(n_components)
    elif isinstance(n_components, (float, np.floating)):
        if 0 < n_components < 1:
            n_components_ = int(max(1, round(n_components * len(raw.ch_names))))
        else:
            raise ValueError(
                f"Float n_components must be in (0,1); got {n_components}"
            )
    else:
        raise ValueError(f"Unsupported type for n_components: {type(n_components)}")

    # ------------------------------------------------------------------
    # Resolve ICA method and fit_params
    # ------------------------------------------------------------------
    allowed_methods = {"fastica", "picard", "infomax", "extended-infomax"}
    method = str(method).lower()
    if method not in allowed_methods:
        raise ValueError(
            f"ICA method {method!r} not supported. "
            f"Use one of {sorted(allowed_methods)}"
        )

    # picard fallback if not installed
    if method == "picard":
        try:
            import picard 
            has_picard = True
        except ImportError:
            has_picard = False

        if not has_picard:
            print(
                "[ICA] WARNING: method='picard' requested but 'picard' package "
                "is not installed. Falling back to method='fastica'."
            )
            method = "fastica"

    # Method-specific fit_params:
    # - fastica: do NOT pass ortho/extended (sklearn.FastICA will error)
    # - picard: use ortho / extended flags
    # - infomax / extended-infomax: use extended flag
    fit_params = None
    if method == "picard":
        fit_params = dict(ortho=False, extended=True, verbose=True)
    elif method in ("infomax", "extended-infomax"):
        fit_params = dict(
            extended=(method == "extended-infomax"),
            verbose=True,
        )

    print(
        f"[ICA] Fitting ICA (method={method}, n_components={n_components_}, "
        f"random_state={random_state})"
    )

    if fit_params is None:
        ica = ICA(
            n_components=n_components_,
            method=method,
            random_state=random_state,
        )
    else:
        ica = ICA(
            n_components=n_components_,
            method=method,
            random_state=random_state,
            fit_params=fit_params,
        )

    # ------------------------------------------------------------------
    # Prepare data & fit ICA (with optional cropping & decimation)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Prepare data & fit ICA (with optional two-stage HP, cropping & decimation)
    # ------------------------------------------------------------------
    raw_fit = raw.copy()

    # --- Two-stage ICA: extra HP only for fitting ---
    if ica_fit_highpass is not None:
        try:
            hp = float(ica_fit_highpass)
        except Exception:
            print(f"[ICA] WARNING: could not parse ica_fit_highpass={ica_fit_highpass!r}, ignoring.")
            hp = None

        if hp is not None and hp > 0:
            print(f"[ICA] Two-stage ICA: high-pass copy at {hp} Hz for ICA fit only")
            # Only filter the copy used for fitting; the cleaned ICA
            # will be applied to the original (less filtered) data.
            raw_fit.filter(
                l_freq=hp,
                h_freq=None,
                method="fir",
                verbose=False,
            )
    if debug:
        max_t = min(100.0, raw_fit.times[-1])
        print(f"[ICA] Debug mode: cropping data to first {max_t:.1f}s for ICA fit")
        raw_fit.crop(0, max_t)

    if decim is not None and isinstance(decim, (int, np.integer)) and decim > 1:
        print(f"[ICA] Using decim={decim} for ICA.fit()")
        ica.fit(raw_fit, decim=int(decim))
    else:
        ica.fit(raw_fit)
    # ------------------------------------------------------------------
    # ICLabel classification
    # ------------------------------------------------------------------
    labels = label_components(raw, ica, method="iclabel")
    ica_probs = labels["y_pred_proba"]    # (n_components, n_classes)
    ica_pred_labels = np.asarray(labels["labels"], dtype=str)  # length n_components
    ic_classes = labels.get("classes", None)  # e.g. ['brain','muscle','eye',...]

    # Map class name -> column index, using canonicalised names
    class_to_idx = {}
    if ic_classes is not None:
        class_to_idx = {
            _canon_label(cls_name): i for i, cls_name in enumerate(ic_classes)
        }

    # ------------------------------------------------------------------
    # Normalise config for remove/keep/thresholds
    # ------------------------------------------------------------------
    # Defaults if user doesn't specify
    remove_default = ["muscle", "eye", "heart", "line_noise", "channel_noise"]
    keep_default = ["brain"]

    # Turn into lower-case lists and then canonicalise label names
    iclabel_remove_list = _norm_list(iclabel_remove, remove_default)
    iclabel_keep_list = _norm_list(iclabel_keep, keep_default)

    iclabel_remove = [_canon_label(lbl) for lbl in iclabel_remove_list]
    iclabel_keep = [_canon_label(lbl) for lbl in iclabel_keep_list]
    # ----------------------------------------------------
    # Normalise iclabel_thresholds to native Python types
    # ----------------------------------------------------
    from omegaconf import DictConfig, OmegaConf
    # Convert DictConfig -> plain dict
    if isinstance(iclabel_thresholds, DictConfig):
        iclabel_thresholds = OmegaConf.to_container(iclabel_thresholds, resolve=True)

    # global default threshold
    global_thr = float(ica_threshhold)
    thr_dict = {}  # per-class thresholds (canonicalised labels)

    if iclabel_thresholds is None:
        # use global ica_threshhold for all remove-classes
        pass
    elif isinstance(iclabel_thresholds, (int, float, np.floating)):
        # single global numeric threshold for all remove-classes
        global_thr = float(iclabel_thresholds)
    elif isinstance(iclabel_thresholds, dict):
        # per-class thresholds; fall back to global_thr if class not in dict
        thr_dict = {
            _canon_label(str(k)): float(v)
            for k, v in iclabel_thresholds.items()
        }
    else:
        raise TypeError(
            "iclabel_thresholds must be None, a float, or a dict[class_name -> float]"
        )

    def _get_thr(lbl_lc: str) -> float:
        """Get threshold for this label (canonical lowercase)."""
        return thr_dict.get(lbl_lc, global_thr)

    # ----------------------------------------------------
    # Decide which components to exclude using ICLabel
    # ----------------------------------------------------
    n_comp = ica_probs.shape[0]
    exclude_ics = []

    print("[ICA] ===== ICLabel component summary =====")
    for i in range(n_comp):
        prob_vec = ica_probs[i]
        pred_label = _canon_label(ica_pred_labels[i])
        max_prob = float(prob_vec.max())

        # Determine main label for logging
        if ic_classes is not None:
            max_idx = int(prob_vec.argmax())
            main_label_name = ic_classes[max_idx]
            main_label_prob = float(prob_vec[max_idx])
        else:
            main_label_name = pred_label
            main_label_prob = max_prob

        remove_this = False

        # 1) Check remove rules: for each remove-class, see if its prob >= threshold
        for lbl in iclabel_remove:
            thr = _get_thr(lbl)

            if lbl in class_to_idx:
                p_lbl = float(prob_vec[class_to_idx[lbl]])
            else:
                # Fallback: if predicted label matches and prob is high
                p_lbl = max_prob if pred_label == lbl else 0.0

            if p_lbl >= thr:
                remove_this = True
                break

        # 2) Protection: if strongly belonging to a keep class with a
        #    user-specified threshold, do not remove.
        if remove_this:
            for lbl in iclabel_keep:
                # Only apply keep-rule if user provided a threshold for that label
                if lbl not in thr_dict:
                    continue
                thr_keep = thr_dict[lbl]
                if lbl in class_to_idx:
                    p_keep = float(prob_vec[class_to_idx[lbl]])
                else:
                    p_keep = max_prob if pred_label == lbl else 0.0

                if p_keep >= thr_keep:
                    remove_this = False
                    break

        mark = "X" if remove_this else " "
        print(
            f"[ICA] {mark} IC {i:3d}: main={main_label_name:<12s} "
            f"p(main)={main_label_prob:5.3f}, max_prob={max_prob:5.3f}"
        )

        if remove_this:
            exclude_ics.append(i)

    exclude_ics = np.array(exclude_ics, dtype=int)
    print(f"[ICA] Excluding {len(exclude_ics)} components: {exclude_ics}")

    # ------------------------------------------------------------------
    # Apply ICA to get cleaned data
    # ------------------------------------------------------------------
    raw_clean = raw.copy()
    ica.exclude = list(exclude_ics)
    ica.apply(raw_clean)

    return raw_clean, ica
def _clean_raw_with_ica(
    raw, 
    n_components=20, 
    method="picard", 
    random_state=97,
    ica_threshhold=0.9,
    debug = False,
    ):
    raise NotImplementedError("The ICA cleaning function is not fully implemented yet.")

    # from mne.preprocessing import ICA    
    from mne_icalabel import label_components

    ##############################
    # Resolve n_components parameter   
    # n_components can be str: "n", "n-1", or float between 0 and 1
    if isinstance(n_components, str):
        if n_components == 'n':
            n_components = len(raw.ch_names)
        elif n_components == 'n-1':
            n_components =len(raw.ch_names)-1
        else:
            raise ValueError(f'n_components {n_components} not supported')
    elif isinstance(n_components, float):
        if n_components > 0 and n_components < 1:
            n_components = int(n_components * len(raw.ch_names))
    else:
        raise ValueError(f'n_components {n_components} not supported')    

    ##############################
    # Pick ICA Method
    allowed_methods = ["fastica", "picard", "infomax", "extended-infomax"]
    if method not in allowed_methods:
        raise ValueError(f'ICA method {method} not supported, use one of {allowed_methods}')
    default_fit_params = {
        "infomax":          dict(ortho=False,extended=False, verbose=True),
        "extended-infomax": dict(ortho=False,extended=True, verbose=True),        
        "fastica":          dict(ortho=False, extended=True, verbose=True),
        "picard":           dict(ortho=False, extended=True, verbose=True),   
    }
    
    ##############################
    # Execute ICA fitting
    if method == "picard": # picard logic is different from others.
        raise NotImplementedError("Method 'picard' is not implemented in this function. Use fastica or infomax instead.")
    else:        
        ica = mne.preprocessing.ICA( 
            n_components=n_components, 
            method=method, random_state=random_state,
            fit_params=default_fit_params[method]
        )
        ica.fit(raw)
    ##############################
    # Classify IC
    labels = label_components(raw, ica, method='iclabel')
    ica_probs = labels['y_pred_proba']
    ica_pred_labels = labels['labels']
    ic_classes = labels.get('classes', None)
    exclude_ics = []
    for i, (prob, label) in enumerate(zip(ica_probs, ica_pred_labels)):
        if label not in ['brain','other'] and prob > 0.9:
            exclude_ics.append(i)
            print(f"Excluding IC {i} with label {label} and probability {prob:.2f}")
    ica.exclude = exclude_ics   
    raw_clean = ica.apply(raw.copy())
    return raw_clean, ica
def _detet_reject(window, start,stop, lo_std, hi_std, ch_names):
    channel_detection_results ={
        'bad_ch_names_per_window':[],
        'start_per_window':start,
        'stop_per_window':stop,
        'need_interpolation':False,
    }
    ch_std = window.std(axis=1)
    median_std = np.median(ch_std)
    if median_std <= 0:
        return channel_detection_results
    flat_mask = ch_std < (median_std * lo_std)
    noisy_mask = ch_std > (median_std * hi_std)
    bad_mask = flat_mask | noisy_mask
    if not bad_mask.any():
        return channel_detection_results
    bad_ch_names = ch_names[bad_mask].tolist() 
    channel_detection_results['bad_ch_names_per_window']=bad_ch_names
    channel_detection_results['start_per_window'] = start
    channel_detection_results['stop_per_window']=stop
    if bad_ch_names:
        channel_detection_results['need_interpolation'] = True
    else:
        channel_detection_results['need_interpolation'] =False
    return channel_detection_results
def _detect_and_interpolate_bad_channels_v1(raw,lo_std,hi_std, win_sec, step_sec, n_jobs=16):
    """
    Sliding-window bad-channel detection and local interpolation on Raw.

    Strategy:
    - Slide a window over the continuous raw signal (e.g. 2 s).
    - For each window:
        * compute per-channel std within that window
        * compare to median std in that window
        * mark channels that are too flat or too noisy
        * interpolate ONLY that window for those channels
    - The rest of the recording for those channels is left untouched.
    - No global 'bads' are kept in raw.info['bads'].

    Config in preproc_params (typically preproc_args['preproc_raw']):
        clean_channels: bool
        bad_std_low: float   (fraction of median; channels below are 'flat')
        bad_std_high: float  (fraction of median; channels above are 'noisy')
        clean_win_sec: float (window size in seconds)
        clean_step_sec: float (step size in seconds; <= win_sec for overlap)
    """
    if not raw.preload:
        raw.load_data()
    data = raw.get_data()  # (n_channels, n_times)
    n_ch, n_t = data.shape
    sfreq = raw.info["sfreq"]
    print("###########################################################")
    print("Sliding-window bad-channel cleaning on Raw (local interpolation)") 
    win_samp = max(1, int(round(win_sec * sfreq)))
    step_samp = max(1, int(round(step_sec * sfreq)))

    data = raw.get_data()  # (n_channels, n_times)
    n_ch, n_t = data.shape
    ch_names = np.array(raw.ch_names)

    total_bad_instances = 0
    total_windows_with_bads = 0

    if n_t <= win_samp:
        starts = [0]
    else:
        starts = list(range(0, n_t - win_samp + 1, step_samp))
    for wi, start in tqdm(enumerate(starts)):
        stop = min(start + win_samp, n_t)
        window = data[:, start:stop]  # (n_channels, win_len)
        ch_std = window.std(axis=1)
        median_std = np.median(ch_std)
        if median_std <= 0:
            continue
        flat_mask = ch_std < (median_std * lo_std)
        noisy_mask = ch_std > (median_std * hi_std)
        bad_mask = flat_mask | noisy_mask
        if not bad_mask.any():
            continue
        bad_ch_names = ch_names[bad_mask].tolist()
        total_bad_instances += len(bad_ch_names)
        total_windows_with_bads += 1
        # use MNE spatial interpolation using neighboring channels
        info_seg = raw.info.copy()
        seg_raw = mne.io.RawArray(window, info_seg, verbose=False)
        seg_raw.info["bads"] = bad_ch_names
        seg_raw.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)

        # write back ONLY this window
        data[:, start:stop] = seg_raw.get_data()

    raw._data = data
    print(
        "Sliding-window cleaning done.\n"
        f"  windows with bad channels: {total_windows_with_bads}\n"
        f"  total bad-channel *instances* interpolated: {total_bad_instances}"
    )
    return raw
def _detect_and_interpolate_bad_channels_v2(raw,lo_std,hi_std, win_sec, step_sec, n_jobs=16):   
    if not raw.preload:
        raw.load_data()
    data = raw.get_data()
    n_ch, n_t = data.shape
    sfreq = raw.info["sfreq"]
    print("###########################################################")
    print("Sliding-window bad-channel cleaning on Raw (local interpolation)") 
    win_samp = max(1, int(round(win_sec * sfreq)))
    step_samp = max(1, int(round(step_sec * sfreq)))
    data = raw.get_data() 
    data_cleaned = data.copy()
    n_ch, n_t = data.shape
    ch_names = np.array(raw.ch_names)
    total_bad_instances = 0
    total_windows_with_bads = 0
    if n_t <= win_samp:
        starts = [0]
    else:
        starts = list(range(0, n_t - win_samp + 1, step_samp))
    stops = [min(start + win_samp, n_t) for start in starts]
    t0 = time.time()
    detection_results_list = Parallel(n_jobs=n_jobs)(
        delayed(_detet_reject)(
            data[:, start:stop], start, stop, lo_std, hi_std, ch_names
        )
        for start, stop in tqdm(zip(starts, stops))
    )
    print('time for detection in all windows:',time.time()-t0)
    for res in detection_results_list:
        if res['need_interpolation']:
            total_bad_instances += len(res['bad_ch_names_per_window'])
            total_windows_with_bads += 1              
    t1 = time.time()
    if total_windows_with_bads ==0:
        print("No bad channels detected in any window. Exiting cleaning.")
    else:
        print(
            f"Interpolating {total_windows_with_bads} windows with bad channels.\n"
        )
        for res in tqdm(detection_results_list):
            start = res['start_per_window']
            stop = res['stop_per_window']
            window = data[:, start:stop]  # (n_channels, win_len)
            bad_ch_names = res['bad_ch_names_per_window']
            # use MNE spatial interpolation using neighboring channels
            info_seg = raw.info.copy()
            seg_raw = mne.io.RawArray(window, info_seg, verbose=False)
            seg_raw.info["bads"] = bad_ch_names     
            seg_raw.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)
            # write back ONLY this window
            data_cleaned[:, start:stop] = seg_raw.get_data()
    raw._data = data_cleaned
    # summarize the noisy channels
    high_freq_noisy_channel = {}
    for res in detection_results_list:
        for ch in res['bad_ch_names_per_window']:
            high_freq_noisy_channel[ch] = high_freq_noisy_channel.get(ch, 0) + 1
    print("High frequency noisy channels summary:", high_freq_noisy_channel)

    print('time for interpolation in all windows:', time.time()-t1)
    return raw
# Epoch rejection: amplitude
def _reject_bad_epochs_amplitude(epochs, preproc_params):
    
    amp_thresh_uv =preproc_params.get("robust_amp_thresh", 100.0)
    print("[Epoch rej: amplitude] Starting amplitude-based epoch rejection with threshold {:.1f} µV".format(amp_thresh_uv))
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs = data.shape[0]
    max_amp_uv = np.max(np.abs(data), axis=(1, 2)) * 1e6  # V -> µV
    bad_idx = np.where(max_amp_uv > amp_thresh_uv)[0]
    if bad_idx.size >0:
        epochs.drop(bad_idx, reason=f"amp>{amp_thresh_uv:.1f}µV")
    print(
        f"[Epoch rej: amplitude] {len(bad_idx)} / {n_epochs} epochs exceed {amp_thresh_uv:.1f} µV "
        f"(max seen {max_amp_uv.max():.1f} µV)"
    )
    return epochs, bad_idx
#  Epoch rejection: GFP robust z-score
def _reject_bad_epochs_gfp_zscore(epochs, preproc_params):
    """
        Reject outlier epochs based on robust z-score of global field power (GFP).

        Strategy:
        - For each epoch, compute GFP_epoch = sqrt(mean(x^2) over channels & time).
        - Across epochs, compute median and MAD of GFP.
        - Robust z = 0.6745 * (GFP - median) / MAD.
        - Drop epochs where |z| > gfp_z_thresh.
    """
    z_thresh = preproc_params.get("gfp_z_thresh", 5.0)
    print("[Epoch rej: GFP] Starting GFP-based epoch rejection with z-score threshold {:.1f}".format(z_thresh))
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs = data.shape[0]
    # GFP per epoch: RMS over channels & time
    gfp = np.sqrt(np.mean(data ** 2, axis=(1, 2)))  # shape: (n_epochs,)
    median_gfp = np.median(gfp)
    mad = np.median(np.abs(gfp - median_gfp))
    bad_idx = []
    if mad == 0:
        print("[Epoch rej: GFP] MAD is zero; cannot compute robust z-scores. Skipping.")
        return epochs, bad_idx
    z = 0.6745 * (gfp - median_gfp) / mad
    bad_mask = np.abs(z) > z_thresh
    bad_idx = np.where(bad_mask)[0]
    if bad_idx.size >0:
        epochs.drop(bad_idx, reason=f"GFP|z|>{z_thresh:.1f}")
    print(
        f"[Epoch rej: GFP] dropping {len(bad_idx)} / {n_epochs} epochs "
        f"with |z| > {z_thresh:.1f} (max |z| = {np.abs(z).max():.2f})"
    )
    return epochs, bad_idx
def _reject_bad_epochs(epochs, preproc_params):
    method = preproc_params.get("reject_epochs_method", "amplitude")

    if method == "amplitude":
        return _reject_bad_epochs_amplitude(epochs, preproc_params)
    elif method == "gfp_zscore":
        return _reject_bad_epochs_gfp_zscore(epochs, preproc_params)
    else:
        raise ValueError(f"Unknown reject_method='{method}'")


def mvnn_whiten_train_test(mne_epoch, mvnn_dim="time"):
    """
        NICE-EEG-style MVNN whitening for (n_epochs, n_channels, n_times) data,
        estimating the covariance *only from training data* and applying the same
        whitening transform to both train and test.

        Parameters
        ----------
        X_train : np.ndarray
            Training data, shape (n_train, n_channels, n_times).
        mvnn_dim : {"time", "epochs"}
            - "time": compute covariance at each time point across epochs, then average.
            - "epochs": compute covariance for each epoch (over time), then average.

        Returns
        -------
        X_train_w, X_test_w : np.ndarray
            Whitened train and test data, same shapes as inputs.
    """
    X_train = mne_epoch.get_data()
    n_train, n_ch, n_t = X_train.shape

    print("###########################################################")
    print(f"[MVNN] Using training data only for whitening (mvnn_dim='{mvnn_dim}')")
    print(f"[MVNN] X_train shape = {X_train.shape}")
    print("###########################################################")
    # ----- estimate covariance only from training data -----
    if mvnn_dim == "time":
        # Covariance at each time point across epochs, then average
        sigma_time = np.empty((n_t, n_ch, n_ch), dtype=np.float64)
        for tt in range(n_t):
            # X_t: (n_train, n_ch) -> samples x features
            X_t = X_train[:, :, tt]
            sigma_time[tt] = _cov(X_t, shrinkage="auto")
        sigma_tot = sigma_time.mean(axis=0)
    elif mvnn_dim == "epochs":
        # Covariance per epoch (over time), then average
        sigma_epoch = np.empty((n_train, n_ch, n_ch), dtype=np.float64)
        for ee in range(n_train):
            # X_e: (n_t, n_ch) -> samples x features
            X_e = X_train[ee].T  # (n_t, n_ch)
            sigma_epoch[ee] = _cov(X_e, shrinkage="auto")
        sigma_tot = sigma_epoch.mean(axis=0)
    else:
        raise ValueError(f"mvnn_dim must be 'time' or 'epochs', got {mvnn_dim}")
    # Inverse square-root of covariance
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

    def _apply_white(X):
        # X: (n_epochs, n_ch, n_t)
        n, ch, t = X.shape
        # Flatten across (epoch, time), whiten across channels
        tmp = X.transpose(0, 2, 1).reshape(-1, ch)   # (n * t, ch)
        tmp_w = tmp @ sigma_inv                      # (n * t, ch)
        return tmp_w.reshape(n, t, ch).transpose(0, 2, 1)

    X_train_w = _apply_white(X_train)
    mne_epoch._data = X_train_w
    return  mne_epoch
################### Channel Spec Creation Tool
def create_channel_spec(cdt_path, output_folder, spec_name):
    # Load the Curry CDT file
    raw = mne.io.read_raw_curry(cdt_path, preload=True)
    # Get the montage (electrode positions)
    montage = raw.get_montage()
    # Extract channel names in their exact order
    channel_names = raw.ch_names
    # Get 3D positions
    pos_3d = montage.get_positions()['ch_pos']  
    
    """
        # layout_2d = mne.channels.find_layout(raw.info, ch_type='eeg')# Create a layout for 2D positions
        # This doesn't work for curry 9 recordings
        Parameters:
            box: tuple of length 4  The box dimension (x_min, x_max, y_min, y_max).
            pos:  array, shape=(n_channels, 4). The unit-normalized positions of the channels in 2d (x, y, width, height).
            names: list of str.  The channel names.
            id; sarray_like of int        The channel ids.
            kind            The type of Layout (e.g. ‘Vectorview-all’).
    """

    # Create lists to store data
    data = []
    for idx, ch_name in enumerate(channel_names):
        data.append({
            'channel_name': ch_name,
            'channel_name_upper_case': ch_name.upper(),
            'x': pos_3d[ch_name][0] if ch_name in pos_3d else 0,
            'y': pos_3d[ch_name][1] if ch_name in pos_3d else 0,
            'z': pos_3d[ch_name][2] if ch_name in pos_3d else 0,            
        })
    # Create DataFrame
    df = pd.DataFrame(data)
    # Display the DataFrame
    print(df)
    #create folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        # grant the permission
        os.chmod(output_folder, 0o777)
    # Optional: Save to CSV
    # df.to_csv('channel_positions.csv', index=False)
    spec_path = os.path.join(output_folder, f'{spec_name}_channel_spec.csv')
    df.to_csv(spec_path, index=False)
    # grant the permission
    os.chmod(spec_path, 0o777)
    print(f'Channel spec saved to {spec_path}')


################### Debugging Tools
def get_ram_usage():
    process = psutil.Process(os.getpid())  # Get the current process
    mem_info = process.memory_info()  # Get memory usage details
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB
def get_ran_ratio():
    total_ram = psutil.virtual_memory().total/(1024 ** 2)  # Total system RAM in bytes
    used_ram = get_ram_usage()
    return used_ram / total_ram
def check_basic_channel_erp_plot(raw):
    # filter 1-40 hz
    raw = raw.filter(l_freq=1,h_freq=40, method='fir', verbose=False, n_jobs=16)
    # average ref
    # raw.set_eeg_reference('average', projection=False, verbose=False)
    #>>>>>>>>>>>>>>>>>>>>>>>> Charles
    events, event_id = mne.events_from_annotations(raw)
    print('events \n',events)
    print('event_id',event_id)
    # wanted_ids = {
    #     '100': 1,
    #     '111': 2,
    #     '102': 3,
    # }
    wanted_ids = {}
    # generate wanted_ids for event 101 to 111
    for i in range(101, 112):
        wanted_ids[str(i)] = i - 99
    epochs = mne.Epochs(raw, events, event_id=wanted_ids, tmin=-1, tmax=2, baseline=None,preload=True)
    e = epochs.copy().load_data()  # ensure data in memory
    baseline_data_before = e.get_data(tmin=-0.5, tmax=0)
    mean_before = baseline_data_before.mean(axis=-1).mean()  # Mean over time, then over epochs/channels
    # Apply baseline correction
    e.apply_baseline((-0.5, 0))
    # Get baseline data after correction
    baseline_data_after = e.get_data(tmin=-0.5, tmax=0)
    mean_after = baseline_data_after.mean(axis=-1).mean()  # Mean over time, then over epochs/channels
    print("baseline mean before:", mean_before)
    print("baseline mean after:", mean_after)
    print("baseline mean after (should be ~0):", baseline_data_after.mean())

    
    # mean_before = e.get_data(tmin=-0.5, tmax=0).mean()
    # e.apply_baseline((-0.5, 0))
    # mean_after = e.get_data(tmin=-0.5, tmax=0).mean()

    # print("baseline mean before:", mean_before)
    # print("baseline mean after :", mean_after)
    # exit(0)
    # extract epochs from 100,101 and 103
    # evoked = epochs.average()
    # evoked.plot(show=False)
    # save plot
    # plt.savefig('./debug/evoked_plot.png')
    # print('evoked saved to ./debug/evoked_plot.png')
    # for each channel, plot its erp
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        plt.figure(figsize=(6,3))
        plt.plot(epochs.times, np.mean(epochs.get_data()[:,ch_idx, :], axis=0))
        plt.title(f'ERP for channel {ch_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (V)')
        # set y lim to be -10 to 10 1e-6
        plt.ylim(-10e-6, 5e-6)
        plt.grid()
        # draw a vertical line at time 0
        plt.axvline(x=0, color='r', linestyle='--')
        # draw a horizontal line at y=0
        plt.axhline(y=0, color='k', linestyle='--')
        # save plot
        plt.savefig(f'./debug/evoked_plot_{ch_name}.png')
        plt.close()
        print(f'evoked for channel {ch_name} saved to ./debug/evoked_plot_{ch_name}.png')
def check_basic_channel_erp_plot_nobandpass(raw):
    # average ref
    # raw.set_eeg_reference('average', projection=False, verbose=False)
    #>>>>>>>>>>>>>>>>>>>>>>>> Charles
    events, event_id = mne.events_from_annotations(raw)
    print('events \n',events)
    print('event_id',event_id)
    # wanted_ids = {
    #     '100': 1,
    #     '111': 2,
    #     '102': 3,
    # }
    wanted_ids = {}
    # generate wanted_ids for event 101 to 111
    for i in range(101, 112):
        wanted_ids[str(i)] = i - 99
    epochs = mne.Epochs(raw, events, event_id=wanted_ids, tmin=-1, tmax=2, baseline=None,preload=True)
    e = epochs.copy().load_data()  # ensure data in memory
    baseline_data_before = e.get_data(tmin=-0.5, tmax=0)
    mean_before = baseline_data_before.mean(axis=-1).mean()  # Mean over time, then over epochs/channels
    # Apply baseline correction
    e.apply_baseline((-0.5, 0))
    # Get baseline data after correction
    baseline_data_after = e.get_data(tmin=-0.5, tmax=0)
    mean_after = baseline_data_after.mean(axis=-1).mean()  # Mean over time, then over epochs/channels
    print("baseline mean before:", mean_before)
    print("baseline mean after:", mean_after)
    print("baseline mean after (should be ~0):", baseline_data_after.mean())

    
    # mean_before = e.get_data(tmin=-0.5, tmax=0).mean()
    # e.apply_baseline((-0.5, 0))
    # mean_after = e.get_data(tmin=-0.5, tmax=0).mean()

    # print("baseline mean before:", mean_before)
    # print("baseline mean after :", mean_after)
    # exit(0)
    # extract epochs from 100,101 and 103
    # evoked = epochs.average()
    # evoked.plot(show=False)
    # save plot
    # plt.savefig('./debug/evoked_plot.png')
    # print('evoked saved to ./debug/evoked_plot.png')
    # for each channel, plot its erp
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        plt.figure(figsize=(6,3))
        plt.plot(epochs.times, np.mean(epochs.get_data()[:,ch_idx, :], axis=0))
        plt.title(f'ERP for channel {ch_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (V)')
        # set y lim to be -10 to 10 1e-6
        plt.ylim(-10e-6, 5e-6)
        plt.grid()
        # draw a vertical line at time 0
        plt.axvline(x=0, color='r', linestyle='--')
        # draw a horizontal line at y=0
        plt.axhline(y=0, color='k', linestyle='--')
        # save plot
        plt.savefig(f'./debug/evoked_plot_{ch_name}.png')
        plt.close()
        print(f'evoked for channel {ch_name} saved to ./debug/evoked_plot_{ch_name}.png')
def check_epoch_channel_erp_plot(epochs,title):
    # e = epochs.copy().load_data()  # ensure data in memory
    # baseline_data_before = e.get_data(tmin=-0.5, tmax=0)
    # mean_before = baseline_data_before.mean(axis=-1).mean()  # Mean over time, then over epochs/channels
    # # Apply baseline correction
    # e.apply_baseline((-0.5, 0))
    # # Get baseline data after correction
    # baseline_data_after = e.get_data(tmin=-0.5, tmax=0)
    # mean_after = baseline_data_after.mean(axis=-1).mean()  # Mean over time, then over epochs/channels
    # print("baseline mean before:", mean_before)
    # print("baseline mean after:", mean_after)
    # print("baseline mean after (should be ~0):", baseline_data_after.mean())
    # create the output debug folder

    out_dir = f'./debug_Dec10{title}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        plt.figure(figsize=(6,3))
        plt.plot(epochs.times, np.mean(epochs.get_data()[:,ch_idx, :]*1e6, axis=0))
        plt.title(f'ERP for channel {ch_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (uV)')
        # set y lim to be -10 to 10 1e-6
        plt.ylim(-10, 10)
        plt.grid()
        # draw a vertical line at time 0
        plt.axvline(x=0, color='r', linestyle='--')
        # draw a horizontal line at y=0
        plt.axhline(y=0, color='k', linestyle='--')
        # save plot
        plt.savefig(f'{out_dir}/evoked_plot_{ch_name}.png')
        plt.close()
        print(f'evoked for channel {ch_name} saved ')
def check_numpy_channel_erp_plot(
    np_dataset,
    channel_names,
    tmin, tmax,
    title):
    out_dir = f'./debug_11_16{title}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    for ch_idx, ch_name in enumerate(channel_names):
        plt.figure(figsize=(6,3))
        t = np.linspace(-1, 2, np_dataset.shape[2])  # assuming time from -1s to 2s
        plt.plot(t, np.mean(np_dataset[:,ch_idx, :], axis=0))
        plt.title(f'ERP for channel {ch_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (V)')
        # set y lim to be -10 to 10 1e-6
        plt.ylim(-10e-6, 5e-6)
        plt.grid()
        # draw a vertical line at time 0
        plt.axvline(x=0, color='r', linestyle='--')
        # draw a horizontal line at y=0
        plt.axhline(y=0, color='k', linestyle='--')
        # save plot
        plt.savefig(f'{out_dir}/evoked_plot_{ch_name}.png')
        plt.close()
        print(f'evoked for channel {ch_name} saved ')


################### Dataset Class
class EEGDatasetBase(Dataset):
    def __init__(
        self,
        phase,
        data_root,
        cache_folder=None,
        split_method='cross-session', 
        subjects=['S05','S07','S08'],
        val_subject='S05',
        val_sessions=[1], 
        test_subject='S05',
        test_sessions=[1],
        select_channels=None,
        select_vocabs  =None,
        select_vocabs_remap_index=False,
        vocab_groupping=None, # manually assign a label to any vocabs. 
        preproc_args = {},
        normalize = True,
        random_seed = 0,
        force_rebuild_index= False,
        force_rebuild_feature = False,
        debug = False,
        output_dict = True,
        output_meta_label = False,        
        decode_time =1.0, # relative to the event on set, how long EEG data to be used for decoding
        decode_time_shift = [-0.5,0.5], # this is a random distribution, the time shift will be sampled from this range
        decode_time_shift_distribution = 'uniform', # Uniform or Gaussian or scalar
    ):
        self.name = self.__class__.__name__ 
        self.preproc_args = preproc_args
        self.preproc_args['root'] = data_root
        self.preproc_args['debug'] = debug
        self.data_root = data_root      
        self.cache_folder = cache_folder  
        self.phase = phase
        if phase == 'valtrain':
            self.phase = 'train'
        self.split_method = split_method        
        self.normalize = normalize     
        self.random_seed = random_seed   
        self.subjects = subjects
        self.val_subject = val_subject
        self.val_sessions = val_sessions
        self.test_subject = test_subject
        self.test_sessions = test_sessions
        self.force_rebuild_index = force_rebuild_index
        self.force_rebuild_feature = force_rebuild_feature
        self.select_channels = select_channels
        self.select_vocabs = select_vocabs
        self.vocab_groupping = vocab_groupping
        self.select_vocabs_remap_index = select_vocabs_remap_index
        self.output_dict = output_dict
        self.output_meta_label = output_meta_label
        self.debug = debug        
        # verify if val_subject and test_subject is in the subjects
        assert self.val_subject in self.subjects
        assert self.test_subject in self.subjects 

        # other internal variables
        self.update_channel = False
        self.update_channel_index = None
        self.update_channel_name = None
        self.decode_time = decode_time
        self.decode_time_shift = decode_time_shift
        self.decode_time_shift_distribution = decode_time_shift_distribution
        self._init_vocab()   
    def _init_vocab(self):
        raise NotImplementedError
    def _get_name_preprocess(self):
        raise NotImplementedError
    def _get_name_cache(self):
        raise NotImplementedError
    def _cached_dataset(self): 
        self.cache_dir = self._get_name_cache()
        print('@@ self.cache_dir',self.cache_dir)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir,exist_ok=True)    
            # grant the permission
            os.chmod(self.cache_dir, 0o777)   
        ##################################
        # Naming convention
        ##################################        
        feature_folder_name,cache_path = self._get_preproc_and_cache_names(prefix=self.name)
        print('@@ feature_folder_name',feature_folder_name)      
        print('@@ cache_path',cache_path)

        ##################################
        # Building feature sets
        ##################################
        self.meta_infos = []
        for subj in self.subjects:
            preproc_args_ = self.preproc_args.copy()
            preproc_args_['subject'] = subj
            subject_output_folder = os.path.join(self.cache_dir,subj,'features',feature_folder_name)
            subject_meta_info_path = os.path.join(subject_output_folder,'meta_info.pkl')
            print('subject_meta_info_path',subject_meta_info_path)
            preproc_args_['subject_output_folder'] = subject_output_folder
            preproc_args_['feature_folder_name']  = feature_folder_name
            preproc_args_['cache_folder'] = self.cache_dir
            if self.force_rebuild_feature or not os.path.exists(subject_meta_info_path):
                print('rebuilding cached feature for',subj,'in',subject_output_folder)
                if self.preproc_args['feature_type'] == 'wavelet':
                    if self.preproc_args['format'] == 'xdf':
                        subject_meta_info = preprocess_xdf_wavelet_V4(**preproc_args_)
                    if self.preproc_args['format'] == 'curry':
                        subject_meta_info = preprocess_wavelet_V4(**preproc_args_)
                if self.preproc_args['feature_type'] == 'wave':     
                    if self.preproc_args['format'] == 'xdf':
                        subject_meta_info = preprocess_xdf_wavelet_V4(**preproc_args_)
                    if self.preproc_args['format'] == 'curry':
                        subject_meta_info = preprocess_raw_wave_V5(**preproc_args_)    
                    if self.preproc_args['format'] == 'cdt':
                        subject_meta_info = preprocess_raw_wave_V5(**preproc_args_)               
            else:
                print('load cached feature for',subj,subject_meta_info_path)
                # just read the meta_info
                with open(subject_meta_info_path,'rb') as f:
                    subject_meta_info = pickle.load(f)
            self.meta_infos.append(subject_meta_info)
        if self.debug:
            print(self.meta_infos[0]['features'])
        ##################################
        # Building dataset cache
        ##################################        
        # see if the cache exists    
          
        if os.path.exists(cache_path) and not self.force_rebuild_index:
            print('loading cached dataset index: ',cache_path)
            with open(cache_path, 'rb') as f:
                dataset_dict = pickle.load(f)
                self.dataset= dataset_dict
        else:
            print('rebuild dataset index: ',cache_path)
            self.dataset = self._build()            
            pickle.dump(self.dataset, open(cache_path, 'wb'))     
            # grant the permission
            os.chmod(cache_path, 0o777)
    def collater(self, samples):
        return default_collate(samples)
    def _check_dataset_meta_validity(self,dataset_meta):
        # check the dataset_meta is a dictionary
        assert isinstance(dataset_meta,dict), '[EEGDatasetBase.get_split] dataset_meta should be a dictionary'
        assert 'class_label' in dataset_meta.keys(), '[EEGDatasetBase.get_split] dataset_meta should have key class_label'
        assert 'session_label' in dataset_meta.keys(), '[EEGDatasetBase.get_split] dataset_meta should have key session_label'
        assert 'subject_label' in dataset_meta.keys(), '[EEGDatasetBase.get_split] dataset_meta should have key subject_label'
        #assert same length
        assert len(dataset_meta['class_label']) == len(dataset_meta['session_label']) == len(dataset_meta['subject_label']), '[EEGDatasetBase.get_split] class_label, session_label, subject_label should have the same length'
    def _save_split(self,splits,dataset_meta):
        train_idx = splits['train']
        val_idx = splits['val']
        test_idx = splits['test']
        output_dir = f'{self.cache_dir}/split_checking'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok=True)
            # grant the permission
            os.chmod(output_dir, 0o777)
        output_file = os.path.join(output_dir,f'{self.name}-{self.split_method}-{self.phase}-{str(self.subjects)}.txt')
        with open(output_file,'w') as f:
            f.write(f'train_idx {train_idx}\n')
            # write the feature path too
            for idx in train_idx:
                f.write(f'{dataset_meta["feature_paths"][idx]}\n')
            f.write(f'val_idx {val_idx}\n')
            for idx in val_idx:
                f.write(f'{dataset_meta["feature_paths"][idx]}\n')
            f.write(f'test_idx {test_idx}\n')
            for idx in test_idx:
                f.write(f'{dataset_meta["feature_paths"][idx]}\n')

        print('save split infomation to',output_file)
    def get_split(self,dataset_meta):
        # check the dataset_meta is a dictionary
        self._check_dataset_meta_validity(dataset_meta)
        #######################################
        # split the data into train, val, test
        #######################################
        subject_label = dataset_meta['subject_label']
        session_label = dataset_meta['session_label']
        class_label = dataset_meta['class_label']     
        print('[dataset get_split]','subject_label',len(subject_label),'session_label',len(session_label),'class_label',len(class_label))
        if self.split_method == 'within-session':
            train_idx = []
            val_idx = []
            test_idx = []
            for i,subj in enumerate(self.subjects):
                subj_idx = np.where(np.array(subject_label) == subj)[0]
                shuffle_idx = np.arange(len(subj_idx))
                np.random.seed(self.random_seed)
                np.random.shuffle(shuffle_idx)
                train_idx.extend(subj_idx[shuffle_idx[:int(len(shuffle_idx)*0.8)]])
                val_idx.extend(subj_idx[shuffle_idx[int(len(subj_idx)*0.8):int(len(subj_idx)*0.9)]])
                test_idx.extend(subj_idx[shuffle_idx[int(len(subj_idx)*0.9):]])
        if self.split_method == 'cross-subject':
            # need to specify the subjects only
            train_idx = []
            val_idx = []
            test_idx = []
            for i,subj in enumerate(self.subjects):
                subj_idx = np.where(np.array(subject_label) == subj)[0]
                print('subj_idx',subj,'val subject',self.val_subject,'test subject',self.test_subject)
                print('subj_idx==self.val_subject',subj==self.val_subject)
                print('subj_idx==self.test_subject',subj==self.test_subject)
                if subj==self.val_subject or subj==self.test_subject:
                    if self.val_subject == self.test_subject:
                        # shuffle the data and split into val and test
                        shuffle_idx = np.arange(len(subj_idx))
                        np.random.seed(self.random_seed)
                        np.random.shuffle(shuffle_idx)
                        val_idx.extend(subj_idx[shuffle_idx[:int(len(shuffle_idx)*0.5)]])
                        test_idx.extend(subj_idx[shuffle_idx[int(len(shuffle_idx)*0.5):]])
                    else:
                        if subj == self.val_subject:
                            val_idx.extend(subj_idx)
                        if subj == self.test_subject:
                            test_idx.extend(subj_idx)
                else:
                    train_idx.extend(subj_idx)
        if self.split_method == 'cross-subject-session':# so we can specify the sessions for the val and test subjects
            # need to specify the subjects only
            train_idx = []
            val_idx = []
            test_idx = []
            for i,subj in enumerate(self.subjects):
                subj_idx = np.where(np.array(subject_label) == subj)[0]
                if subj==self.val_subject or subj==self.test_subject:
                    for i,idx in enumerate(subj_idx):
                        sess = int(session_label[idx])
                        if sess in self.val_sessions:
                            val_idx.append(idx)
                        if sess in self.test_sessions:
                            test_idx.append(idx)
                else:
                    train_idx.extend(subj_idx)
        if self.split_method == 'leave-one-session-out':   
            # within subject, cross session or cross subject cross session
            train_idx = []
            val_idx = []
            test_idx = []
            for subj in self.subjects:
                subj_idx = np.where(np.array(subject_label) == subj)[0]
                if subj != self.val_subject and subj != self.test_subject: # this subject is only for training
                    train_idx.extend(subj_idx)
                else:# some sessions from this subject are for val or test                    
                    # check if the val and test subject are the same                    
                    if self.val_subject == self.test_subject:
                        # val_test_session = self.val_sessions + self.test_sessions
                        # iterate through the subject index in the session label list
                        for i,idx in enumerate(subj_idx):
                            sess = int(session_label[idx])       #  Get the session number
                            if sess not in self.val_sessions and sess not in self.test_sessions:                                
                                train_idx.append(idx)
                            else:
                                if sess in self.val_sessions:
                                    val_idx.append(idx)
                                if sess in self.test_sessions:
                                    test_idx.append(idx)
                    else:
                        # print('val_subject not equal to test_subject')
                        if subj == self.val_subject:
                            for i,idx in enumerate(subj_idx):
                                # print('val_subject',sess)
                                sess = session_label[idx]
                                if sess not in self.val_sessions:
                                    train_idx.append(idx)
                                else:
                                    val_idx.append(idx)
                        if subj == self.test_subject:
                            for i,idx in enumerate(subj_idx):
                                sess = session_label[idx]
                                if sess not in self.test_sessions:
                                    train_idx.append(idx)
                                else:
                                    test_idx.append(idx)
        if self.split_method == 'multi-subject-cross-session':   # for all subject listed , we always use the same val and test session as the val and test data
            train_idx = []
            val_idx = []
            test_idx = []
            for subj in self.subjects:
                subj_idx = np.where(np.array(subject_label) == subj)[0] # sample id where the subject is the same, through this id, we can find the session information. 
                # session_idx =session_label[subj_idx]
                print('subject',subj,'len(subj_idx)',len(subj_idx))
                for i, idx in enumerate(subj_idx):
                    sess = session_label[idx]
                    # print('subj',subj,'sess_idx',sess_idx, 'sess_idx in self.val_sessions',sess_idx in self.val_sessions, 'sess_idx in self.test_sessions',sess_idx in self.test_sessions, 'sess_idx in train sessions', sess_idx not in self.val_sessions and sess_idx not in self.test_sessions)
                    if sess in self.val_sessions:
                        val_idx.append(idx)
                    if sess in self.test_sessions:
                        test_idx.append(idx)
                    if sess not in self.val_sessions and sess not in self.test_sessions:
                        train_idx.append(idx)       
                print('train:',len(train_idx),'val:',len(val_idx),'test:',len(test_idx))
        if self.split_method =='no-split':
            # depending on the phase, return the whole dataset 
            if self.phase == 'train':
                train_idx = np.arange(len(class_label))
                val_idx = []
                test_idx = []
            if self.phase == 'val':
                val_idx = np.arange(len(class_label))
                train_idx = []
                test_idx = []
            if self.phase == 'test':
                test_idx = np.arange(len(class_label))
                train_idx = []
                val_idx = []
            print('train:',len(train_idx),'val:',len(val_idx),'test:',len(test_idx))
        index_dict = {
            'train':train_idx,
            'val':val_idx,
            'test':test_idx,
        }
        print('train:',len(train_idx),'val:',len(val_idx),'test:',len(test_idx))
        # self._save_split(index_dict,dataset_meta)
        
        return index_dict
    def _filter_channel(self,dataset, channel_names):
        assert 'channel_names' in dataset.keys(), 'channel_names not in the dataset'
        # select the channels
        channel_idx = [dataset['channel_names'].index(ch) for ch in select_channels if ch in dataset['channel_names']]
        return channel_idx
    @property
    def tmin(self):
        return self.preproc_args['tmin']
    @property
    def tmax(self):
        return self.preproc_args['tmax']
    @property
    def sampling_rate(self):
        return self.preproc_args['resample_fs']
    @property
    def channel_names(self):
        if self.requires_update_channel:
            return self.update_channel_name
        else:
            return self.dataset['channel_names']
    @property
    def get_original_channel_names(self):
        return self.dataset['channel_names']
    @property
    def requires_update_channel(self):
        return self.update_channel
    @property
    def get_update_channel_index(self):
        return self.update_channel_index
    @property
    def unique_sessions(self):
        return np.unique(self.dataset['session'])
    @property
    def unique_subjects(self):
        return np.unique(self.dataset['subject'])   
    
    def _filter_channel_by_name(self,selected_channel_names, use_intersection = True):
        print("self.channel_names",self.channel_names)
        original_channel_names = [ch.upper() for ch in self.channel_names]
        if use_intersection:
            select_channels = [ch.upper() for ch in selected_channel_names]
            select_channels = list(set(original_channel_names).intersection(set(selected_channel_names))) 
            # reorder the select channels by the original order
            select_channels = [ch for ch in select_channels if ch in selected_channel_names]
            # print('selected_channel_names',select_channels,len(select_channels))  
            # print(self.get_original_channel_names)
        else:
            select_channels = selected_channel_names
            #check if the selected channels are in the channel names
            assert all([ch in self.channel_names for ch in select_channels]), 'selected_channel_names not in the channel_names'

        # select the channels from the dataset 
        channel_idx = [original_channel_names.index(ch) for ch in select_channels]
        # print('channel_idx',channel_idx,len(channel_idx))
        print('selected_channel_names',select_channels,len(select_channels))
        # update the dataset
        self.update_channel_name = select_channels
        self.update_channel = True
        self.update_channel_index = channel_idx
    def _remove_channel_by_name(self,selected_channel_names):
        # remove the channels from the dataset
        select_channels = [ch for ch in self.channel_names if ch not in selected_channel_names]
        # select the channels from the dataset 
        channel_idx = [self.dataset['channel_names'].index(ch) for ch in select_channels]
        # print('channel_idx',channel_idx,len(channel_idx))
        # update the dataset
        self.update_channel_name = select_channels
        self.update_channel = True
        self.update_channel_index = channel_idx
    @property
    def uni_class(self):
        return np.unique(self.dataset['label'])
    def _filter_vocabs(self,dataset,select_vocabs):
          
        indxs = [i for i in range(len(dataset['label'])) if dataset['label'][i] in select_vocabs]
        return indxs
    def _standardize(self, data):
        # Standardization: Zero mean and unit variance
        mean_value = data.mean()
        std_value = data.std()
        standardized_data = (data - mean_value) / std_value
        return standardized_data
    def _build(self):
        raise NotImplementedError
    def __len__(self):
        return len(self.dataset['label'])
    def _load_eeg(self,eeg_path):
        eeg = np.load(eeg_path,allow_pickle=True)
        eeg = eeg.squeeze()
        # if self.preproc_args['fspecial']=='old':
        #     eeg = eeg[:,:40,]
        # if normalize:
        #     eeg = (eeg - eeg.mean()) / eeg.std()
        return eeg
    def __getitem__(self, index):
        # print('getting item',index)
        eeg_path = self.dataset['eeg'][index]
        eeg = self._load_eeg(eeg_path)        
        if self.requires_update_channel:
            eeg = eeg[self.get_update_channel_index,:]
        label = self.dataset['label'][index]    
        if self.vocab_groupping is not None:
            # print('original type',label,type(label))
            # print('groupping',self.vocab_groupping)
            label = self.vocab_groupping[int(label)]
            # convert to np.int64
            label = np.int64(label)
        # get the correct segment of the eeg data
        if self.decode_time_shift is not None:
            if self.decode_time_shift_distribution =='uniform':
                # assert self.decode_time_shift is a list
                assert isinstance(self.decode_time_shift, (list, tuple)), 'decode_time_shift should be a list when decode_time_shift_distribution=uniform, instead we get {} {}'.format(type(self.decode_time_shift),self.decode_time_shift)
                assert len(self.decode_time_shift) == 2, 'decode_time_shift should be a list of length 2 when decode_time_shift_distribution=uniform'
                shift = np.random.uniform(self.decode_time_shift[0], self.decode_time_shift[1])
            elif self.decode_time_shift_distribution == 'scalar':
                # assert self.decode_time_shift is a scalar
                assert isinstance(self.decode_time_shift, (int, float)), 'decode_time_shift should be a scalar when decode_time_shift_distribution=scalar' 
                shift = self.decode_time_shift
            else:
                raise ValueError('decode_time_shift_distribution should be uniform or scalar')
        else:
            shift = 0    
        
        if self.select_channels is not None:
            pass

        
        # print('shift',shift)
        eeg=eeg[
            :,
            int((abs(self.tmin)+shift)*self.sampling_rate):int((abs(self.tmin)+shift+self.decode_time)*self.sampling_rate)
            ]
        # all unique session label for the subject
        unique_session_label = np.unique(self.dataset['session'])
        subject_name = self.dataset['subject'][index]
        subject_label = self.subjects.index(subject_name)
        session_label = self.dataset['session'][index] + subject_label*len(unique_session_label)
        if self.normalize:
            eeg = self._standardize(eeg)
        eeg = torch.tensor(eeg).float()
        label = torch.tensor(label).long()
        # print("self.output_dict",self.output_dict)
        if self.output_dict:
            if self.output_meta_label:
                return {
                    'eeg': eeg,
                    'label': label,
                    'session': session_label-1,
                    'subject': subject_label,
                }
            else:
                return {
                    'eeg': eeg,
                    'label': label,
                }
        else:
            if self.output_meta_label:
                return eeg,label,session_label-1,subject_label
            else:
                return eeg,label
class EEGTSDatasetBase(EEGDatasetBase):
    def __init__(
        self,
        phase,
        data_root,
        cache_folder=None,
        split_method='cross-session', 
        subjects=['S05','S07','S08'],
        val_subject='S05',
        val_sessions=[1], 
        test_subject='S07',
        test_sessions=[1],
        seq_len = 500,
        pred_len = 100, 
        seq_stride = 100,
        select_channels=None,
        select_vocabs  =None,
        select_vocabs_remap_index=False,
        preproc_args = {},
        normalize = True,
        random_seed = 0,
        force_rebuild_index= False,
        force_rebuild_feature = False,
        full_loading= False,
        debug = False,
        output_dict = True,
        output_meta_label = False
    ):
        self.name = self.__class__.__name__ 
        self.preproc_args = preproc_args
        self.preproc_args['root'] = data_root
        self.preproc_args['debug'] = debug
        self.data_root = data_root        
        self.cache_folder = cache_folder
        self.phase = phase
        if phase == 'valtrain':
            self.phase = 'train'
        self.split_method = split_method        
        self.normalize = normalize     
        self.random_seed = random_seed   
        self.subjects = subjects
        self.val_subject = val_subject
        self.val_sessions = val_sessions
        self.test_subject = test_subject
        self.test_sessions = test_sessions
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.seq_stride = seq_stride
        self.output_meta_label = output_meta_label
        self.force_rebuild_index = force_rebuild_index
        self.force_rebuild_feature = force_rebuild_feature
        self.select_channels = select_channels
        self.select_vocabs = select_vocabs
        self.full_loading = full_loading
        self.select_vocabs_remap_index = select_vocabs_remap_index
        self.output_dict = output_dict
        self.debug = debug        
        # verify if val_subject and test_subject is in the subjects
        assert self.val_subject in self.subjects
        assert self.test_subject in self.subjects 

        # other internal variables
        self.update_channel = False
        self.update_channel_index = None
        self.update_channel_name = None

    def set_phase(self,phase):
        self.phase = phase
        if phase == 'valtrain':
            self.phase = 'train'
        if self.full_datasets is not None:
            self.dataset = self.full_datasets[self.phase]
        else:
            self._cached_dataset()
    def _cached_dataset(self): 
        self.cache_dir = self._get_name_cache()
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir,exist_ok=True)  
            # grant the permission
            os.chmod(self.cache_dir, 0o777)

        ##################################
        # Naming convention
        ##################################        
        feature_folder_name,cache_path = self._get_preproc_and_cache_names(prefix=self.name)
        print('@@@ feature_folder_name \n',feature_folder_name,'\n@@@ cache_path\n',cache_path)
        ##################################
        # Building feature sets
        ##################################
        self.meta_infos = []
        for subj in self.subjects:
            preproc_args_ = self.preproc_args.copy()
            preproc_args_['subject'] = subj
            subject_output_folder = os.path.join(self.cache_dir,subj,'features',feature_folder_name)
            subject_meta_info_path = os.path.join(subject_output_folder,'meta_info.pkl')
            #os.path.join(self.data_root,subj,'features',feature_folder_name)
            preproc_args_['subject_output_folder'] = subject_output_folder
            preproc_args_['feature_folder_name']  = feature_folder_name
            preproc_args_['cache_folder'] = self.cache_dir
            if self.force_rebuild_feature or not os.path.exists(subject_meta_info_path):
                print('rebuilding cached feature for',subj,'in',subject_output_folder) 
                if self.preproc_args['feature_type'] == 'wave':
                    if self.preproc_args['format'] == 'curry':
                        subject_meta_info = preprocess_raw_wave_V5(**preproc_args_)     
                        if self.debug:
                            print('subject_meta_info',subject_meta_info)
            else:
                # subject has its own cached feature
                print('load subject cached file for',subj,subject_meta_info_path) 
                # just read the meta_info
                with open(subject_meta_info_path,'rb') as f:
                    subject_meta_info = pickle.load(f)
            self.meta_infos.append(subject_meta_info)
        
        if self.debug:
            print(self.meta_infos[0]['features_h5']) 
        ##################################
        # Building dataset cache
        ##################################        
        # see if the cache exists         
        if os.path.exists(cache_path) and not self.force_rebuild_index:
            print('loading cached dataset index: ',cache_path)
            with open(cache_path, 'rb') as f:
                self.full_datasets = pickle.load(f)                
                self.dataset = self.full_datasets[self.phase]
        else:
            print('rebuild dataset index: ',cache_path)
            self.full_datasets = self._build()     
            self.dataset = self.full_datasets[self.phase]   
            print('full_datasets',self.full_datasets.keys())
            pickle.dump(self.full_datasets, open(cache_path, 'wb')) 

        # to avoid inter-process contention, we can only load for each process
        # need to get all file reading handle to the dataset
        # self.dataset['eeg_h5_handle'] = []
        # for eeg_path in self.dataset['eeg']:
        #     print('loading h5py file handle',eeg_path)
        #     self.dataset['eeg_h5_handle'].append(h5py.File(eeg_path,'r', libver="latest", swmr=True ))# driver="mpio", comm=MPI.COMM_WORLD
        # eeg_data = self.dataset['eeg_h5_handle'][i]['eeg'][:] # load the whole data into memory
        if self.full_loading:
           self._full_loading()
    def _full_loading(self,n_jobs=16):
        self.unique_dataset_mapper ={}
        self.full_eeg_data = []# store the whole data in memory
        time_start = time.time()
        # load all splits 
        for phase in self.full_datasets.keys():
            if phase == 'split_info':
                continue
            print('loading phase',phase)            
            print('len full_datasets ', phase ,len(self.full_datasets[phase]['eeg']))
            for i in range(len(self.full_datasets[phase]['eeg'])):
                print('eeg_path',self.full_datasets[phase]['eeg'][i])
            # run in parallel to load the h5 files, but it does work 
            # eeg_data_list = Parallel(n_jobs=n_jobs)(
            #     delayed(_load_h5_raw)(
            #         self.full_datasets[phase]['eeg'][i] for i in range(len(self.full_datasets[phase]['eeg']))
            #     )
            # )

            # for i in range(len(self.full_datasets[phase]['eeg'])):
            #     eeg_path = self.full_datasets[phase]['eeg'][i]# the path to the h5 file
            #     # print('loading eeg_path',eeg_path)
            #     if eeg_path not in self.unique_dataset_mapper.keys():
            #         eeg_data =eeg_data_list[i]                     
            #         self.unique_dataset_mapper[eeg_path] = len(self.full_eeg_data)
            #         self.full_eeg_data.append(eeg_data)

            for i in range(len(self.full_datasets[phase]['eeg'])):
                eeg_path = self.full_datasets[phase]['eeg'][i]# the path to the h5 file
                # print('loading eeg_path',eeg_path)
                if eeg_path not in self.unique_dataset_mapper.keys():
                    t0 = time.time()
                    eeg_data =h5py.File(eeg_path,'r', libver="latest", swmr=True)['eeg'][:]
                    print(len(self.full_eeg_data),'loading data to memory',self.full_datasets[phase]['eeg'][i],eeg_data.shape,'time',time.time()-t0)
                    self.unique_dataset_mapper[eeg_path] = len(self.full_eeg_data)
                    self.full_eeg_data.append(eeg_data)                    
        print('loading all data to memory time',time.time()-time_start)

    def __del__(self):
        # if self.dataset['eeg_h5_handle']
        # if 'eeg_h5_handle' in self.dataset.keys():
        #     for h5_handle in self.dataset['eeg_h5_handle']:
        #         h5_handle.close()
        pass
    def _save_split(self,splits,dataset_meta):
        train_idx = splits['train']
        val_idx = splits['val']
        test_idx = splits['test']
        output_dir = '/projects/SilSpeech/Dev/SilentSpeech_Se2/LBM/LLaBrain/split_checking'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok=True)
            # grant the permission
            os.chmod(output_dir, 0o777)
        output_file = os.path.join(output_dir,f'{self.name}-{self.split_method}-{self.phase}-{str(self.subjects)}.txt')
        with open(output_file,'w') as f:
            f.write(f'train_idx {train_idx}\n')
            # write the feature path too
            for idx in train_idx:
                f.write(f'{dataset_meta["feature_paths"][idx]}\n')
            f.write(f'val_idx {val_idx}\n')
            for idx in val_idx:
                f.write(f'{dataset_meta["feature_paths"][idx]}\n')
            f.write(f'test_idx {test_idx}\n')
            for idx in test_idx:
                f.write(f'{dataset_meta["feature_paths"][idx]}\n')

        print('save split infomation to',output_file)
    def _get_size_from_index_range(self,index_ranges):
        total_size = 0
        for start_idx,end_idx in index_ranges:
            total_size += end_idx-start_idx
        return total_size
    def _get_session_label_from_index_range(self,index_range, session_seq_mapper):
        # given a range of index, return the session index using the mapping of the session shape
        if isinstance(index_range,int):
            for sess_key in session_seq_mapper.keys():
                # print('sess_key',sess_key)
                session_corresponding_index_lists = session_seq_mapper[sess_key]
                for i,seq_id_range in enumerate(session_corresponding_index_lists):
                    if index_range >= seq_id_range[0] and index_range <= seq_id_range[1]:
                        return int(sess_key)
            return None


        else:
            start_idx,end_idx = index_range
            for sess_key in session_seq_mapper.keys():
                # print('sess_key',sess_key)
                session_corresponding_index_lists = session_seq_mapper[sess_key]
                for i,seq_id_range in enumerate(session_corresponding_index_lists):
                    if start_idx >= seq_id_range[0] and end_idx <= seq_id_range[1]:
                        return int(sess_key)
            return None
    def _get_idx_range_from_single_index(self,index,idx_ranges_phase):
        # print('idx_ranges_phase',idx_ranges_phase)
        # given a single index, return the range of the index
        for i,idx_range in enumerate(idx_ranges_phase):
            if int(index) >= int(idx_range[0]) and int(index) <= int(idx_range[1]):
                # print('query',index,'index',index,idx_range,int(index) >= int(idx_range[0]),int(index) <= int(idx_range[1]))
                return i
        # idx_ranges_phase [(0, 1688132), (2110167, 4014407), (4490468, 6660468), (7202969, 8960569), (9399970, 11146930), (11583671, 13941431), (14530872, 16355272), (16811373, 18735773), (19216874, 21063280), (21524883, 23285203), (23725284, 25815284), (26337785, 28144585), (28596286, 30460846), (30926987, 32761387), (33219988, 35150788), (35633489, 36781969)]
        print('query',index,'index',index,idx_range)
        print('idx_ranges_phase',idx_ranges_phase)
        raise ValueError('index not in the range')
        return None
    def _get_subject_label_from_index_range(self,index_range, subject_seq_mapper):
        if isinstance(index_range,int):
            for subj_key in subject_seq_mapper.keys():
                subject_corresponding_index_lists = subject_seq_mapper[subj_key]
                for i,seq_id_range in enumerate(subject_corresponding_index_lists):
                    if index_range >= seq_id_range[0] and index_range <= seq_id_range[1]:
                        return subj_key
            return None
        else:
            # given a range of index, return the subject index using the mapping of the subject shape
            start_idx,end_idx = index_range
            for subj_key in subject_seq_mapper.keys():
                subject_corresponding_index_lists = subject_seq_mapper[subj_key] # this is a list of all sessions of the subject, we just need to know if the index is in one of the session, and return the subject key
                for i,seq_id_range in enumerate(subject_corresponding_index_lists):
                    if start_idx >= seq_id_range[0] and end_idx <= seq_id_range[1]:
                        return subj_key
            return None
    def __len__(self):
        return self._get_size_from_index_range(self.dataset['eeg_index'])
    def __getitem__(self, index):
        real_index =self.dataset['query_index_to_sample_index_mapper'][index]
        range_i_all = self._get_idx_range_from_single_index(real_index,self.dataset['original_sequence_index_list'])
        range_i = self._get_idx_range_from_single_index(real_index,self.dataset['eeg_index'])
        # assert range_i is not None, 'range_index is None'
        feature_file_path = self.dataset['eeg'][range_i]
        start=(real_index-self.dataset['original_sequence_index_list'][range_i_all][0])* self.seq_stride        
        end = (real_index-self.dataset['original_sequence_index_list'][range_i_all][0]) * self.seq_stride  +self.seq_len
        pred=(real_index-self.dataset['original_sequence_index_list'][range_i_all][0])* self.seq_stride +self.seq_len+self.pred_len
        # print("self.dataset['original_sequence_index_list']")
        # print(self.dataset['original_sequence_index_list'])      
        # print('start',start,'real_index',real_index,'-',self.dataset['original_sequence_index_list'][range_i_all][0],'*',self.seq_stride,'=',(real_index-self.dataset['original_sequence_index_list'][range_i_all][0])* self.seq_stride)


        # Each process open the file to avoid inter-process contention
        # with h5py.File(feature_file_path,'r', libver="latest", swmr=True) as h5_file:
        #     eeg_dataset = h5_file['eeg'][:]
        #     # print(eeg_dataset.shape)
        #     eeg = eeg_dataset[:, start:end]   
        #     eeg_pred = eeg_dataset[:, end:pred]    
        #     if self.normalize:
        #         eeg = self._standardize(eeg)
        #         eeg_pred = self._standardize(eeg_pred)
        #     eeg = torch.tensor(eeg).float()    
        #     eeg_pred = torch.tensor(eeg_pred).float()
        # print('real_index',real_index,'is from range_i',range_i,self.dataset['eeg_index'][range_i],  'feature_file_path',self.dataset['eeg'][range_i],start,end,pred)
        
        if self.full_loading:
            eeg_file = self.dataset['eeg'][range_i] # this is the corresponding file path, but we need to find the index of the file path in the unique dataset mapper
            eeg_dataset = self.full_eeg_data[self.unique_dataset_mapper[eeg_file]]
        else:
            # load the data from the file first
            dataset_hanlde = h5py.File(feature_file_path,'r', libver="latest", swmr=True)
            eeg_dataset = dataset_hanlde['eeg'][:]
            # use the preloaded file handle
            # eeg_dataset = self.dataset['eeg_h5_handle'][range_i]['eeg'][:]
        # print('self.unique_dataset_mapper',self.unique_dataset_mapper,'eeg_file',eeg_file,'range_i',range_i)
        # print('query index' , index,' real_index',real_index,'is from range_i',range_i,'range_i_all',range_i_all,self.dataset['eeg_index'][range_i],'feature_file_path',self.dataset['eeg'][range_i],start,end,pred, 'eeg_dataset',eeg_dataset.shape)
        eeg = eeg_dataset[:, start:end]
        # print('query index' , index,' real_index',real_index,'is from range_i',range_i,self.dataset['eeg_index'][range_i],'feature_file_path',self.dataset['eeg'][range_i],start,end,pred, 'eeg_dataset',eeg_dataset.shape)
        if eeg.shape[1] != self.seq_len:
            print('Error', 'query index' , index,' real_index',real_index,'is from range_i',range_i,'range_i_all',range_i_all,self.dataset['eeg_index'][range_i],'feature_file_path',self.dataset['eeg'][range_i],start,end,pred, 'eeg_dataset',eeg_dataset.shape)
            exit(0)

        eeg_pred = eeg_dataset[:, end:pred]
        if self.normalize:
            eeg = self._standardize(eeg)
            eeg_pred = self._standardize(eeg_pred)
        eeg = torch.tensor(eeg).float()
        eeg_pred = torch.tensor(eeg_pred).float()
        # print('real_index',real_index,'is from range_i',range_i,self.dataset['eeg_index'][range_i],  'feature_file_path',self.dataset['eeg'][range_i],eeg_dataset.shape)  
        # print(real_index,self.dataset['original_sequence_index_list'][range_i][0],real_index-self.dataset['original_sequence_index_list'][range_i][0], start)    
        # print('real_index',real_index,'start',start,': end ',end,': pred',pred,'eeg',eeg_dataset.shape, self.__len__())
        # print('real_index',real_index,'query',start,':',end,':',pred,'eeg',eeg.shape,'eeg_pred',eeg_pred.shape)     
        # if eeg.shape[1]!=self.seq_len or eeg_pred.shape[1]!=self.pred_len:
        #     print("self.dataset['eeg_index']",self.dataset['eeg_index'])
        if self.output_dict:
            if self.output_meta_label:
                # get the unique session label for the subject
                unique_session_label = len(self.dataset['session_seq_mapper'].keys())
                # print("self.dataset['session_seq_mapper'].keys()",self.dataset['session_seq_mapper'].keys())
                # print(unique_session_label)                
                subject_name = self._get_subject_label_from_index_range(real_index,self.dataset['subject_seq_mapper'])
                subject_label = self.subjects.index(subject_name)
                session_id =  int(self._get_session_label_from_index_range(real_index,self.dataset['session_seq_mapper'])) -1
                session_label = unique_session_label*subject_label+session_id
                # print("len(unique_session_label)",unique_session_label,"subject_label",subject_label,"session_id",session_id,"session_label",session_label)

                return {
                    'eeg': eeg,
                    'eeg_pred': eeg_pred,
                    'session_label': session_label,
                    'subject_label': subject_label,
                }
            else:
                return {
                    'eeg': eeg,
                    'eeg_pred': eeg_pred,
                }
        else:
            if self.output_meta_label:
                unique_session_label = len(self.dataset['session_seq_mapper'].keys())
                subject_name = self._get_subject_label_from_index_range(real_index,self.dataset['subject_seq_mapper'])
                subject_label = self.subjects.index(subject_name)
                session_id =  int(self._get_session_label_from_index_range(real_index,self.dataset['session_seq_mapper'])) -1
                session_label = unique_session_label*subject_label+session_id
                return eeg,eeg_pred,session_label,subject_label
            else:
                return eeg,eeg_pred
    def get_split(self,dataset_meta):
        # check the dataset_meta is a dictionary
        self._check_dataset_meta_validity(dataset_meta)
        #######################################
        # split the data into train, val, test
        #######################################
        subject_label = dataset_meta['subject_label']
        session_label = dataset_meta['session_label']
        sequence_index_list = dataset_meta['sequence_index_list']
        # print(len(subject_label),len(session_label),len(sequence_index_list))
        
        # print("sequence_index_list",sequence_index_list)
        # print("subject_label",subject_label)
        feature_path = dataset_meta['feature_paths']
        if self.split_method == 'within-session':
            train_idx = []            
            val_idx = []            
            test_idx = []
            # all_idx = []

            train_feature_path = []
            val_feature_path = []
            test_feature_path = []            
            for i,subj in enumerate(self.subjects):
                subj_idx = np.where(np.array(subject_label) == subj)[0]
                # the last 20% of the sequence is used for validation and test
                for idx in subj_idx:
                    start_idx,end_idx = sequence_index_list[idx]
                    train_idx.append((start_idx,int(start_idx+(end_idx-start_idx)*0.8)))
                    train_feature_path.append(feature_path[idx])
                    val_idx.append((int(start_idx+(end_idx-start_idx)*0.8),int(start_idx+(end_idx-start_idx)*0.9)))
                    val_feature_path.append(feature_path[idx])
                    test_idx.append((int(start_idx+(end_idx-start_idx)*0.9),end_idx))              
                    test_feature_path.append(feature_path[idx])
                    # all_idx.append((start_idx,end_idx))
        if self.split_method == 'cross-subject':
            train_idx = []
            val_idx = []
            test_idx = []
            train_feature_path = []
            val_feature_path = []
            test_feature_path = []
            for i,subj in enumerate(self.subjects):
                subj_idx = np.where(np.array(subject_label) == subj)[0]
                for idx in subj_idx:
                    start_idx,end_idx = sequence_index_list[idx]
                    if subj != self.val_subject and subj != self.test_subject:
                        # train_idx.extend(range(start_idx,1+end_idx))
                        train_idx.append((start_idx,end_idx))
                        train_feature_path.append(feature_path[idx])
                    else:
                        if subj == self.val_subject:
                            val_idx.append((start_idx,end_idx))
                            val_feature_path.append(feature_path[idx])
                            # val_idx.extend(range(start_idx,1+end_idx))
                        if subj == self.test_subject:
                            test_idx.append((start_idx,end_idx))
                            test_feature_path.append(feature_path[idx])
                            # test_idx.extend(range(start_idx,1+end_idx))
        if self.split_method == 'leave-one-session-out':          
            train_idx = []
            val_idx = []
            test_idx = []
            train_feature_path = []
            val_feature_path = []
            test_feature_path = []
            # print('session_label',session_label)
            # print('sequence_index_list',sequence_index_list)
            for i,subj in enumerate(self.subjects):
                subj_idx = np.where(np.array(subject_label) == subj)[0]
                # print('subj',subj,'subj_idx',subj_idx)
                for idx in subj_idx: # sessions from the same subject
                    # further get session information
                    session_lbl = session_label[idx]
                    if subj != self.val_subject and subj != self.test_subject:
                        # data from this subejct is used for training
                        start_idx,end_idx =  sequence_index_list[idx]
                        train_idx.append((start_idx,end_idx))
                        train_feature_path.append(feature_path[idx])
                    else:
                        used_in_val_test = False
                        if session_lbl in self.val_sessions:
                            start_idx,end_idx =  sequence_index_list[idx]
                            val_idx.append((start_idx,end_idx))
                            val_feature_path.append(feature_path[idx])
                            used_in_val_test = True
                        if session_lbl in self.test_sessions:
                            start_idx,end_idx =  sequence_index_list[idx]
                            test_idx.append((start_idx,end_idx))
                            test_feature_path.append(feature_path[idx])
                            used_in_val_test = True
                        if not used_in_val_test:
                            start_idx,end_idx =  sequence_index_list[idx]
                            train_idx.append((start_idx,end_idx))
                            train_feature_path.append(feature_path[idx])

            # print('train_feature_path',np.unique(train_feature_path))
            # print('val_feature_path',np.unique(val_feature_path))
            # print('test_feature_path',np.unique(test_feature_path))
            # exit(0)
        if self.split_method == 'no-split':
            train_idx = []
            val_idx = []
            test_idx = []
            train_feature_path = []
            val_feature_path = []
            test_feature_path = []

            for i,subj in enumerate(self.subjects):
                subj_idx = np.where(np.array(subject_label) == subj)[0]
                for idx in subj_idx:
                    start_idx,end_idx = sequence_index_list[idx]
                    # train_idx.extend(range(start_idx,1+end_idx))    
                    if self.phase == 'train':
                        train_idx.append((start_idx,end_idx))
                        train_feature_path.append(feature_path[idx])
                    if self.phase == 'val':
                        val_idx.append((start_idx,end_idx))
                        val_feature_path.append(feature_path[idx])
                    if self.phase == 'test':
                        test_idx.append((start_idx,end_idx))    
                        test_feature_path.append(feature_path[idx])
        index_dict = {
            'train':{
                'range':train_idx,
                'feature_path':train_feature_path},
            'val':{
                'range':val_idx,
                'feature_path':val_feature_path},
            'test':{
                'range':test_idx,
                'feature_path':test_feature_path},    
        }

        len_train = self._get_size_from_index_range(train_idx)  
        len_val = self._get_size_from_index_range(val_idx)
        len_test = self._get_size_from_index_range(test_idx)
        if self.debug:
            print(self.split_method, 'train:',len_train,'val:',len_val,'test:',len_test)       
        return index_dict
class SpeechWordDatasetV5(EEGDatasetBase): # this is a trial-label format dataset. 
    def __init__(
        self,
        phase,
        data_root,
        cache_folder=None,
        split_method='cross-session', 
        subjects=['S05','S07','S08'],
        val_subject='S05',
        val_sessions=[1], 
        test_subject='S07',
        test_sessions=[1],
        select_channels=None,
        select_vocabs=None,
        select_vocabs_remap_index=False,
        vocab_groupping=None,
        preproc_args ={},
        normalize = True,
        random_seed = 0,
        force_rebuild_index= False,
        force_rebuild_feature = False,
        debug = False,
        output_dict = True,
        output_meta_label=False,
        decode_time =1.0, 
        decode_time_shift = None,
        decode_time_shift_distribution = None,
    ):
        super().__init__(
            phase=phase,
            data_root= data_root,
            cache_folder= cache_folder,
            split_method= split_method, 
            subjects= subjects,
            val_subject= val_subject,
            val_sessions= val_sessions, 
            test_subject= test_subject,
            test_sessions= test_sessions,
            select_channels = select_channels,
            select_vocabs= select_vocabs,
            select_vocabs_remap_index= select_vocabs_remap_index,
            vocab_groupping = vocab_groupping,
            preproc_args = preproc_args,
            normalize= normalize,
            random_seed = random_seed,
            force_rebuild_index = force_rebuild_index,
            force_rebuild_feature = force_rebuild_feature,
            debug = debug,
            output_dict = output_dict,
            output_meta_label = output_meta_label,
            decode_time = decode_time,
            decode_time_shift = decode_time_shift,
            decode_time_shift_distribution = decode_time_shift_distribution,
        )
        self.name = 'V5'
        self._cached_dataset()
    def _get_name_cache(self):
        if self.cache_folder is not None:
            return self.cache_folder
        else:
            return os.path.join(self.data_root, 'cache')
    def _init_vocab(self):
        self.marker_to_word_template = {
            '100':'Jumping',
            '101':'Running',
            '102':'Swimming',
            '103':'Going',
            '104':'Happy',
            '105':'Sad',
            '106':'Fun',
            '107':'Horrible',
            '108':'College',
            '109':'Home',
            '110':'Battlefield',
            '111':'Here',
            '112':'Mother',
            '113':'Cowboy',
            '114':'Professor',
            '115':'Me',
            '116':'One',
            '117':'Three',
            '118':'Eleven',
            '119':'Million',
            '120':'Spoon',
            '121':'Alfa',
            '122':'Python',
            '123':'Telephone',
        }
        
        self.marker_to_word={}
        self.word2id = {}
        self.id2word = {}
        relevant_events = [str(e) for e in self.preproc_args['relevant_events']]
        for i, e in enumerate(relevant_events):
            self.marker_to_word[e] = self.marker_to_word_template[e]
            self.word2id[self.marker_to_word[e]] = i
            self.id2word[i] = self.marker_to_word[e]
        # for marker in self.preproc_args['relevant_events']:
        #     if marker in self.marker_to_word_template.keys():
        #         self.marker_to_word[marker] = self.marker_to_word_template[marker]
        # self.word2id = {}
        # self.id2word = {}
        # for k,v in self.marker_to_word.items():
        #     self.word2id[v] = int(k)-100
        #     self.id2word[int(k)-100] = v
    def _get_preproc_and_cache_names(self,prefix = 'xdf'):
        val_sess = [str(i) for i in self.val_sessions]
        test_sess = [str(i) for i in self.test_sessions]
        val_str ='{}'.format(self.val_subject) 
        #'{}s{}'.format(self.val_subject,''.join(val_sess))
        test_str = '{}'.format(self.test_subject)
        #'{}s{}'.format(self.test_subject,''.join(test_sess))
        sub_str= '{}val{}test{}'.format(''.join(self.subjects),val_str,test_str)
        if self.preproc_args['fspecial'] is None:
            feature_folder_name = '{}{}_{}_{}{}_t{}t{}_avgref{}_{}lo{}hi{}n{}{}{}'.format(
                prefix,
                self.preproc_args['format'],
                self.preproc_args['pp_postfix'].replace('.set',''),
                self.preproc_args['feature_type'],
                self.preproc_args['wavelet_method'],
                self.preproc_args['tmin'],self.preproc_args['tmax'], self.preproc_args['avg_ref'],
                self.preproc_args['fspacing'],self.preproc_args['fmin'],self.preproc_args['fmax'],self.preproc_args['fnum'],
                "_fs{}".format(self.preproc_args['resample_fs']),"_ft{}".format(self.preproc_args['resample_freq_time']),
            )            
        else: 
            feature_folder_name = '{}{}_{}_{}{}_t{}t{}_avgref{}_{}{}{}_fspecial{}'.format(
                prefix,
                self.preproc_args['format'],
                self.preproc_args['pp_postfix'].replace('.set',''),
                self.preproc_args['feature_type'],
                self.preproc_args['wavelet_method'],
                self.preproc_args['tmin'],self.preproc_args['tmax'], self.preproc_args['avg_ref'],
                self.preproc_args['fspecial'],
                "_fs{}".format(self.preproc_args['resample_fs']),"_ft{}".format(self.preproc_args['resample_freq_time']),
                self.preproc_args['fspecial'],
                )
        # print('feature_folder_name',feature_folder_name)    
        feature_folder_name = feature_folder_name.replace('None','')
        # print('feature_folder_name',feature_folder_name)    
        if self.preproc_args['debug']:
            feature_folder_name = '_debug'+feature_folder_name
        if self.select_channels is not None:
            channel_str = '_ch{}'.format('_'.join(self.select_channels))
        else:
            channel_str = ''
        if self.select_vocabs is not None:
            vos= [str(v) for v in self.select_vocabs]
            vocab_str = '_vocab{}'.format('_'.join(vos))
        else:
            vocab_str = ''
        cache_path = os.path.join(self.cache_dir, f'{prefix}-{self.split_method}-{feature_folder_name}-{sub_str}-seed{self.random_seed}-{self.phase}-{channel_str}{vocab_str}.pkl')
        return feature_folder_name,cache_path
    def _build(self):
        gc.enable()# enable garbage collection
        dataset_meta = {
            'sentences': [], # contain sentence id, sentence text, subject id
            'words': [],     # subject id, sentence id, word id, word
            'vocab': [],     # word, label, word_freq
        }
        ##############################################
        # read the feature file
        ##############################################
        feature_paths= []
        subject_label = []
        session_label = []
        class_label = []
        channel_names = []
        for i,subject_meta_info in enumerate(self.meta_infos):
            feature_paths.extend(subject_meta_info['features'])
            subject_label.extend(subject_meta_info['subject_labels'])
            session_label.extend(subject_meta_info['session_labels'])
            class_label.extend(subject_meta_info['label'])
            channel_names = subject_meta_info['channel_names']
        class_label = np.array(class_label)
        
        class_label = np.array(class_label)
        # convert session_label to integer
        session_label = [int(s) for s in session_label]
        session_label = np.array(session_label)
        subject_label = np.array(subject_label) 
        feature_paths = np.array(feature_paths)
        dataset_meta['subject_label'] = subject_label
        dataset_meta['session_label'] = session_label
        dataset_meta['class_label']   = class_label
        dataset_meta['channel_names'] = channel_names
        dataset_meta['feature_paths'] = feature_paths
        index_dict = self.get_split(dataset_meta)
        
        split_idx = index_dict[self.phase]
        label_phase=class_label[split_idx]
        feature_phase=feature_paths[split_idx]
        session_phase=session_label[split_idx]
        subject_phase=subject_label[split_idx]
        
        label_phase = np.array(label_phase)
        session_phase = np.array(session_phase)
        subject_phase = np.array(subject_phase)
        
        feature_phase = np.array(feature_phase)
        
        dataset = {
            "eeg": feature_phase,
            "label": label_phase,
            "session": session_phase,
            "subject": subject_phase,
            "channel_names": channel_names,
            }
        print('before label mapping',np.unique(dataset['label']))
        dataset['label'] = dataset['label'] - min(dataset['label']) # make the label start from 0
        print('after label mapping',np.unique(dataset['label']))
        
        print('relevant words',self.preproc_args['relevant_events'])
        
        self.final_word2id ={}
        self.final_id2word = {}
        if self.select_vocabs is not None:
            vocab_idx =self._filter_vocabs(dataset,self.select_vocabs)
            print('len vocab_idx after select_vocabs',len(vocab_idx))            
            dataset['label'] = dataset['label'][vocab_idx]
            print('unique labels after vocab selection',np.unique(dataset['label']))
            if self.select_vocabs_remap_index:
                label_mappping = {}
                for i, label in enumerate(self.select_vocabs):
                    original_word_of_the_label = self.id2word[label]
                    label_mappping[label] = i
                    self.final_word2id[original_word_of_the_label] = i
                    self.final_id2word[i] = original_word_of_the_label
            #         print('label remapping','label',label,'to',i)
            # print('word2id',self.word2id)
            # print('id2word',self.id2word)
            dataset['eeg'] = dataset['eeg'][vocab_idx]
            dataset['label'] = np.array([label_mappping[l] for l in dataset['label']])
            dataset['session'] = dataset['session'][vocab_idx]
            dataset['subject'] = dataset['subject'][vocab_idx] 
        else:
            for label in np.unique(dataset['label']):
                original_word_of_the_label = self.id2word[label]
                self.final_word2id[original_word_of_the_label] = label
                self.final_id2word[label] = original_word_of_the_label
        print('final_word2id',self.final_word2id)
        print('final_id2word',self.final_id2word)

        print('>>>>>>>>>>>> EEG data {} >>>>>>>>>>>>>'.format(self.phase))
        print('feature_phase',dataset['eeg'].shape)
        print('label_phase',dataset['label'].shape)
        print('session_phase',dataset['session'].shape)
        print('subject_phase',dataset['subject'].shape)
        print('unique class_label',np.unique(dataset['label']))
        print('unique session_label',np.unique(dataset['session']))
        print('unique subject_label',np.unique(dataset['subject']))
        return dataset
    def keep_session(self,sessions):
        print('Only keep Session',sessions,'from dataset')
        # only keep the sessions in the list using the dataset
        index_keep = []
        for i,session in enumerate(self.dataset['session']):
            if session in sessions:
                index_keep.append(i)
        self.dataset['eeg'] = self.dataset['eeg'][index_keep]
        self.dataset['label'] = self.dataset['label'][index_keep]
        self.dataset['session'] = self.dataset['session'][index_keep]
        self.dataset['subject'] = self.dataset['subject'][index_keep]

        print(f'>>>>>>>>>>>> keep_session {sessions} >>>>>>>>>>>>>')
        print('feature_phase',self.dataset['eeg'].shape)
        print('label_phase',self.dataset['label'].shape)
        print('session_phase',self.dataset['session'].shape)
        print('subject_phase',self.dataset['subject'].shape)
        print('unique class_label',np.unique(self.dataset['label']))
        print('unique session_label',np.unique(self.dataset['session']))
        print('unique subject_label',np.unique(self.dataset['subject']))
    def keep_subject(self,subjects):
        # only keep the subjects in the list using the dataset
        print('Only keep Subject',subjects,'from dataset')
        index_keep = []
        for i,subject in enumerate(self.dataset['subject']):
            if subject in subjects:
                index_keep.append(i)
        self.dataset['eeg'] = self.dataset['eeg'][index_keep]
        self.dataset['label'] = self.dataset['label'][index_keep]
        self.dataset['session'] = self.dataset['session'][index_keep]
        self.dataset['subject'] = self.dataset['subject'][index_keep]

        print(f'>>>>>>>>>>>> keep_subject {subjects} >>>>>>>>>>>>>')
        print('feature_phase',self.dataset['eeg'].shape)
        print('label_phase',self.dataset['label'].shape)
        print('session_phase',self.dataset['session'].shape)
        print('subject_phase',self.dataset['subject'].shape)
        print('unique class_label',np.unique(self.dataset['label']))
        print('unique session_label',np.unique(self.dataset['session']))
        print('unique subject_label',np.unique(self.dataset['subject']))
    def keep_session_subject(self,sub_sess_list):
        # the first element is the subject and the second element is the session
        print('Only keep Subject and Session',sub_sess_list,'from dataset')
        index_keep = []
        for i,sub_sess in enumerate(zip(self.dataset['subject'],self.dataset['session'])):
            # print('sub_sess',sub_sess)
            if sub_sess in sub_sess_list:
                index_keep.append(i)
        self.dataset['eeg'] = self.dataset['eeg'][index_keep]
        self.dataset['label'] = self.dataset['label'][index_keep]
        self.dataset['session'] = self.dataset['session'][index_keep]
        self.dataset['subject'] = self.dataset['subject'][index_keep]
        print(f'>>>>>>>>>>>> keep_session_subject {sub_sess_list} >>>>>>>>>>>>>')
        print('feature_phase',self.dataset['eeg'].shape)
        print('label_phase',self.dataset['label'].shape)
        print('session_phase',self.dataset['session'].shape)
        print('subject_phase',self.dataset['subject'].shape)
        print('unique class_label',np.unique(self.dataset['label']))
        print('unique session_label',np.unique(self.dataset['session']))
        print('unique subject_label',np.unique(self.dataset['subject']))
    def get_session(self,sessions):
        # get the session from the dataset
        # return a subset 
        result_dataset = {}
        for key in self.dataset.keys():
            result_dataset[key] = []
        result_dataset['channel_names'] = self.dataset['channel_names']
        for i,session in enumerate(self.dataset['session']):
            if session in sessions:
                result_dataset['eeg'].append(self.dataset['eeg'][i])
                result_dataset['label'].append(self.dataset['label'][i])
                result_dataset['session'].append(self.dataset['session'][i])
                result_dataset['subject'].append(self.dataset['subject'][i])
        result_dataset['eeg'] = np.array(result_dataset['eeg'])
        result_dataset['label'] = np.array(result_dataset['label'])
        result_dataset['session'] = np.array(result_dataset['session'])
        result_dataset['subject'] = np.array(result_dataset['subject'])
        # print(f'>>>>>>>>>>>> get_session {sessions} >>>>>>>>>>>>>')
        # print('feature_phase',result_dataset['eeg'].shape)
        # print('label_phase',result_dataset['label'].shape)
        # print('session_phase',result_dataset['session'].shape)
        # print('subject_phase',result_dataset['subject'].shape)
        # print('unique class_label',np.unique(result_dataset['label']))
        # print('unique session_label',np.unique(result_dataset['session']))
        # print('unique subject_label',np.unique(result_dataset['subject']))
        return result_dataset
    def get_subject(self,subjects):
        # get the subject from the dataset
        # return a subset 
        result_dataset = {}
        for key in self.dataset.keys():
            result_dataset[key] = []
        result_dataset['channel_names'] = self.dataset['channel_names']
        for i,subject in enumerate(self.dataset['subject']):
            if subject in subjects:
                result_dataset['eeg'].append(self.dataset['eeg'][i])
                result_dataset['label'].append(self.dataset['label'][i])
                result_dataset['session'].append(self.dataset['session'][i])
                result_dataset['subject'].append(self.dataset['subject'][i])
        result_dataset['eeg'] = np.array(result_dataset['eeg'])
        result_dataset['label'] = np.array(result_dataset['label'])
        result_dataset['session'] = np.array(result_dataset['session'])
        result_dataset['subject'] = np.array(result_dataset['subject'])
        # print(f'>>>>>>>>>>>> get_subject {subjects} >>>>>>>>>>>>>')
        # print('feature_phase',result_dataset['eeg'].shape)
        # print('label_phase',result_dataset['label'].shape)
        # print('session_phase',result_dataset['session'].shape)
        # print('subject_phase',result_dataset['subject'].shape)
        # print('unique class_label',np.unique(result_dataset['label']))
        # print('unique session_label',np.unique(result_dataset['session']))
        # print('unique subject_label',np.unique(result_dataset['subject']))
        return result_dataset
    def get_session_subject(self,sub_sess_list):
        # the first element is the subject and the second element is the session
        # get the session and subject from the dataset
        # return a subset 
        result_dataset = {}
        for key in self.dataset.keys():
            result_dataset[key] = []
        result_dataset['channel_names'] = self.dataset['channel_names']
        for i,sub_sess in enumerate(zip(self.dataset['subject'],self.dataset['session'])):
            # print('sub_sess',sub_sess)
            if sub_sess in sub_sess_list:
                result_dataset['eeg'].append(self.dataset['eeg'][i])
                result_dataset['label'].append(self.dataset['label'][i])
                result_dataset['session'].append(self.dataset['session'][i])
                result_dataset['subject'].append(self.dataset['subject'][i])
                
        result_dataset['eeg'] = np.array(result_dataset['eeg'])
        result_dataset['label'] = np.array(result_dataset['label'])
        result_dataset['session'] = np.array(result_dataset['session'])
        result_dataset['subject'] = np.array(result_dataset['subject'])
        # print(f'>>>>>>>>>>>> get_session_subject {sub_sess_list} >>>>>>>>>>>>>')
        # print('feature_phase',result_dataset['eeg'].shape)
        # print('label_phase',result_dataset['label'].shape)
        # print('session_phase',result_dataset['session'].shape)
        # print('subject_phase',result_dataset['subject'].shape)
        # print('unique class_label',np.unique(result_dataset['label']))
        # print('unique session_label',np.unique(result_dataset['session']))
        # print('unique subject_label',np.unique(result_dataset['subject']))
        return result_dataset
def TrainingDatasetClustering(train_dataset, test_dataset,mectrics=['euclidean'],cluster_method='gmm',n_clusters=10, visualize=True, n_jobs=16, debug=False):
    print('train dataset size',len(train_dataset))
    print('test dataset size',len(test_dataset))
    print('TrainingDatasetClustering Extracting Features from target Dataset...')
    block_feature_target = [] # feature for each session in the target dataset
    block_sub_sess_target = []
    for sub in test_dataset.unique_subjects:
        for sess in test_dataset.unique_sessions:
            print('processing subject',sub,'session',sess)            
            block = test_dataset.get_session_subject([(sub,sess)])
            # check if the block is empty
            if len(block['eeg']) == 0:
                print('block is empty')
                continue
            if debug:
                block_trials=Parallel(n_jobs=n_jobs)(delayed(compute_band_power)(test_dataset, block['eeg'][i]) for i in range(10))
            else:
                block_trials=Parallel(n_jobs=n_jobs)(delayed(compute_band_power)(test_dataset, block['eeg'][i]) for i in range(len(block['eeg'])))
            block_mean = np.mean(block_trials,axis=0)
            block_feature_target.append(block_mean)
            block_sub_sess_target.append((sub,sess))
    block_feature_target = np.array(block_feature_target)
    print('block_feature_target',block_feature_target.shape)
    # mdm_matrix = np.array([compute_distances(sub, subject_features) for sub in subject_features])    
    print('TrainingDatasetClustering Extracting Features from source Dataset...')
    block_feature_source = [] # feature for each session in the target dataset
    block_sub_sess_source = []
    for sub in train_dataset.unique_subjects:
        for sess in train_dataset.unique_sessions:
            print('processing subject',sub,'session',sess)            
            block = train_dataset.get_session_subject([(sub,sess)])
            # check if the block is empty
            if len(block['eeg']) == 0:
                print('block is empty')
                continue
            if debug:
                block_trials=Parallel(n_jobs=n_jobs)(delayed(compute_band_power)(train_dataset, block['eeg'][i]) for i in range(10))
            else:
                block_trials=Parallel(n_jobs=n_jobs)(delayed(compute_band_power)(train_dataset, block['eeg'][i]) for i in range(len(block['eeg'])))
            block_mean = np.mean(block_trials,axis=0)
            block_feature_source.append(block_mean)
            block_sub_sess_source.append((sub,sess))
    block_feature_source = np.array(block_feature_source)
    print('block_feature_source',block_feature_target.shape)
    dataset_features = np.concatenate((block_feature_source, block_feature_target), axis=0)
    print('dataset_features',dataset_features.shape)
    distances = []
    for mec in mectrics:
        distance = [compute_distances(mec, block_feature, dataset_features) for block_feature in dataset_features]
        distances.append(distance)
    mdm_matrix = np.concatenate(distances, axis=0)
    # print('mdm_matrix',mdm_matrix.shape)
    # transpose the mdm_matrix to (n_samples, n_features)
    mdm_matrix = np.transpose(mdm_matrix)
    # print('mdm_matrix',mdm_matrix.shape)
    mdm_matrix = mdm_matrix.reshape(len(dataset_features), -1)
    # print('mdm_matrix',mdm_matrix.shape)

    print("MDM Matrix Stats:")
    print("Min:", np.min(mdm_matrix))
    print("Max:", np.max(mdm_matrix))
    print("Mean:", np.mean(mdm_matrix))
    print("Std Dev:", np.std(mdm_matrix))
    print("Contains NaN:", np.any(np.isnan(mdm_matrix)))
    print("Contains Inf:", np.any(np.isinf(mdm_matrix)))
    
    scaler = sklearn.preprocessing.StandardScaler()
    mdm_matrix = scaler.fit_transform(mdm_matrix)

    block_clusters, gmm_model = cluster_subjects(mdm_matrix, max_clusters=n_clusters, cluster_method=cluster_method)
    print("Cluster assignments:", block_clusters)
    # Cluster assignments: [1 1 2 2 2 2 0 0 0 0 2 2 2 2 1]

    # get the cluster of the target dataset
    target_block_clusters = block_clusters[:len(block_sub_sess_target)]
    source_block_clusters = block_clusters[len(block_sub_sess_target):]
    relevant_training_block_indexes = [ i for i,c in enumerate(source_block_clusters) if c in target_block_clusters]
    relevant_training_subject_sessions = [block_sub_sess_source[i] for i in relevant_training_block_indexes]
    print('relevant_training_subject_sessions',relevant_training_subject_sessions)
    # visualization 
    if visualize:
        print("Visualizing clusters...")
        pca = sklearn.decomposition.PCA(n_components=2)
        pca_result = pca.fit_transform(mdm_matrix)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=block_clusters, palette='viridis', s=60)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Subject Clustering (eldBETA Dataset)")
        plt.legend(title="Clusters")
        # put some text on the figure 
        text1 = "target_block_clusters: {}".format(target_block_clusters)
        text2 = "source_block_clusters: {}".format(source_block_clusters)
        text3 = "relevant_training_subject_sessions: {}".format(relevant_training_subject_sessions)
        plt.text(0.5, 0.9, text1, fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
        plt.text(0.5, 0.8, text2, fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
        plt.text(0.5, 0.7, text3, fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)

        fig_save_path = os.path.join('./figures', 'subject_clustering.png')
        if not os.path.exists('./figures'):
            os.makedirs('./figures',exist_ok=True)
        plt.savefig(fig_save_path)
    return relevant_training_subject_sessions    
class SpeechTSDatasetV2(EEGTSDatasetBase): # Stride is actually valid for this dataset.
    def __init__(
        self,
        phase,
        data_root,
        cache_folder=None,
        split_method='cross-session', 
        subjects=['S05'],
        val_subject='S05',
        val_sessions=[1], 
        test_subject='S07',
        test_sessions=[1],
        seq_len = 500,
        pred_len = 100, 
        seq_stride = 100,
        select_channels=None,
        select_vocabs=None,
        select_vocabs_remap_index=False,
        preproc_args ={},
        normalize = True,
        random_seed = 0,
        force_rebuild_index= False,
        force_rebuild_feature = False,
        full_loading=False,
        debug = False,
        output_dict = True,
        output_meta_label = False,
    ):
        super().__init__(
            phase=phase,
            data_root=data_root,
            cache_folder=cache_folder,
            split_method=split_method, 
            subjects=subjects,
            val_subject=val_subject,
            val_sessions=val_sessions, 
            test_subject=test_subject,
            test_sessions=test_sessions,
            seq_len =seq_len,
            pred_len = pred_len,
            seq_stride = seq_stride,
            select_channels=select_channels,
            select_vocabs  =select_vocabs,
            select_vocabs_remap_index=select_vocabs_remap_index,
            preproc_args =preproc_args,
            normalize = normalize,
            random_seed = random_seed,
            force_rebuild_index= force_rebuild_index,
            force_rebuild_feature = force_rebuild_feature,
            full_loading = full_loading,
            debug = debug,
            output_dict = output_dict,
            output_meta_label = output_meta_label,
        )
        self.name = 'TSV2'
        self._cached_dataset()
    def _check_dataset_meta_validity(self,dataset_meta):
        # check the dataset_meta is a dictionary
        assert isinstance(dataset_meta,dict), '[EEGDatasetBase.get_split] dataset_meta should be a dictionary'
        assert 'session_label' in dataset_meta.keys(), '[EEGDatasetBase.get_split] dataset_meta should have key session_label'
        assert 'subject_label' in dataset_meta.keys(), '[EEGDatasetBase.get_split] dataset_meta should have key subject_label'
        #assert same length
        assert len(dataset_meta['session_label']) == len(dataset_meta['subject_label']), '[EEGDatasetBase.get_split] session_label, subject_label should have the same length'
    def _get_name_cache(self):
        if self.cache_folder is not None:
            return self.cache_folder
        else:
            return os.path.join(self.data_root, 'cache')
    def _get_preproc_and_cache_names(self,prefix = 'xdf'):
        val_sess = [str(i) for i in self.val_sessions]
        test_sess = [str(i) for i in self.test_sessions]
        val_str ='{}'.format(self.val_subject) 
        #  '{}s{}'.format(self.val_subject,'_'.join(val_sess))
        test_str = '{}'.format(self.test_subject)
        # '{}s{}'.format(self.test_subject,'_'.join(test_sess))
        sub_str= '{}_val{}_test{}'.format(''.join(self.subjects),val_str,test_str)
        feature_folder_name = '{}{}_{}_{}{}_t{}t{}_avgref-{}_{}lo{}hi{}n{}{}{}'.format(
            prefix,
            self.preproc_args['format'],
            self.preproc_args['pp_postfix'].replace('.set',''),
            self.preproc_args['feature_type'],
            self.preproc_args['wavelet_method'],
            self.preproc_args['tmin'],self.preproc_args['tmax'], self.preproc_args['avg_ref'],
            self.preproc_args['fspacing'],self.preproc_args['fmin'],self.preproc_args['fmax'],self.preproc_args['fnum'],
            "_fs{}".format(self.preproc_args['resample_fs']),"_ft{}".format(self.preproc_args['resample_freq_time']),
        ) 
        feature_folder_name = feature_folder_name.replace('None','')
        # print('feature_folder_name',feature_folder_name)    
        if self.preproc_args['debug']:
            feature_folder_name = '_debug'+feature_folder_name
        cache_path = os.path.join(self.cache_dir, f'{prefix}-{self.split_method}-{feature_folder_name}-s{self.seq_len}-p{self.pred_len}-s{self.seq_stride}-{sub_str}-seed{self.random_seed}.pkl') # -{self.phase}
        return feature_folder_name,cache_path
    def _build(self):
        # if phase is not None:
        #     self.phase = phase
        gc.enable()
        dataset_meta = {
            'sentences': [], # contain sentence id, sentence text, subject id
            'words': [],     # subject id, sentence id, word id, word
            'vocab': [],     # word, label, word_freq
            'session_seq_mapper':{},
            'subject_seq_mapper':{},
        }
        feature_paths=  []
        subject_label = []
        session_label = []
        channel_names = []
        sequence_index_list = [] # start and end index of the sequence.         
        cur_seq_idx = 0
        for i,subject_meta_info in enumerate(self.meta_infos):            
            subject_label.extend(subject_meta_info['subject_labels'])
            session_label.extend(subject_meta_info['session_labels'])
            channel_names = subject_meta_info['channel_names']
            # make a index mapping for the session. 
            # for each session, the sequence_index will extend 
            for j in range(len(subject_meta_info['features_h5'])):
                subj = subject_label[i*len(subject_meta_info['features_h5'])+j]
                feature_paths.append(subject_meta_info['features_h5'][j])
                # load this h5 file to get the shape of the feature
                with h5py.File(subject_meta_info['features_h5'][j], 'r') as h5file:
                    session_dataset = h5file['eeg']
                    session_seq_shape_h5 = session_dataset.shape# this is the shape after resampling, so this one is correct to use
                # session_seq_shape = subject_meta_info['session_shape'][j] # this is the shape of the session data before resampling, so sampling rate is 1000Hz. 
                # print('session_seq_shape_h5',session_seq_shape_h5,'session_seq_shape',session_seq_shape)
                session_seq_len  = session_seq_shape_h5[1] #session_seq_shape[1]
                useable_sequence_number = (session_seq_len - self.seq_len - self.pred_len)//self.seq_stride
                # if self.debug:
                # print('session',j,'session_seq_len',session_seq_len,' useable_sequence_number',useable_sequence_number)
                seq_id_range = [cur_seq_idx,cur_seq_idx+useable_sequence_number]
                # add to session_seq_mapper and subject_seq_mapper
                if session_label[j] not in dataset_meta['session_seq_mapper'].keys():
                    dataset_meta['session_seq_mapper'][session_label[j]] = []
                dataset_meta['session_seq_mapper'][session_label[j]].append(seq_id_range)
                if subject_label[i*len(subject_meta_info['features_h5'])+j] not in dataset_meta['subject_seq_mapper'].keys():                    
                    dataset_meta['subject_seq_mapper'][subj] = []
                dataset_meta['subject_seq_mapper'][subj].append(seq_id_range)
                sequence_index_list.append(seq_id_range)
                cur_seq_idx += useable_sequence_number+1 
                if self.debug:
                    print('seq_id_range',seq_id_range) # this list is still all the sequence index, not the start and end index of the sequence.
        if self.debug:
            print('sequence_index_list',sequence_index_list)

        print("dataset_meta['subject_seq_mapper']",dataset_meta['subject_seq_mapper'])
        print("dataset_meta['session_seq_mapper']",dataset_meta['session_seq_mapper'])
        print('unique session_label',np.unique(session_label))
        print('unique subject_label',np.unique(subject_label))
        # print('subject_label',subject_label)

        # convert session_label to integer
        session_label = [int(s) for s in session_label]
        session_label = np.array(session_label)
        subject_label = np.array(subject_label) 
        dataset_meta['subject_label'] = subject_label
        dataset_meta['session_label'] = session_label
        dataset_meta['channel_names'] = channel_names
        dataset_meta['feature_paths'] = feature_paths
        dataset_meta['sequence_index_list'] = sequence_index_list


        
        index_dict = self.get_split(dataset_meta) 
        full_datasets = {
            'train':{},
            'val':{},
            'test':{},
        }
        print('index_dict',index_dict['val'])
        # exit(0)
        for phase in full_datasets.keys():
                   
            range_phase = index_dict[phase]['range']
            feature_phase = index_dict[phase]['feature_path']
            session_phase=[]
            subject_phase=[]
            sample_index_phase = []
            # print('feature_paths',dataset_meta['feature_paths'])
            for idx_range in range_phase:
                session_lbl = self._get_session_label_from_index_range(idx_range,dataset_meta['session_seq_mapper'])
                subject_lbl = self._get_subject_label_from_index_range(idx_range,dataset_meta['subject_seq_mapper'])
                session_phase.append(session_lbl)
                subject_phase.append(subject_lbl)
                sample_index_phase.extend(range(idx_range[0],idx_range[1])) # records the index of the sample index
            if self.debug:
                print('feature_phase',feature_phase)
                print('session_phase',session_phase)
                print('subject_phase',subject_phase)
            # print('sample_index_phase')
            # if phase=='val':
            #     print(sample_index_phase)
            #     # exit(0)
            # print(sample_index_phase)
            # exit(0)
            # TODO: get the chunk size from the h5 file or from the function. 
            dataset = {
                "eeg": feature_phase,
                "eeg_index": range_phase,  # record the start and end index of the feature
                "session": session_phase,
                "subject": subject_phase,
                "channel_names": channel_names,
                "query_index_to_sample_index_mapper":sample_index_phase,  # record the index of the query index to the sample index
                "original_sequence_index_list":dataset_meta['sequence_index_list'], # record each raw EEG session start and end index
                'session_seq_mapper':dataset_meta['session_seq_mapper'],
                'subject_seq_mapper':dataset_meta['subject_seq_mapper'],
                'h5_chunk_size': (122,1000)
                }
            full_datasets[phase] = dataset
            if self.debug:
                print('>>>>>>>>>>>> EEG data {} >>>>>>>>>>>>>'.format(phase))
                print('feature_phase',len(dataset['eeg']))
                print('eeg_index_phase',len(dataset['eeg_index']))
                print('session_phase',len(dataset['session']))
                print('subject_phase',len(dataset['subject']))
                print('sample_index_phase',len(sample_index_phase))
                print('actual size', self._get_size_from_index_range(range_phase))
        full_datasets['split_info'] = index_dict
        return full_datasets
class SpeechWordDatasetFewshotMeta(EEGDatasetBase):
    def __init__(
        self,
        phase,
        data_root,
        split_method='cross-session', 
        subjects=['S05','S07','S08'],
        val_subject='S05',
        val_sessions=[1], 
        test_subject='S07',
        test_sessions=[1],
        select_channels=None,
        select_vocabs =None,
        preproc_args = {},
        normalize = True,
        random_seed = 0,
        force_rebuild_index= False,
        force_rebuild_feature = False,
        K_shot = 5,
        K_query = 5,
        debug = False,
    ):
        super().__init__(
            phase,
            data_root,
            split_method, 
            subjects,
            val_subject,
            val_sessions, 
            test_subject,
            test_sessions,
            select_channels,
            select_vocabs,
            preproc_args,
            normalize,
            random_seed,
            force_rebuild_index,
            force_rebuild_feature,
            debug,
        )
        self.name =  'SpeechWordDatasetFewshotMeta'
        self._cached_dataset()

        self.n_way = len(np.unique(self.dataset['label']))
        self.k_shot = K_shot
        self.k_query = K_query
        self.num_sessions = len(np.unique(self.dataset['session']))
        self.num_subjects = len(np.unique(self.dataset['subject']))
        print('n_way',self.n_way)
        print('k_shot',self.k_shot)
        print('k_query',self.k_query)
        print('sessions',self.num_sessions)
        print('subjects',self.num_subjects)
    def _get_name_cache(self):
        return os.path.join(self.data_root, 'cache')
    def _init_vocab(self):
        self.marker_to_word = {
            '100':'Jumping',
            '101':'Running',
            '102':'Swimming',
            '103':'Going',
            '104':'Happy',
            '105':'Sad',
            '106':'Fun',
            '107':'Horrible',
            '108':'College',
            '109':'Home',
            '110':'Battlefield',
            '111':'Here',
            '112':'Mother',
            '113':'Cowboy',
            '114':'Professor',
            '115':'Me',
            '116':'One',
            '117':'Three',
            '118':'Eleven',
            '119':'Million',
            '120':'Spoon',
            '121':'Alfa',
            '122':'Python',
            '123':'Telephone',
        }
        self.word2id = {}
        self.id2word = {}
        for i,word in enumerate(self.marker_to_word.values()):
            self.word2id[word] = i-100
            self.id2word[i-100] = word
    def _get_name_cache(self):
        return os.path.join(self.data_root, 'cache')
    def _get_preproc_and_cache_names(self,prefix = 'xdf'):
        val_sess = [str(i) for i in self.val_sessions]
        test_sess = [str(i) for i in self.test_sessions]
        val_str = '{}s{}'.format(self.val_subject,'_'.join(val_sess))
        test_str = '{}s{}'.format(self.test_subject,'_'.join(test_sess))
        sub_str= '{}_val{}_test{}'.format('_'.join(self.subjects),val_str,test_str)
        if self.preproc_args['fspecial'] is None:
            feature_folder_name = '{}{}_{}{}_t{}t{}_avgref-{}_{}lo{}hi{}n{}{}{}'.format(
                prefix,
                self.preproc_args['format'],
                self.preproc_args['feature_type'],
                self.preproc_args['wavelet_method'],
                self.preproc_args['tmin'],self.preproc_args['tmax'], self.preproc_args['avg_ref'],
                self.preproc_args['fspacing'],self.preproc_args['fmin'],self.preproc_args['fmax'],self.preproc_args['fnum'],
                "_fs{}".format(self.preproc_args['resample_fs']),"_ft{}".format(self.preproc_args['resample_freq_time'])
            )            
        else: 
            feature_folder_name = '{}{}_{}{}_t{}t{}_avgref{}_{}{}{}_fspecial{}'.format(
                prefix,
                self.preproc_args['format'],
                self.preproc_args['feature_type'],
                self.preproc_args['wavelet_method'],
                self.preproc_args['tmin'],self.preproc_args['tmax'], self.preproc_args['avg_ref'],
                self.preproc_args['fspecial'],
                "_fs{}".format(self.preproc_args['resample_fs']),"_ft{}".format(self.preproc_args['resample_freq_time']),
                self.preproc_args['fspecial'] )
        # print('feature_folder_name',feature_folder_name)    
        feature_folder_name = feature_folder_name.replace('None','')
        # print('feature_folder_name',feature_folder_name)    
        if self.preproc_args['debug']:
            feature_folder_name = '_debug'
        cache_path = os.path.join(self.cache_dir, f'{prefix}-{self.split_method}-{feature_folder_name}-{sub_str}-seed{self.random_seed}.pkl')
        return feature_folder_name,cache_path
    def _build(self):
        gc.enable()# enable garbage collection
        dataset_meta = {
            'sentences': [], # contain sentence id, sentence text, subject id
            'words': [],     # subject id, sentence id, word id, word
            'vocab': [],     # word, label, word_freq
        }
        ##############################################
        # read the feature file
        ##############################################
        feature_paths= []
        subject_label = []
        session_label = []
        class_label = []
        channel_names = []
        for i,subject_meta_info in enumerate(self.meta_infos):
            feature_paths.extend(subject_meta_info['features'])
            subject_label.extend(subject_meta_info['subject_labels'])
            session_label.extend(subject_meta_info['session_labels'])
            class_label.extend(subject_meta_info['label'])
            channel_names = subject_meta_info['channel_names']
        class_label = np.array(class_label)
        print('unique class_label',np.unique(class_label))
        print('unique session_label',np.unique(session_label))
        print('unique subject_label',np.unique(subject_label))

        class_label = np.array(class_label)
        # convert session_label to integer
        session_label = [int(s) for s in session_label]
        session_label = np.array(session_label)
        subject_label = np.array(subject_label) 
        feature_paths = np.array(feature_paths)
        dataset_meta['subject_label'] = subject_label
        dataset_meta['session_label'] = session_label
        dataset_meta['class_label']   = class_label
        dataset_meta['channel_names'] = channel_names
        dataset_meta['feature_paths'] = feature_paths


        index_dict = self.get_split(dataset_meta)

        
        split_idx = index_dict[self.phase]
        label_phase=class_label[split_idx]
        feature_phase=feature_paths[split_idx]
        session_phase=session_label[split_idx]
        subject_phase=subject_label[split_idx]
        
        label_phase = np.array(label_phase)
        session_phase = np.array(session_phase)
        subject_phase = np.array(subject_phase)
        
        feature_phase = np.array(feature_phase)
        dataset = {
            "eeg": feature_phase,
            "label": label_phase,
            "session": session_phase,
            "subject": subject_phase,
            "channel_names": channel_names,
            }
        if self.select_channels is not None:
            raise NotImplementedError
        if self.select_vocabs is not None:
            vocab_idx =self._filter_vocabs(dataset,self.select_vocabs)
            print('len vocab_idx after select_vocabs',len(vocab_idx))
            
            dataset['label'] = dataset['label'][vocab_idx]
            print('unique labels after vocab selection',np.unique(dataset['label']))
            label_mappping = {}
            for i, label in enumerate(self.select_vocabs):
                label_mappping[label] = i
            dataset['eeg'] = dataset['eeg'][vocab_idx]
            dataset['label'] = np.array([label_mappping[l] for l in dataset['label']])
            dataset['session'] = dataset['session'][vocab_idx]
            dataset['subject'] = dataset['subject'][vocab_idx]             

        print('>>>>>>>>>>>> EEG data {} >>>>>>>>>>>>>'.format(self.phase))
        print('feature_phase',dataset['eeg'].shape)
        print('label_phase',dataset['label'].shape)
        print('session_phase',dataset['session'] .shape)
        print('subject_phase',dataset['subject'].shape)
        return dataset

    def __len__(self):
        return len(self.dataset['label'])
    def __getitem__(self, index):
        support_x = [] 
        support_y = []
        query_x = []
        query_y = []
        # get the session and label for the index 
        item_session = self.dataset['session'][index]
        item_label = self.dataset['label'][index]

        # now sample self.k_shot+ self.k_query samples from the same session and label in the dataset
        task_index_list = np.where(self.dataset['session']==item_session)[0]
        # print('task_index_list',task_index_list)
        classes_index_list = np.where(self.dataset['label']==item_label)[0]
        # print('classes_index_list',classes_index_list)
        task_classes_index_list = np.intersect1d(task_index_list,classes_index_list)
        # randomly select k_shot + k_query samples from the task_classes_index_list
        selected_eeg_index = np.random.choice(task_classes_index_list, self.k_shot + self.k_query, False)

        #support set
        support_x_paths =self.dataset['eeg'][selected_eeg_index[:self.k_shot]]
        support_x.extend([self._load_eeg(eeg_path,normalize=self.normalize) for eeg_path in support_x_paths])
        support_y.extend([item_label]*self.k_shot)
        # query set
        query_x_paths =self.dataset['eeg'][selected_eeg_index[self.k_shot:]]
        query_x.extend([self._load_eeg(eeg_path,normalize=self.normalize) for eeg_path in query_x_paths])
        query_y.extend([item_label]*self.k_query)

        support_x = np.array(support_x)
        support_y = np.array(support_y)
        query_x = np.array(query_x)
        query_y = np.array(query_y)
        # print('support_x',support_x.shape,'support_y',support_y.shape,'query_x',query_x.shape,'query_y',query_y.shape)
        return support_x, support_y, query_x, query_y
    def get_batch(self,batchsz):
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        # np.zeros((self.batchsz[mode], setsz) + self.eeg_shape)        
        support_x = [] 
        # np.zeros((self.batchsz[mode], setsz), dtype=int)
        support_y = []
        # np.zeros((self.batchsz[mode], querysz) + self.eeg_shape)
        query_x = []
        # np.zeros((self.batchsz[mode], querysz), dtype=int)
        query_y = []

        all_sessions = np.unique(self.dataset['session'])
        if batchsz> len(all_sessions):
            selected_tasks = np.random.choice(all_sessions, batchsz, replace=True)
        else:
            selected_tasks = np.random.choice(all_sessions, batchsz, replace=False)
        # from the dataset, get the index for the chosen session and the chosen class 
        # print('selected_tasks',selected_tasks)
        for i, cur_task in enumerate(selected_tasks):
            support_x.append([])
            support_y.append([])
            query_x.append([])
            query_y.append([])
            shuffle_idx_support = np.arange(self.n_way)
            np.random.shuffle(shuffle_idx_support)
            shuffle_idx_query = np.arange(self.n_way)
            np.random.shuffle(shuffle_idx_query)
            for j in range(self.n_way):
                task_index_list = np.where(self.dataset['session']==cur_task)[0]
                # print('task_index_list',task_index_list)
                classes_index_list = np.where(self.dataset['label']==j)[0]
                # print('classes_index_list',classes_index_list)
                task_classes_index_list = np.intersect1d(task_index_list,classes_index_list)
                # print('task_classes_index_list',task_classes_index_list)
                # print('task_classes_index_list',len(task_classes_index_list),self.k_shot + self.k_query)
                assert len(task_classes_index_list) >= self.k_shot + self.k_query
                selected_eeg_index = np.random.choice(task_classes_index_list, self.k_shot + self.k_query, False)
                # print('selected_eeg_index',selected_eeg_index)               
                #support set
                support_x_paths =self.dataset['eeg'][selected_eeg_index[:self.k_shot]]
                support_x[i].extend([self._load_eeg(eeg_path,normalize=self.normalize) for eeg_path in support_x_paths])
                support_y[i].extend([j]*self.k_shot)
                # query set
                query_x_paths =self.dataset['eeg'][selected_eeg_index[self.k_shot:]]
                query_x[i].extend([self._load_eeg(eeg_path,normalize=self.normalize) for eeg_path in query_x_paths])
                query_y[i].extend([j]*self.k_query)
        support_x = np.array(support_x)
        support_y = np.array(support_y)
        query_x = np.array(query_x)
        query_y = np.array(query_y)
        # print('support_x',support_x.shape,'support_y',support_y.shape,'query_x',query_x.shape,'query_y',query_y.shape)
        return support_x, support_y, query_x, query_y
class GWilliamsDataset(EEGDatasetBase):
    def __init__(
        self,
        phase,
        data_root,
        split_method='cross-session', 
        subjects=['S05','S07','S08'],
        val_subject='S05',
        val_sessions=[1], 
        test_subject='S07',
        test_sessions=[1],
        select_channels=None,
        select_vocabs =None,
        preproc_args = {},
        normalize = True,
        random_seed = 0,
        force_rebuild_index= False,
        force_rebuild_feature = False,
        debug = False,
    ): 
        super().__init__(
            phase,
            data_root,
            split_method, 
            subjects,
            val_subject,
            val_sessions, 
            test_subject,
            test_sessions,
            select_channels,
            select_vocabs,
            preproc_args,
            normalize,
            random_seed,
            force_rebuild_index,
            force_rebuild_feature,
            debug,
        )
        self.name = 'GWilliamsDataset'
        self._cached_dataset()    
    def _get_preproc_and_cache_names(self):
        # define the feature folder name and the cache path here
        return "debug","cache_path"
    def _get_name_cache(self):
        return os.path.join(self.data_root, 'cache')

    def _init_vocab(self):
        pass
    def _build(self):
        pass

class EEGRealtimeDataset(Dataset):
    def __init__(
            self,
            data_root=None, 
            data_identifier: dict = None, 
            preproc_args={},
            ##################
            # Parameters for spliting
            phase='train',
            split_method='calibration', # 'calibration' or 'random'
            split_ratio=0.8,
            return_dict = False,
            ##################
            # Parameters for data set variation
            select_channels=None,
            label_mapper=None,
            select_vocabs=[],
            select_vocabs_remap_index=False,
            vocab_groupping=None, # if not None, we will group the vocabularies
            anchor='start', # when segmenting the epoch, do we use the start [10x] or end [2x] of the event 
            decode_time =1.0,           
            **kwargs # for compatibility
        ): 
        ################ All input params
        self.data_root = data_root
        self.preproc_args = preproc_args
        self.data_identifier = data_identifier
        self.data_condition_id = data_identifier.get('condition_id', None) if data_identifier else None
        self.data_subject_id = data_identifier.get('subject_id', None) if data_identifier else None
        self.data_session_id = data_identifier.get('session_id', None) if data_identifier else None
        self.data_block_id = data_identifier.get('block_id', None) if data_identifier else None
        self.data_day_id = data_identifier.get('day_id', None) if data_identifier else None
        self.all_data_identifier = None  
        self.split_ratio = split_ratio
        self.phase = phase
        self.label_mapper = label_mapper
        self.vocab_groupping =vocab_groupping
        self.select_vocabs = [str(e) for e in select_vocabs] if not select_vocabs is None else []
        self.select_vocabs_remap_index = select_vocabs_remap_index
        self.n_jobs = self.preproc_args.get('n_jobs', 4) # TODO: remove this in the future versions. 
        ################ All output data and meta info
        self.meta_info = {'files': []}
        self.ica=None # Not implemented yet. 
        self.whitening_weights = None # Not implemented yet.
        self.all_eeg = []
        self.all_label = []
        self.all_mne_epochs = None # optional, if we want to access the mne epochs
        self.split_method = split_method
        self.count_total_epochs = 0        
        ################ get_item related variables         
        self.decode_time = decode_time
        self.select_channels = select_channels
        
        ################ Uncategorized internal variables
        self.labels = np.array([])
        self.anchor = anchor
        self.locks = {}  # Dictionary to store lock file descriptors

        ################ Initalizers to setup internal variables
        self._init_sampling_rate() 
        self._init_channel_names() 
        self._init_identifiers()
        self._init_vocab()
        self._init_pipeline_name() 
    def _init_vocab(self):
        """
            Use the relevant_events from the preproc_args to initialize the vocabularies.
            Output is marker2word, word2id, id2word, marker2id, id2marker
        """
        relevant_events = [str(e) for e in self.preproc_args['relevant_events']]
        self.marker_to_word = self.preproc_args.get('relevant_events_to_words', {})
        if len(self.marker_to_word) == 0:
            # assign the event id as the word
            for e in relevant_events:
                self.marker_to_word[e] = e
        ##############
        # processing select_vocabs and select_vocabs_remap_index
        # select_vocabs must be in the relevant_events, otherwise raise error
        if len(self.select_vocabs)>0 :
            for select_vocab in self.select_vocabs:
                if select_vocab not in self.relevant_events:
                    raise ValueError(f'select_vocabs must be in the relevant_events list, found {select_vocab}not in the list')         
        
        
        self.word2id = {}
        self.id2word = {} 
        self.marker2id = {}
        self.id2marker = {}
        counter=0
        for i, e in enumerate(relevant_events):
            if len(self.select_vocabs)>0:
                if e in self.select_vocabs:
                    if self.select_vocabs_remap_index: 
                        self.word2id[self.marker_to_word[e]] = counter
                        self.id2word[counter] = self.marker_to_word[e]
                        self.marker2id[e] = counter
                        self.id2marker[counter] = e
                        counter+=1
                    else:
                        self.word2id[self.marker_to_word[e]] = i
                        self.id2word[i] = self.marker_to_word[e]
                        self.marker2id[e] = i
                        self.id2marker[i] = e
                else:
                    pass
            else:
                self.word2id[self.marker_to_word[e]] = i
                self.id2word[i] = self.marker_to_word[e]
                self.marker2id[e] = i
                self.id2marker[i] = e
    def _init_identifiers(self):
        # if all None, then just return
        if self.data_subject_id is None and self.data_day_id is None and self.data_session_id is None and self.data_block_id is None and self.data_condition_id is None: 
            self.data_subject_id = ['-1']*len(self.data_root)
            self.data_day_id = ['-1']*len(self.data_root)
            self.data_session_id = ['-1']*len(self.data_root)
            self.data_block_id = ['-1']*len(self.data_root)
            self.data_condition_id = ['-1']*len(self.data_root)
        # check if the data_subject_id, data_day_id, data_session_id, data_block_id has the same length, if true, then we stack them together
        len_list = [len(self.data_subject_id), len(self.data_day_id), len(self.data_session_id), len(self.data_block_id), len(self.data_condition_id)]
        if len(set(len_list)) != 1:
            print('len_list',len_list)
            raise ValueError(' data_subject_id, data_day_id, data_session_id, data_block_id, data_condition_id must have the same length if they are not None')
        if list(set(len_list))[0] != len(self.data_root):
            raise ValueError(' data_subject_id, data_day_id, data_session_id, data_block_id, data_condition_id must have the same length as self.data_root')
        self.all_data_identifier = np.stack([
            self.data_condition_id,
            self.data_subject_id, 
            self.data_day_id, 
            self.data_session_id, 
            self.data_block_id], axis=0)
        # add one dimension to the end
        self.all_data_identifier = np.expand_dims(self.all_data_identifier, axis=-1)
        # 3 dimensions: (id_type, n_files, epoch)
    def _init_pipeline_name(self):
        hash_params = self.preproc_args.copy()
        # put the root into the hash params
        hash_params['data_root'] = self.data_root
        print('hash_params self.data_root', self.data_root)
        self.pipeline_name = self.preproc_args.get("preprocess_name","")    
        self.hash_identifier =hash_params_simple(hash_params)        
    def _identifier_multiplier(self, dim, repetition):
        print('expand self.all_data_identifier',self.all_data_identifier.shape,'dim',dim,'repetition',repetition)
        if dim==1:                
            # self.all_data_identifier = np.repeat(self.all_data_identifier, repetition, axis=1)      
            results =[]
            for i, rep in enumerate(repetition):
                repeated = np.repeat(self.all_data_identifier[:, i:i+1, :], rep, axis=1)
                # print('repeated',repeated.shape)
                results.append(repeated)
            self.all_data_identifier = np.concatenate(results, axis=1) 
            print('self.all_data_identifier',self.all_data_identifier.shape)
            # results =
            # all_data_identifier= results
        if dim==2:
            # results = []
            # for i, rep in enumerate(repetition):
            #     repeated = np.repeat(self.all_data_identifier[:, i:i+1, :], rep, axis=2)
            #     results.append(repeated)

            results = np.concatenate([
                np.repeat(self.all_data_identifier[:, i:i+1, :], rep, axis=2) 
                for i, rep in enumerate(repetition)
            ], axis=2)
            self.all_data_identifier = results
        # print('self.all_data_identifier',self.all_data_identifier.shape)
        # print('self.all_data_identifier',self.all_data_identifier)
    def _identifier_remove_epochs(self, remove_idx):
        """
            This is to remove 1 epoch from the all_data_identifier. 
            First we need to flatten the last two dimensions, then remove the epoch, then reshape back to the original shape.
        """
        print('remove epochs from self.all_data_identifier',self.all_data_identifier.shape,'remove_idx',remove_idx)
        new_identifier = []
        for i in range(self.all_data_identifier.shape[0]):# for each id type
            identifier_block_i = self.all_data_identifier[i] # [mne_file, epoch number ]
            new_identifier_block_i = [] 
            for j in range(identifier_block_i.shape[0]):# for each mnefile

                # print("remove_idx",type(remove_idx), remove_idx)
                remove_idx_in_file = [idx for idx in remove_idx if idx<identifier_block_i.shape[1]]
                within_file_identifier = identifier_block_i[j]                
                within_file_identifier_new = np.delete(within_file_identifier, remove_idx_in_file)
                # print('delete',remove_idx_in_file,'from file',j,'original len',len(within_file_identifier),'new len',len(within_file_identifier_new))
                new_identifier_block_i.append(within_file_identifier_new)
            new_identifier_block_i = np.array(new_identifier_block_i)
        
            new_identifier.append(new_identifier_block_i)
        new_identifier = np.array(new_identifier)
        # print('new_identifier',new_identifier.shape)
        # print('new_identifier',new_identifier)

        # stack back to the original shape
        self.all_data_identifier = new_identifier
        # print('self.all_data_identifier',self.all_data_identifier.shape)
    def _init_channel_names(self):
        self.final_channel_names = None 
        self.channel_update_flag = False  # flag to indicate if the channel names are updated from the raw raw_channel_names by default
        self.require_channel_reorder_after_loading_raw = False
        """
        - The raw_channel_names defined here is used for lsl or npz cases where you don't have the channel names in the data file. For all other cases, the raw_channel_names will be overridden when loading the data file.
        - channel_update_flag and final_channel_names are used to keep track of whether the channels have been updated after loadng the raw data file or n
        # the channel names is in upper case ot. for example, dropping channels, or selecting channels based on demands. 
        """
        channel_info_path = self.preproc_args.get('channel_info_path',None)
        if channel_info_path is  None:
            self.raw_channel_names = None
        else:
            channel_info = pd.read_csv(channel_info_path)
            self.raw_channel_names = channel_info['channel_name_upper_case'].values.tolist()
        self.final_channel_names = self.raw_channel_names

        manual_set_channel_order_path = self.preproc_args.get('manual_set_channel_order_path',None)
        if manual_set_channel_order_path is not None:
            channel_order_info = pd.read_csv(manual_set_channel_order_path)
            self.manual_set_channel_order = channel_order_info['channel_name_upper_case'].values.tolist()
            self.require_channel_reorder_after_loading_raw = True
        else:
            self.manual_set_channel_order = None
            self.require_channel_reorder_after_loading_raw = False
    def _init_sampling_rate(self):
        """
            - self.raw_sampling_rate defined here is used to define a sampling rate for 'lsl' or 'npz' cases where you don't have the sampling rate in the data file. For all other cases, the raw_sampling_rate will be overridden when loading the data file.
            - self.final_sampling_rate is the final sampling rate after any resampling.
            - self.sampling_rate_update_flag is used to keep track of whether the sampling rate has been updated after loadng the raw data file or not.        
        """ 
        self.raw_sampling_rate = self.preproc_args.preproc_raw.get("original_fs", 1000)
        self.final_sampling_rate = self.preproc_args.preproc_raw.get("resample_fs",self.raw_sampling_rate)
        if self.final_sampling_rate != self.raw_sampling_rate:
            self.sampling_rate_update_flag = True
        else:
            self.sampling_rate_update_flag = False
    def _get_epoched_data_from_npz(self, data_dict, single_trial=False):
        # in this step, we map the event to the label id. 
        event_map = {}
        event_map_reverse = {}
        relevant_events = [str(e) for e in self.preproc_args['relevant_events']]
        for i, e in enumerate(relevant_events):
            event_map[e] = i+1
            event_map_reverse[i+1] = e 
        eeg_labels = []
        eeg_epochs = []
        eeg_segment = data_dict['eeg']  # shape: (n_samples, n_channels)
        eeg_time = data_dict['eeg_time']
        event = data_dict['event']
        event_time = data_dict['event_time']
        # make my own data and label here
        # iterate through the event and get the data
        last_word_event_idx = None
        events_needed = [int(i) for i in self.label_mapper.keys()]
        for i in range(len(event)):
            if single_trial:
                last_word_event_idx = 0
            else:
                if int(event[i]) in events_needed:
                    last_word_event_idx = i
            if int(event[i]) in [2,21,22,23] :
                if last_word_event_idx is not None:
                    end_speech_event_time = event_time[i]
                    # find the eeg_time that is closest to the end_speech_event_time                        
                    end_speech_eeg_time= np.argmin(np.abs(eeg_time-end_speech_event_time))
                    start_speech_eeg_time = np.argmin(np.abs(eeg_time-event_time[last_word_event_idx]))
                    assert start_speech_eeg_time < end_speech_eeg_time, 'start_speech_eeg_time should be less than end_speech_eeg_time' 
                    if self.anchor == 'start':
                        eeg_trial = eeg_segment[
                            start_speech_eeg_time+int(self.tmin*self.original_sampling_rate):start_speech_eeg_time+int(self.tmax*self.original_sampling_rate),
                            :
                        ]
                    else:
                        print('end_speech_eeg_time',end_speech_eeg_time)
                        raise ValueError('anchor should be start only because the curry data is epoched using start of the event')
                        eeg_trial = eeg_segment[
                            end_speech_eeg_time-self.original_sampling_rate+int(self.tmin*self.original_sampling_rate):end_speech_eeg_time-self.original_sampling_rate+int(self.tmax*self.original_sampling_rate),
                            :
                        ]
                    eeg_epochs.append(eeg_trial)
                    eeg_label = event_map[event[last_word_event_idx]]
                    eeg_labels.append(eeg_label)
                    last_word_event_idx = None
                self.count_total_epochs += 1
        return eeg_epochs, eeg_labels
    def _get_mne_epoch_from_npz(self, data_dict, single_trial=False):# rework the logic based on the consistency check of the data. 
        # in this step, we map the event to the label id. 
        event_map = {}
        event_map_reverse = {}
        relevant_events = [str(e) for e in self.preproc_args['relevant_events']]
        for i, e in enumerate(relevant_events):
            event_map[e] = i+1
            event_map_reverse[i+1] = e 
        eeg_labels = []
        eeg_epochs = []

        eeg_segment = data_dict['eeg']  # shape: (n_samples, n_channels)
        eeg_time = data_dict['eeg_time']
        event = data_dict['event']
        event_time = data_dict['event_time']
        # only keep events that are in the relevant events
        marker_list = [] 
        marker_indice_list= []
        marker_time_list = []
        # print("self.preproc_args['relevant_events']",self.preproc_args['relevant_events'])
        for i, e in enumerate(event):
            if e in self.preproc_args['relevant_events']:
                marker = event[i]
                marker_index = i
                marker_time = event_time[i]
                marker_list.append(marker)
                marker_indice_list.append(marker_index)
                marker_time_list.append(marker_time)
                eeg_label = event_map[marker]
                eeg_labels.append(eeg_label)
                if marker_time in eeg_time:
                    marker_eeg_index = np.where(eeg_time == marker_time)[0][0]
                else:# Find the closest time in eeg_time to marker_time                   
                    closest_index = (np.abs(eeg_time - marker_time)).argmin()
                    closest_time = eeg_time[closest_index]
                    # Get the sample index
                    marker_eeg_index = closest_index
                    # print(f"Sample index for marker {marker_eeg_index}")
                start_index = int(marker_eeg_index + self.tmin * self.original_sampling_rate)
                end_index   = int(marker_eeg_index + self.tmax * self.original_sampling_rate)+1
                eeg_trial = eeg_segment[start_index:end_index]
                eeg_epochs.append(eeg_trial)
        return eeg_epochs, eeg_labels 
    def _get_mne_epoch_from_xdf(self, file_path, EEG_stream_name=None,EEG_stream_type=None,Marker_stream_name=None, relevant_events=None):
        # print('read_xdf_as_mne',file_path)
        # Load the XDF file using pyxdf
        streams, header = pyxdf.load_xdf(file_path)
        # check EEG_stream_name and EEG_stream_type cannot be both None
        if EEG_stream_name is None and EEG_stream_type is None:
            raise ValueError("EEG_stream_name and EEG_stream_type cannot be both None") 
        # Find the EEG stream (assuming you have multiple streams in the XDF file)
        eeg_stream = None
        events_stream = None

        for stream in streams:
            # print('stream',stream['info'])
            """
                {'name': ['ch16'], 'unit': ['uV'], 'type': ['ExG']})]}), defaultdict(<class 'list'>, {'channel': [defaultdict(<class 'list'>, 
                {'name': ['ax'], 'unit': ['mg'], 'type': ['ORN']}), defaultdict(<class 'list'>, 
                {'name': ['ay'], 'unit': ['mg'], 'type': ['ORN']}), defaultdict(<class 'list'>, 
                {'name': ['az'], 'unit': ['mg'], 'type': ['ORN']}), defaultdict(<class 'list'>, 
                {'name': ['gx'], 'unit': ['mdps'], 'type': ['ORN']}), defaultdict(<class 'list'>, 
                {'name': ['gy'], 'unit': ['mdps'], 'type': ['ORN']}), defaultdict(<class 'list'>,
                {'name': ['gz'], 'unit': ['mdps'], 'type': ['ORN']}), defaultdict(<class 'list'>, 
                {'name': ['mx'], 'unit': ['mgauss'], 'type': ['ORN']}), defaultdict(<class 'list'>, 
                {'name': ['my'], 'unit': ['mgauss'], 'type': ['ORN']}), defaultdict(<class 'list'>, 
                {'name': ['mz'], 'unit': ['mgauss'], 'type': ['ORN']})]})]})], 'stream_id': 2, 'effective_srate': 249.90611716051097})
            """
            if EEG_stream_type is not None and stream['info']['type'][0] == EEG_stream_type:
                eeg_stream = stream
            if EEG_stream_name is not None and stream['info']['name'][0] == EEG_stream_name:
                eeg_stream = stream
            if Marker_stream_name is not None and stream['info']['name'][0] == Marker_stream_name:
                events_stream = stream

        if eeg_stream is None:
            raise ValueError("No EEG stream found in the XDF file.")
        # Get EEG data and timestamps
        eeg_data = np.array(eeg_stream['time_series'])#.T  # Transpose to match MNE format
        eeg_time = np.array(eeg_stream['time_stamps'])
        event_data = np.array(events_stream['time_series'])
        # flatten
        event_data = event_data.flatten()   
        event_time = np.array(events_stream['time_stamps'])
        # print('eeg_data',eeg_data.shape,eeg_time.shape,'event_data',event_data.shape,event_time.shape)
        # print('relevant_events',relevant_events)
        relevant_events = [int(e) for e in relevant_events]
        eeg_labels = []
        eeg_epochs = []
        # print(file_path,'unique event',np.unique(event_data))
        for i in range(len(event_data)):
            # print('event_data[i]',event_data[i])
            if int(event_data[i]) in relevant_events:
                start_speech_event_time = event_time[i]
                start_speech_eeg_index= np.argmin(np.abs(eeg_time-start_speech_event_time))
                
                start_index = int(start_speech_eeg_index + self.tmin * self.original_sampling_rate)
                end_index   = int(start_speech_eeg_index + self.tmax * self.original_sampling_rate)+1
                # int(start_speech_event_time)
                eeg_trial = eeg_data[start_index:end_index]
                eeg_epochs.append(eeg_trial.T)
                eeg_labels.append(event_data[i])
        eeg_epochs = np.array(eeg_epochs)
        eeg_labels = np.array(eeg_labels)-min(eeg_labels)
        # print('eeg_epochs',eeg_epochs.shape,'eeg_labels',eeg_labels.shape,np.unique(eeg_labels,return_counts=True))
        # exit(0)
        return eeg_epochs,eeg_labels
    def _get_mne_epoch_from_cdt(self, cft_file_path, events_of_interest, enable_cache, cache_root, force_rebuild):
        cdt_level_hash_parameters = self.preproc_args.copy()
        cdt_level_hash_parameters['file_path'] = cft_file_path
        cdt_level_hash = hash_params_simple(cdt_level_hash_parameters)
        cache_file = "{}/{}/{}.pkl".format(cache_root,self.pipeline_name,cdt_level_hash) # just dump in the same folder. 
        cache_dir = os.path.dirname(cache_file)
        lock_file_path = Path(cache_dir) / f".{Path(cache_file).name}.lock"
        # make folder if not exist
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        if enable_cache:
            if cache_root is not None and not force_rebuild:
                # Fast path: Check if cache exists (no lock needed)        
                if os.path.exists(cache_file) and not force_rebuild:  
                    print('Loading cached mne epoch from', cache_file)
                    with open(cache_file, 'rb') as f:
                        epochs, relevant_event_id = pickle.load(f)                        
                    return epochs, relevant_event_id
            # Slow path: Cache doesn't exist, acquire lock to build it
            with self.file_lock_nfs(lock_file_path, lock_name=cdt_level_hash):
                # After acquiring lock, check again if cache was built by another process
                if os.path.exists(cache_file) and not force_rebuild:
                    print('Loading cached mne epoch from', cache_file)
                    with open(cache_file, 'rb') as f:
                        epochs, relevant_event_id = pickle.load(f)                        
                    return epochs, relevant_event_id 
                print('Processing raw data to get mne epoch from cdt file:', cft_file_path)
                epochs, relevant_event_id = self._preprocess_raw_and_extract_epoches(cft_file_path, events_of_interest)
                print('Saving raw data epoch to cache', cache_file)
                with open(cache_file, 'wb') as f:
                    pickle.dump((epochs, relevant_event_id), f)
                os.chmod(cache_file, 0o777)
        else:
            print('Processing raw data to get mne epoch from cdt file without caching:', cft_file_path)
            epochs, relevant_event_id = self._preprocess_raw_and_extract_epoches(cft_file_path, events_of_interest) 
        return epochs, relevant_event_id
    def _get_mne_epoch_from_cdt_non_blocking(self, cft_file_list, events_of_interest, enable_cache, cache_root, force_rebuild):
        pending = list(cft_file_list)   # files still needing processing
        results = {}                # f -> (epochs, relevant_event_id)
        eeg_epochs = []
        eeg_epoch_event_id_list = []
        epoch_repetition_list =[]
        while pending:
            still_pending = []
            progress_made = False
            for f in pending:
                cdt_level_hash_parameters = self.preproc_args.copy()
                cdt_level_hash_parameters['file_path'] = f
                cdt_level_hash = hash_params_simple(cdt_level_hash_parameters)
                cache_file = "{}/{}/{}.pkl".format(cache_root, self.pipeline_name, cdt_level_hash)
                lock_file_path = Path(os.path.dirname(cache_file)) / f".{Path(cache_file).name}.lock"

                # Fast path: already cached, no lock needed
                if enable_cache and cache_root and not force_rebuild and os.path.exists(cache_file):
                    epochs, relevant_event_id = self._get_mne_epoch_from_cdt(
                        cft_file_path=f, events_of_interest=events_of_interest,
                        enable_cache=enable_cache, cache_root=cache_root, force_rebuild=force_rebuild
                    )
                    results[f] = (epochs, relevant_event_id)
                    progress_made = True
                    continue
                # Try to acquire lock non-blocking
                with self.file_lock_nfs_non_blocking(lock_file_path, lock_name=cdt_level_hash) as acquired:
                    if not acquired:
                        print(f"[Cache] Lock busy for {f}, skipping for now, waiting lock {lock_file_path}")
                        still_pending.append(f)
                        continue
                    # We hold the lock — delegate to existing method
                    # but it will try to lock again internally, so inline the logic here:
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    if enable_cache and os.path.exists(cache_file) and not force_rebuild:
                        with open(cache_file, 'rb') as fh:
                            epochs, relevant_event_id = pickle.load(fh)
                    else:
                        print('Processing:', f)
                        epochs, relevant_event_id = self._preprocess_raw_and_extract_epoches(f, events_of_interest)
                        if enable_cache:
                            with open(cache_file, 'wb') as fh:
                                pickle.dump((epochs, relevant_event_id), fh)
                            os.chmod(cache_file, 0o777)

                    results[f] = (epochs, relevant_event_id)
                    progress_made = True
            pending = still_pending
            if pending and not progress_made:
                # All remaining files are locked by other processes — just wait
                print(f"[Cache] {len(pending)} file(s) still locked, waiting...")
                time.sleep(1)
        # Preserve original file order
        for f in cft_file_list:
            epochs, relevant_event_id = results[f]
            epoch_repetition_list.append(len(epochs.events[:, -1]))
            eeg_epochs.append(epochs)
            eeg_epoch_event_id_list.append(relevant_event_id)       
        return eeg_epochs, eeg_epoch_event_id_list, epoch_repetition_list

    def _preprocess_raw_and_extract_epoches(self, file_path, events_of_interest):
        relevant_event_id = {}  # {'102': 3, '103': 4,...} # key: the actual marker, value: mne internal number for the marker
        raw =  mne.io.read_raw_curry(file_path, preload=True,verbose=False) 
        raw = _make_channel_names_uppercase(raw)
        if self.require_channel_reorder_after_loading_raw:
            print('################ Reordering channels based on manual_set_channel_order')
            raw = raw.reorder_channels(self.manual_set_channel_order)
        # compare sample1 and sample2 to make sure the channel reorder is correct by checking the first 30 samples of the first channel
        self.raw_sampling_rate = raw.info['sfreq']
        self.raw_channel_names = raw.ch_names 
        self.channel_update_flag = False
        self.sampling_rate_update_flag = False
        self.final_channel_names = self.raw_channel_names
        self.final_sampling_rate = self.raw_sampling_rate

        ##################
        # Main Raw Preprocessing Steps
        if self.preproc_args.get("preproc_raw",None) is not None:
            raw, ica = _preproc_core_raw(raw, self.preproc_args['preproc_raw'])                    
            self.ica = ica        
        
        events, event_id = mne.events_from_annotations(raw,verbose=False) # already get all the events from the raw data
        reverse_mapping = {v: k for k, v in event_id.items()}
        existing_events_in_datast = list(event_id.keys())
        relevant_events = []
        for j,e in enumerate(events):# only keep the relevant events in the events
            if str(reverse_mapping[e[2]]) not in events_of_interest:
                pass
            else:
                relevant_events.append(e)
        relevant_events = np.array(relevant_events)                
        # keep those in the relevant events and in the event of interest
        for k,v in event_id.items():
            if str(k) in events_of_interest:
                relevant_event_id[str(k)] = v
        epochs = mne.Epochs(raw, relevant_events, relevant_event_id, tmin=self.preproc_args.preproc_epochs['tmin'], tmax=self.preproc_args.preproc_epochs['tmax'], baseline=None, preload=True,verbose=False)
        return epochs, relevant_event_id
    def _preprocess_epochs(
        self, eeg_data=None,
        eeg_labels=None, 
        eeg_info=None, 
        mne_epochs=None,
        preprocess_params=None,
        return_mne_epoch=False,
    ):  
        if mne_epochs is None:
            # Create a new MNE Epochs object using the preprocessed data and labels
            sfreq=self.final_sampling_rate  if self.sampling_rate_update_flag else self.raw_sampling_rate
            ch_names = self.final_channel_names if self.channel_update_flag else self.raw_channel_names
            tmin=preprocess_params.preproc_epochs.get('tmin',0)
            tmax=preprocess_params.preproc_epochs.get('tmax',1)
            if eeg_info is None:
                eeg_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg',verbose=False) 
            if eeg_labels is None:
                events = np.array([[i, 0, 0] for i in range(len(eeg_data))])
            else:
                events = np.array([[i, 0, eeg_labels[i]] for i in range(len(eeg_labels))])
            epochs = mne.EpochsArray(eeg_data, eeg_info, events, tmin=tmin, verbose=False)   
        else:
            epochs = mne_epochs
        
        
        if preprocess_params.get('default_drop_channels',None) is not None and len(preprocess_params.get('default_drop_channels',None))>0:
            print('Dropping channels:',self.preproc_args['default_drop_channels'])
            epochs = self._drop_channels(epochs,self.preproc_args['default_drop_channels'])         


        ###########################
        # Main Epoch Preprocessing Steps
        epochs, bad_epoch_ids = _preproc_core_epoch(
            epochs, 
            preprocess_params.preproc_epochs
        )


        if bad_epoch_ids is not None:
            self._identifier_remove_epochs(bad_epoch_ids) 
        self.final_sampling_rate = epochs.info['sfreq']
        self.sampling_rate_update_flag = True
        self.final_channel_names = epochs.ch_names
        self.channel_update_flag = True
        return epochs, bad_epoch_ids
    def _process_trials_old(self, data_dict, anchor='start'):
        event_map = {}
        event_map_reverse = {}
        relevant_events = [str(e) for e in self.preproc_args['relevant_events']]
        for i, e in enumerate(relevant_events):
            event_map[e] = i+1
            event_map_reverse[i+1] = e 
        possible_events = [52,53,56,57, 58,59, 60, 61, 98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122 ]  
        # convert to str
        possible_events = [str(e) for e in possible_events]
        # if not in the event_map, add to the event_map
        for i, e in enumerate(possible_events):
            if e not in event_map.keys():
                event_map[e] = len(event_map)+i+1
                event_map_reverse[len(event_map)+i+1] = e 
        eeg_labels = []
        eeg_epochs = []
        eeg_segment = data_dict['eeg']  # shape: (n_samples, n_channels)
        eeg_time = data_dict['eeg_time']
        event = data_dict['event']
        event_time = data_dict['event_time']
        
        # let's say there are many events in the data, but we only care about the last word event. 
        if anchor =='start':
            # print('all event',event)
            # take segments based on start of speech events
            for i, e in enumerate(event):
                if e in possible_events:
                    # print('start of speech event',e)
                    marker = event[i]
                    marker_index = i
                    marker_time = event_time[i]
                    eeg_label = event_map[marker]
                    eeg_labels.append(eeg_label)
                    if marker_time in eeg_time:
                        marker_eeg_index = np.where(eeg_time == marker_time)[0][0]
                    else:
                        # Find the closest time in eeg_time to marker_time                   
                        closest_index = (np.abs(eeg_time - marker_time)).argmin()
                        closest_time = eeg_time[closest_index]
                        # Get the sample index
                        marker_eeg_index = closest_index
                    print("raw_sampling_rate",self.raw_sampling_rate)
                    start_index = int(marker_eeg_index + self.tmin * self.raw_sampling_rate)
                    end_index   = int(marker_eeg_index + self.tmax * self.raw_sampling_rate)+1
                    eeg_trial = eeg_segment[start_index:end_index]
                    eeg_epochs.append(eeg_trial) 
        elif anchor == 'end':
            last_start_of_speech_event = None
            last_start_of_speech_event_index = None            
            for i, e in enumerate(event):
                if e in self.preproc_args['relevant_events']:
                    last_start_of_speech_event = event[i]
                    last_start_of_speech_event_index = i                
                if int(e) in [2,21,22,23,24,25]: # end of speech.
                    if last_word_event_idx is not None:
                        marker =e
                        marker_index = i
                        marker_time = event_time[i]
                        eeg_label = event_map[marker]
                        eeg_labels.append(eeg_label)
                        if marker_time in eeg_time:
                            marker_eeg_index = np.where(eeg_time == marker_time)[0][0]
                        else:
                            # Find the closest time in eeg_time to marker_time                   
                            closest_index = (np.abs(eeg_time - marker_time)).argmin()
                            closest_time = eeg_time[closest_index]
                            # Get the sample index
                            marker_eeg_index = closest_index
                        start_index = int(marker_eeg_index + (-1+self.tmin) * self.raw_sampling_rate)
                        end_index   = int(marker_eeg_index + (-1+self.tmax) * self.raw_sampling_rate)+1
                        eeg_trial = eeg_segment[start_index:end_index]
                        eeg_epochs.append(eeg_trial)
        else:
            raise ValueError('anchor should be start or end')
        if len(eeg_epochs) > 1:
            eeg_epochs = eeg_epochs[-1]
            eeg_labels = eeg_labels[-1]
            eeg_epochs = np.array([eeg_epochs])
            eeg_labels = [eeg_labels]
        else:
            eeg_epochs = np.array(eeg_epochs)
        eeg_epochs = eeg_epochs.transpose(0, 2, 1)       
        print('raw eeg_epochs',eeg_epochs.shape,'eeg_labels',len(eeg_labels))
        # need to convert the numpy as mne epoch. So we have channel_info_path = self.preproc_args.get('channel_info_path',None) to get the channel names, and we have the sampling rate to create the mne epoch, then we can apply the same preprocessing steps as in the _preprocess_data function.
        info = mne.create_info(ch_names=self.final_channel_names, sfreq=self.raw_sampling_rate, ch_types='eeg',verbose=False)
        mne_epochs = mne.EpochsArray(eeg_epochs, info, tmin=self.tmin, verbose=False)
        print('mne_epochs',mne_epochs.get_data().shape)
        self.all_mne_epochs,_ = self._preprocess_epochs(            
            eeg_data=None,
            eeg_labels=None, 
            eeg_info=None, 
            mne_epochs=mne_epochs,
            preprocess_params=self.preproc_args,              
        )
        eeg_epochs= self.all_mne_epochs.get_data()
        self.all_eeg = eeg_epochs
        self.all_label = eeg_labels

        print('self.all_eeg',self.all_eeg.shape,'self.all_label',len(self.all_label))
        return self.__getitem__(0)
        # return eeg_epochs, eeg_labels
    
    def _process_trials(self, data_dict, anchor='start'):
        event_map = {}
        event_map_reverse = {}
        relevant_events = [str(e) for e in self.preproc_args['relevant_events']]
        for i, e in enumerate(relevant_events):
            event_map[e] = i+1
            event_map_reverse[i+1] = e 
        possible_events = [52,53,56,57,58,59,60,61,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122]
        possible_events = [str(e) for e in possible_events]
        for i, e in enumerate(possible_events):
            if e not in event_map.keys():
                event_map[e] = len(event_map)+i+1
                event_map_reverse[len(event_map)+i+1] = e 

        eeg_labels = []
        eeg_segment = data_dict['eeg']        # shape: (n_samples, n_channels)
        eeg_time    = data_dict['eeg_time']
        event       = data_dict['event']
        event_time  = data_dict['event_time']

        # ------------------------------------------------------------------ #
        # 1. Build an MNE Raw object from the continuous EEG data             #
        # ------------------------------------------------------------------ #
        info = mne.create_info(
            ch_names=self.raw_channel_names,
            sfreq=self.raw_sampling_rate,
            ch_types='eeg',
            verbose=False
        )
        # print('info', info)
        # MNE Raw expects shape (n_channels, n_samples)
        raw = mne.io.RawArray(eeg_segment.T, info, verbose=False)
        raw, _ = _preproc_core_raw(raw, self.preproc_args['preproc_raw'], force_skip_channel_cleaning=True) # Channel Clearning is too harsh for the continuous data, we will do the channel cleaning after epoching.
        # raw.times starts at 0; eeg_time may have an arbitrary offset
        eeg_time_offset = eeg_time[0]
        raw_times = raw.times  # time axis of the resampled raw



        print(f'raw, raw.get_data().shape, raw.info["sfreq"]', raw.get_data().shape, raw.info['sfreq'])
        # ------------------------------------------------------------------ #
        # 2. Build the MNE events array  [sample_index, 0, event_id]          #
        # ------------------------------------------------------------------ #
        mne_events = []   # will become (n_events, 3)
        if anchor == 'start':
            # find the last matching event in possible_events
            last_event_idx = None
            for i, e in enumerate(event):
                if e in possible_events:
                    last_event_idx = i

            if last_event_idx is not None:
                e           = event[last_event_idx]
                marker_time = event_time[last_event_idx]
                eeg_label   = event_map[e]
                eeg_labels.append(eeg_label)

                # map marker_time into raw.times coordinate (raw starts at 0)
                sample_idx = int((np.abs(raw_times - (marker_time - eeg_time_offset))).argmin())
                mne_events.append([sample_idx, 0, eeg_label])

        elif anchor == 'end':
            last_start_of_speech_event_index = None
            for i, e in enumerate(event):
                if e in self.preproc_args['relevant_events']:
                    last_start_of_speech_event_index = i

                if int(e) in [2, 21, 22, 23, 24, 25]:   # end-of-speech markers
                    if last_start_of_speech_event_index is not None:
                        marker_time = event_time[i]
                        eeg_label   = event_map[e]
                        eeg_labels.append(eeg_label)

                        sample_idx = int((np.abs(raw_times - (marker_time - eeg_time_offset))).argmin())
                        mne_events.append([sample_idx, 0, eeg_label])
        else:
            raise ValueError("anchor should be 'start' or 'end'")

        mne_events = np.array(mne_events, dtype=int)  # shape (n_events, 3)
        print('mne_events:', mne_events)
        

        # ------------------------------------------------------------------ #
        # 3. Epoch the Raw object using the events array                      #
        # ------------------------------------------------------------------ #
        tmin = self.tmin if anchor == 'start' else (-self.decode_time + self.tmin)
        tmax = self.tmax if anchor == 'start' else (-self.decode_time + self.tmax)

        event_id = {str(label): label for label in np.unique(mne_events[:, 2])}
        print('event_id mapping:', event_id)

        mne_epochs = mne.Epochs(
            raw,
            mne_events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False
        )

        # keep only the last epoch
        if len(mne_epochs) > 1:
            mne_epochs = mne_epochs[-1:]
            eeg_labels = [eeg_labels[-1]]

        print('raw eeg_epochs', mne_epochs.get_data().shape, 'eeg_labels', len(eeg_labels))

        # ------------------------------------------------------------------ #
        # 4. Apply the same preprocessing pipeline as the original code       #
        # ------------------------------------------------------------------ #
        self.all_mne_epochs, _ = self._preprocess_epochs(
            eeg_data=None, eeg_labels=None, eeg_info=None,mne_epochs=mne_epochs,
            preprocess_params=self.preproc_args,
        )

        self.all_eeg   = self.all_mne_epochs.get_data()
        self.all_label = eeg_labels

        print('self.all_eeg', self.all_eeg.shape, 'self.all_label', len(self.all_label))
        return self.__getitem__(0)

    def _drop_channels(self, epochs, channels_to_drop=None):
        if channels_to_drop is not None:
            # convert channels_to_drop to uppercase
            channels_to_drop = [ch.upper() for ch in channels_to_drop if ch.upper() in epochs.ch_names]
            if len(channels_to_drop)>0:
                # drop the channels
                epochs.drop_channels(channels_to_drop)
            
            # for each channels_to_drop, remove from self.channel_names        
            for ch in channels_to_drop:
                if ch in self.final_channel_names:
                    self.final_channel_names.remove(ch)
            self.channel_update_flag= True
        return epochs
    def _standardize(self, data):
        # Standardization: Zero mean and unit variance
        mean_value = data.mean()
        std_value = data.std()
        standardized_data = (data - mean_value) / std_value
        return standardized_data  
    
    #################################
    ## Locks
    @contextmanager
    def file_lock_unix(self, lock_file_path, lock_name="default"):
        """Context manager for file locking with named locks"""
        lock_file_path = str(lock_file_path)
        lock_fd = os.open(lock_file_path, os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self.locks[lock_name] = lock_fd
            print(f"[Cache] Lock acquired: {lock_name}")
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            self.locks.pop(lock_name, None)
            print(f"[Cache] Lock released: {lock_name}")
    @contextmanager
    def file_lock_nfs(self, lock_file_path, lock_name="default"):
        """Context manager for file locking with named locks — NFS compatible"""
        lock_file_path = str(lock_file_path)
        lock_dir = lock_file_path + ".lock"        
        while True:
            try:
                os.makedirs(lock_dir, exist_ok=False)  # atomic on NFS
                break
            except FileExistsError:
                time.sleep(0.5)
        self.locks[lock_name] = lock_dir
        print(f"[Cache] Lock acquired: {lock_name}")
        try:
            yield
        finally:
            os.rmdir(lock_dir)
            self.locks.pop(lock_name, None)
            print(f"[Cache] Lock released: {lock_name}")
    @contextmanager
    def file_lock_nfs_non_blocking(self, lock_file_path, lock_name="default"):
        """Non-blocking lock attempt. Returns (context_manager, acquired)."""
        lock_file_path = str(lock_file_path)
        lock_dir = lock_file_path + ".lock"
        try:
            os.makedirs(lock_dir, exist_ok=False)  # atomic on NFS
            acquired = True
        except FileExistsError:
            acquired = False
        if acquired:
            self.locks[lock_name] = lock_dir
            print(f"[Cache] Lock acquired: {lock_name}")
        try:
            yield acquired
        finally:
            if acquired:
                os.rmdir(lock_dir)
                self.locks.pop(lock_name, None)
                print(f"[Cache] Lock released: {lock_name}")
    
    def _build(self,cache_file=None,  enable_cache=True, cache_root=None, force_rebuild=False, no_duplicate=True):
        ###########################################
        # Start to build from available data files
        ###########################################
        # check if root is None
        if self.data_root is None:
            print('data_root is None, no data to scan...')
            return 
        # Scan the path first , this is shared among all formats
        if isinstance(self.data_root, list):# check if self.data_root is a list 
            new_files = self.data_root            
        elif isinstance(self.data_root, str):# or if it is a string
            new_files = [self.data_root]
        elif isinstance(self.data_root, omegaconf.listconfig.ListConfig):
            new_files = list(self.data_root)
        else:
            print(type(self.data_root))
            raise ValueError('data_root should be a list or a string', 'but got:', type(self.data_root))        
        new_files_ = []
        file_repetition_list = []
        for i, f in enumerate(new_files):
            if '*' in f or '?' in f:
                flist = glob(f) 
            else:
                flist = [f]            
            if no_duplicate:
                flist = [fi for fi in flist if fi not in self.meta_info['files']]
            if len(flist) > 0:
                flist.sort()
                file_repetition_list.append(len(flist))                 
                new_files_.extend(flist)
        self._identifier_multiplier(1, file_repetition_list) # expand on the number of files axis
       
        new_files = new_files_  
        if len(new_files) == 0:
            print('[EEGRealtimeDataset] get no new files')
            return
        else:
            print(f'[EEGRealtimeDataset] get {len(new_files)} new files:')
            self.meta_info['files'].extend(new_files)        
        
        ###########################################
        # Start to read data based on different formats
        ###########################################
        eeg_epochs = []
        eeg_epoch_event_id_list = [] # thgis is the event id (only kept relevant events) for each epoch
        epoch_repetition_list = []
        events_of_interest = [str(e) for e in self.preproc_args['relevant_events']] # this is global

        if self.preproc_args['format']=='npz':
            raise NotImplementedError('npz format is not implemented yet, please use npy format instead')
            # do it in parallel
            npy_data_dicts =Parallel(n_jobs=self.n_jobs)(delayed(safe_np_load)(f) for f in new_files) 
            for k,data_dict in enumerate(npy_data_dicts):
                eeg_epochs_f, eeg_labels_f = self._get_mne_epoch_from_npz(data_dict,single_trial=False)
                eeg_epochs.extend(eeg_epochs_f)
                eeg_labels.extend(eeg_labels_f)
                epoch_repetition_list.append(len(eeg_labels_f))       
            eeg_epochs = np.array(eeg_epochs) # 133 Channels. # this is [n, time, n_channels]
            eeg_epochs = eeg_epochs.transpose(0, 2, 1) 
            eeg_labels = np.array(eeg_labels)-1
            if self.preproc_args.get('scaling_factor',None) is not None:
                print('scaling factor',self.preproc_args['scaling_factor'],type(self.preproc_args['scaling_factor']),'type of eeg_epochs',type(eeg_epochs),'dtype of eeg_epochs',eeg_epochs.dtype)
                eeg_epochs = eeg_epochs * float(self.preproc_args['scaling_factor'])            
        elif self.preproc_args['format']=='set':  
            raise NotImplementedError('set format is not implemented yet, please use cdt format instead')           
            event_map = {}
            event_map_reverse = {}
            relevant_events = [str(e) for e in self.preproc_args['relevant_events']]
            for i, e in enumerate(relevant_events):
                event_map[e] = i+1
                event_map_reverse[i+1] = e
            original_EEG_sampling_rate = None
            for f in new_files:
                print('processing:', f)
                full_path = os.path.join(self.data_root, f)
                raw = mne.io.read_raw_eeglab(full_path, preload=True,verbose=False)  
                self.original_sampling_rate = raw.info['sfreq']
                raw = _make_channel_names_uppercase(raw)
                if self.preproc_args.get("preproc_raw",None) is not None:
                    raw, ica = _preproc_core_raw(raw, self.preproc_args['preproc_raw'])
                    self.ica = ica
                self.channel_names_full = raw.ch_names
                events, event_id = mne.events_from_annotations(raw,verbose=False)
                reverse_mapping = {v: k for k, v in event_id.items()}
                reformed_events = []
                for j,e in enumerate(events):
                    if str(reverse_mapping[e[2]]) not in relevant_events:
                        pass
                    else:
                        reformed_events.append(e)
                for j,e in enumerate(reformed_events):
                    e_label = reverse_mapping[e[2]]
                    e[2] = event_map[e_label]
                reformed_events = np.array(reformed_events)
                filtered_event_id = {key: event_map[key] for key in relevant_events if key in event_map}
                epochs = mne.Epochs(raw, reformed_events, filtered_event_id, tmin=self.preproc_args['tmin'], tmax=self.preproc_args['tmax'], baseline=None, preload=True,verbose=False)
                if original_EEG_sampling_rate is None:
                    original_EEG_sampling_rate = epochs.info['sfreq']
                # print('epochs',epochs.get_data().shape)
                eeg_epochs.extend(epochs.get_data())
                eeg_labels.extend(epochs.events[:,-1])   
                epoch_repetition_list.append(len(epochs.events[:,-1]))         
            eeg_epochs = np.array(eeg_epochs)
            eeg_epochs = eeg_epochs[:,:,:int((self.preproc_args['tmax']-self.preproc_args['tmin'])*original_EEG_sampling_rate)]
            # print('eeg_epochs',eeg_epochs.shape)
            eeg_labels = np.array(eeg_labels)-np.min(eeg_labels) # make the label start from 0
            # print('eeg_epochs transpose',eeg_epochs.shape)
            # print('eeg_labels',eeg_labels.shape)  
        elif self.preproc_args['format']=='xdf':
            raise NotImplementedError('xdf format is not implemented yet, please use npy format instead')            
            eeg_epoch_list = []
            eeg_label_list = []
            # Parallel(n_jobs=self.n_jobs)(delayed(safe_np_load)(f) for f in new_files) 
            for f in new_files:
                eeg_epoch, eeg_label = self._get_mne_epoch_from_xdf(
                    f,
                    EEG_stream_name=self.preproc_args['eeg_stream_name'],
                    EEG_stream_type= self.preproc_args['eeg_stream_type'],
                    Marker_stream_name= self.preproc_args['eeg_marker_name'],
                    relevant_events=    self.preproc_args['relevant_events']
                )
                eeg_epoch_list.append(eeg_epoch)
                eeg_label_list.append(eeg_label)
                epoch_repetition_list.append(len(eeg_label))
            eeg_epochs = np.concatenate(eeg_epoch_list, axis=0)
            # eeg_epochs = eeg_epochs.transpose(0, 2, 1) 
            eeg_labels = np.concatenate(eeg_label_list, axis=0)
        elif self.preproc_args['format']=='cdt':
            # Update to a more parallized method
            # for f in new_files:
            #     epochs, relevant_event_id = self._get_mne_epoch_from_cdt(
            #         cft_file_path=f, events_of_interest=events_of_interest,
            #         enable_cache=enable_cache, cache_root=cache_root, force_rebuild=force_rebuild
            #     )   
            #     epoch_repetition_list.append(len(epochs.events[:,-1]))
            #     eeg_epochs.append(epochs)
            #     eeg_epoch_event_id_list.append(relevant_event_id)
            eeg_epochs, eeg_epoch_event_id_list, epoch_repetition_list = self._get_mne_epoch_from_cdt_non_blocking(
                cft_file_list=new_files, events_of_interest=events_of_interest,
                enable_cache=enable_cache, cache_root=cache_root, force_rebuild=force_rebuild
            )


        self._identifier_multiplier(2, epoch_repetition_list) # expand on the number of epochs axis
        print('Shapes','eeg_epochs:',[e.get_data().shape for e in eeg_epochs ], self.all_data_identifier.reshape(self.all_data_identifier.shape[0], -1).shape)

        ###############################################
        # put all epochs together and make the labels for the training 
        print('Combining all epochs together...' )
        eeg_labels = []
        for i, epoch in enumerate(eeg_epochs):
            n_epochs = epoch.get_data().shape[0]
            relevant_event_id = eeg_epoch_event_id_list[i] 
            relevant_event_id_reverse = {v: k for k, v in relevant_event_id.items()}
            each_epoch_label_list = []
            for j in range(n_epochs):
                event_value = epoch.events[j,-1]
                event_marker = relevant_event_id_reverse[event_value]
                if event_marker in self.marker2id: # selected vocabs                    
                    mapped_label = self.marker2id[event_marker]
                    each_epoch_label_list.append(mapped_label)
            eeg_labels.extend(each_epoch_label_list)
        eeg_labels = np.array(eeg_labels)
         
        self.final_word2id ={}
        self.final_id2word = {}
        self.final_marker2id = {}
        self.final_id2marker = {}
        available_labels = np.unique(eeg_labels)
        sorted_available_labels = sorted(available_labels)
        print('available labels in the dataset',sorted_available_labels)
        final_mapping = {}
        for new_label, original_label in enumerate(sorted_available_labels):
            final_mapping[original_label] = new_label
        # print('final_mapping',final_mapping)
        for original_label, new_label in final_mapping.items():
            original_word_of_the_label = self.id2word[original_label]
            original_marker_of_the_label = self.id2marker[original_label]
            self.final_word2id[original_word_of_the_label] = new_label
            self.final_id2word[new_label] = original_word_of_the_label
            self.final_marker2id[original_marker_of_the_label] = new_label
            self.final_id2marker[new_label] = original_marker_of_the_label
        # print('final_word2id',self.final_word2id)
        print('final_id2word',self.final_id2word)
        # print('final_marker2id',self.final_marker2id)
        # print('final_id2marker',self.final_id2marker)


        # first relabel the eeg_labels to make them continuous from 0 to N
        eeg_labels = np.array([final_mapping[l] for l in eeg_labels])
        # print('eeg_labels after remapping',eeg_labels, len(eeg_labels))
        # build the final word2id and id2word, marker2id and id2marker
        eeg_data  = [e.get_data() for e in eeg_epochs]
        eeg_info = [e.info for e in eeg_epochs]
        eeg_data = np.concatenate(eeg_data, axis=0) 
        
        self.all_mne_epochs, bad_epoch_ids = self._preprocess_epochs(
            eeg_data, eeg_labels, eeg_info[0],
            preprocess_params=self.preproc_args,              
        )
        if bad_epoch_ids is not None:
            eeg_labels = np.delete(eeg_labels, bad_epoch_ids)
        eeg_epochs= self.all_mne_epochs.get_data()
        print('full eeg_epochs',eeg_epochs.shape)
        print('full eeg_labels',eeg_labels.shape)
        print('Final sampling rate:',self.final_sampling_rate)
        print('Final channel number :',len(self.final_channel_names))
        ###########################################################
        # epoch_channel_names = self.all_mne_epochs.ch_names
        # # make sure all channel names are uppercase
        # epoch_channel_names = [ch.upper() for ch in epoch_channel_names]
        # check_numpy_channel_erp_plot(eeg_epochs, epoch_channel_names,-1, 2, title='5 blocks')
        ###########################################################

        # filther the channel if select_channels is not None
        if self.select_channels is not None:
            self.channel_update_flag = True
            
            channel_indices = [self.final_channel_names.index(ch) for ch in self.select_channels if ch in self.final_channel_names]
            eeg_epochs = eeg_epochs[:,channel_indices,:]
            self.final_channel_names = [self.final_channel_names[i] for i in channel_indices]
            print('After channel selection, eeg_epochs',eeg_epochs.shape)
            print('After channel selection, channel names',self.final_channel_names) 
        if self.split_method == 'calibration':
            # raise error if split_ratio is not between 0 and 1
            assert 0 < self.split_ratio < 1, f'split_ratio should be between 0 and 1 for <calibration> split but got {self.split_ratio}' 
            # Sequential split
            n_samples = len(eeg_epochs)
            train_size = int(n_samples*self.split_ratio)
            # Balanced split
            X_train, X_test, y_train, y_test = eeg_epochs[:train_size], eeg_epochs[train_size:], eeg_labels[:train_size], eeg_labels[train_size:]
        if self.split_method == 'random':
            assert 0 < self.split_ratio < 1, f'split_ratio should be between 0 and 1 for <random> split but got {self.split_ratio}' 
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(eeg_epochs, eeg_labels, test_size=1-self.split_ratio, random_state=42)
        if self.split_method == 'no-split':
            X_train, X_test, y_train, y_test = eeg_epochs, eeg_epochs, eeg_labels, eeg_labels
        if self.phase == 'train':
            self.all_eeg = X_train
            self.all_label = y_train
        else:
            self.all_eeg = X_test
            self.all_label = y_test         
        
        # if enable_cache and cache_root is not None:
        #     cache_file_dir = os.path.dirname(cache_file)
        #     cache_file_name = os.path.basename(cache_file)
        #     cache_config_file = "{}/{}/preprocess_config.yaml".format(cache_file_dir,self.pipeline_name)
        #     # create the folder if not exist
        #     # create a
        #     os.makedirs(cache_file_dir, exist_ok=True)             
        #     self._save_cache(cache_file)
        #     self._save_config(cache_config_file)


        if enable_cache and cache_root is not None:
            cache_file_dir = os.path.dirname(cache_file)
            cache_config_file = "{}/{}/preprocess_config.yaml".format(cache_file_dir, self.pipeline_name)
            os.makedirs(cache_file_dir, exist_ok=True)

            save_lock_path = Path(cache_file_dir) / f".{Path(cache_file).name}.lock"
            with self.file_lock_nfs(save_lock_path, lock_name=f"save_{self.hash_identifier}"):
                if os.path.exists(cache_file) and not force_rebuild:
                    print(f"[Cache] Cache already saved by another process, skipping save.")
                else:
                    print(f"[Cache] Saving cache to {cache_file}")
                    self._save_cache(cache_file)
                    self._save_config(cache_config_file)



    def _scan_data(self, no_duplicate=False, enable_cache=False, cache_root=None, force_rebuild=False):        
        ###########################################
        # Global Level Caching. 
        if enable_cache and cache_root is not None:
            # cache_file = "{}/{}/{}.pkl".format(cache_root, self.pipeline_name, self.hash_identifier)
            cache_file = "{}/{}.pkl".format(cache_root, self.hash_identifier)
            cache_dir = os.path.dirname(cache_file)
            lock_file_path = Path(cache_dir) / f".{Path(cache_file).name}.lock"                
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)            
            # Fast path: Check if cache exists (no lock needed)
            if os.path.exists(cache_file) and not force_rebuild:
                print(f"[Cache] Loading cached dataset from {cache_file}")
                instance = EEGRealtimeDataset._load_cache(cache_file)
                self.__dict__.clear() 
                self.__dict__.update(instance.__dict__)
                return             
            # Slow path: Cache doesn't exist or requires force rebuild, acquire lock to build it
            # print(f"[Cache] rebuilding cache, waiting for lock...")
            # with self.file_lock_nfs(lock_file_path, lock_name=self.hash_identifier):
            #     # Double-check after acquiring lock (another process might have built it)
            #     if os.path.exists(cache_file) and not force_rebuild:
            #         print(f"[Cache] Cache was built by another process, loading...")
            #         instance = EEGRealtimeDataset._load_cache(cache_file)
            #         self.__dict__.clear() 
            #         self.__dict__.update(instance.__dict__)
            #         return                    
            #     # Only this process will proceed to scan and build
            #     print(f"[Cache] Acquired lock, building cache, proceeding to scan available data files...")
            #     # Fall through to "Start to scan available data files" section below
            #     self._build(cache_file=cache_file, enable_cache=enable_cache, cache_root=cache_root, force_rebuild=force_rebuild, no_duplicate=no_duplicate)

            print(f"[Cache] rebuilding cache...")
            self._build(cache_file=cache_file, enable_cache=enable_cache, cache_root=cache_root, force_rebuild=force_rebuild, no_duplicate=no_duplicate)
        else:
            print('[Cache] cache_root is set to "None", or force_rebuild is "True". cannot enable cache, disable cache')
            enable_cache = False
            self._build(enable_cache=enable_cache, cache_root=cache_root, force_rebuild=force_rebuild, no_duplicate=no_duplicate)


            
    def _save_config(self, filepath):
        # remove the post fix .json or .yaml from the filepath
        filestem = pathlib.Path(filepath).stem
        file_parent = pathlib.Path(filepath).parent 
        major_config = self.preproc_args.copy()
        major_config['data_root'] = self.data_root
        # create the folder if not exist
        os.makedirs(file_parent, exist_ok=True)

        # check if it is omegaconf.dictconfig.DictConfig type
        if isinstance(major_config['data_root'], omegaconf.listconfig.ListConfig):
            omegaconf.OmegaConf.save(major_config, f"{file_parent}/{filestem}.yaml")
            print(f"Preprocessing config saved to {file_parent}/{filestem}.yaml")
        else:
            with open(filestem+'.json', 'w') as f:
                json.dump(major_config, f, indent=4)
            print(f"Preprocessing config saved to {file_parent}/{filestem}.yaml")
    def _save_cache(self, filepath):   
        if self.all_eeg is not None:
            eeg_data_size = _calculate_np_array_size(np.array(self.all_eeg))# in GB
        threshold = 2 # so if the data size is larger than 2 GB, we do not save the data
        if eeg_data_size > threshold:
            # print a warning
            print(f"Warning: The dataset size is {eeg_data_size:.2f} GB, which is larger than the threshold of {threshold} GB. The dataset should be saved by trials to avoid pickle file too large.")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.chmod(filepath, 0o777)
        print(f"Dataset saved to {filepath}")
    @classmethod 
    def _load_cache(cls, filepath):
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"Dataset loaded from {filepath}")
        return obj
    def __len__(self):
        return len(self.all_label)
    def __getitem__(self, idx):
        eeg= self.all_eeg[idx]  
        eeg = torch.tensor(eeg).float()
        label = torch.tensor(self.all_label[idx], dtype=torch.long)
        if self.vocab_groupping is not None:
            label = self.vocab_groupping[int(label)]
            label = np.int64(label)
        return eeg, label, 0, 0
    @property
    def tmin(self):
        return self.preproc_args.preproc_epochs['tmin']
    @property
    def tmax(self):
        return self.preproc_args.preproc_epochs['tmax']
    @property
    def sampling_rate(self):
        return self.final_sampling_rate if self.sampling_rate_update_flag else self.raw_sampling_rate    
def compare_lsl_epoch_and_curry_epoch():
    label_mapper ={
        '100':0,'101':1,'102':2,'103':3,'104':4,'105':5,
        '106':6,'107':7,'108':8,'109':9, '110':10,'111':11,
        '112':12,'113':13,'114':14,'115':15,'116':16,'117':17,
        '118':18,'119':19,'120':20,'121':21,'122':22,'123':23,
    }
    relevant_events = ['100','101','102','103','104','105','108','109','110','111','113','115','120','122']
    data_root = "/projects/SilSpeech/Spoken_EEG/Subjects_Corrected_v2_Cleaned/S601_Check/lsl/sess26"
    channel_info_path ='/projects/SilSpeech/Dev/SilentSpeech_Se2/LBM/LLaBrain/labrain/datasets/datasets/assets/Info_128EEG.csv'
    default_drop_channels = None
    default_preproc_args = {
        'feature_type':'wave',            
        'format':'npz',
        'output_format': 'raw',
        'pp_postfix': '_NoProcessing.set',
        'wavelet_method':None,
        'fmin':None,'fmax':None, 'fnum':None, 'fspacing':None,'fspecial':None,
        'tmin':0,'tmax':1.0, 'avg_ref':False, 'combine_sessions': False,
        'resample_fs':250, 
        'resample_freq_time':None, 
        'n_jobs':16,'batch_size':128,'epoch_data':False,'cache_session':True,'cache_trials':False,
        'relevant_events': relevant_events,
        'input_122': True,
        'channel_info_path':channel_info_path,
        'default_drop_channels':['10','11', 'VEO', 'HEO', 'EKG', 'EMG', '84', '85', '110', '111', 'Trigger']
    }
    lsl_dataset = EEGRealtimeDataset(
        data_root=data_root, 
        phase='train',
        split_method='no-split',
        split_ratio=0.8,
        label_mapper=label_mapper,
        preproc_args=default_preproc_args,
    )
    lsl_dataset._scan_data()
    
    preproc_args_curry = {
        'feature_type':'wave',            
        'format':'curry',
        'output_format': 'epoch',
        'pp_postfix': '_NoProcessing.set',
        'wavelet_method':None,
        'fmin':None,'fmax':None,'fnum':None, 'fspacing':None,'fspecial':None,
        'tmin':0,'tmax':1.0, 'avg_ref':False, 'combine_sessions': True,
        'resample_fs':250, 
        'resample_freq_time':None, 
        'n_jobs':16,'batch_size':128,'epoch_data':False,'cache_session':False,'cache_trials':True,
        'relevant_events': ['100','101','104','105','108','109','110','111','113','115','120','122'],
    }
    curry_dataset = SpeechWordDatasetV5(
        phase='val', 
        data_root='/projects/SilSpeech/Spoken_EEG/Subjects_Corrected_v2_Cleaned/S601_Check/curry',
        cache_folder='/projects/SilSpeech/Spoken_EEG/Subjects_Corrected_v2_Cleaned/S601_Check/curry',
        split_method='leave-one-session-out',
        subjects= ['S601'], 
        val_subject='S601',
        val_sessions=[26], 
        test_subject='S601',
        test_sessions=[26],
        select_channels=None,
        select_vocabs =None, 
        preproc_args = preproc_args_curry,            
        force_rebuild_index =True,
        force_rebuild_feature = False,
        debug = False,
        output_dict=True,
        random_seed = 42,
        output_meta_label = False,
    )
    print('lsl dataset size dataset_train {}'.format(len(lsl_dataset)))
    print('curry_dataset size dataset {}'.format(len(curry_dataset)))
    # for i in range(1):
    #     print('eeg',curry_dataset[i]['eeg'].shape, 'label',curry_dataset[i]['label'])

    curry_epoch = curry_dataset[0]['eeg']
    curry_label = curry_dataset[0]['label']
    lsl_epoch = lsl_dataset[0][0]
    lsl_label = lsl_dataset[0][1]
    print('lsl epoch shape', lsl_epoch.shape)
    print('curry epoch shape', curry_epoch.shape)
    # check if the two epochs are equal
    # make the 
    # 1: check channel order 
    print('lsl channel names:\n', lsl_dataset.channel_names)
    print('curry channel names:\n', curry_dataset.channel_names)

    # 2: check if the two epochs are equal
    # output to a csv file by creating a pandas 
    # dataframe
    import pandas as pd
    output_folder= './preprocessing_check'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True)
        #grant permission to the folder
        os.chmod(output_folder, 0o777)
    # for each channel, save the data
    lsl_epoch_dict = {}
    for i in range(len(lsl_epoch)):
        lsl_epoch_dict[lsl_dataset.channel_names[i]] = lsl_epoch[i]
    lsl_epoch_df = pd.DataFrame(lsl_epoch_dict)
    lsl_epoch_df.to_csv(f'{output_folder}/lsl_epoch.csv', index=False)
    curry_epoch_dict = {}
    for i in range(len(curry_epoch)):
        curry_epoch_dict[curry_dataset.channel_names[i]] = curry_epoch[i]
    curry_epoch_df = pd.DataFrame(curry_epoch_dict)
    curry_epoch_df.to_csv(f'{output_folder}/curry_epoch.csv', index=False)
    # check if the two epochs are equal
    # for each channel in the channel name, plot the two epochs
    for i in range(len(lsl_epoch)):
        plt.plot(lsl_epoch[i], label='lsl')
        plt.plot(curry_epoch[i], label='curry')
        plt.legend()
        plt.title(f'Channel {lsl_dataset.channel_names[i]}')
        plt.savefig(f'{output_folder}/channel_{lsl_dataset.channel_names[i]}.png')
        plt.close()
        # exit(0)

# TODO: allow other process who could not get the lock for a single cdt file to be able to process the others firstly, and then come back to load or process the locked cdt file, instead of just waiting for the lock to be released. This can improve the efficiency when there are multiple cdt files to process and some of them are locked by other processes.
# if __name__ == '__main__':
#     # cdt_path = "/projects/SSNFB/BrainGPT_Collected_Dataset/NeuroFeedback_Stim_Selection_Pilot/S11/day_1/condition_A_1.cdt"
#     # output_folder = '/projects/SilSpeech/Dev/SilentSpeech_Se2/LBM/LLaBrain/labrain/datasets/datasets/assets/'
#     # create_channel_spec(cdt_path, output_folder, 'curry8_128_channel')
#     # # debug_cdt_datasets_imageined_speech_Pilot_test_glob_and_test_channel()
#     # cdt_path = "/projects/SSNFB/BrainGPT_Collected_Dataset/NeuroFeedback_Stim_Selection_Pilot/S1/day_1/condition_1_1.cdt"
#     # output_folder = '/projects/SilSpeech/Dev/SilentSpeech_Se2/LBM/LLaBrain/labrain/datasets/datasets/assets/'
#     # create_channel_spec(cdt_path, output_folder, 'curry9_128_channel')


#     cdt_path = "/projects/SSNFB/BrainGPT_Collected_Dataset/Co_Training/Sub2/CoTrainimgData.cdt"
#     output_folder = '/projects/SilSpeech/Dev/SilentSpeech_Se2/LBM/LLaBrain/labrain/datasets/datasets/assets/'
#     create_channel_spec(cdt_path, output_folder, 'curry8_64_channel')


import argparse
from omegaconf import OmegaConf


def build_test_dataset(cfg_path: str):
    print("path", cfg_path)
    # ── 1. load config ────────────────────────────────────────────────────────
    raw      = OmegaConf.load(cfg_path)
    cfg      = raw

    # ── 3. build test dataset ─────────────────────────────────────────────────
    test_dataset = EEGRealtimeDataset(
        data_root              = cfg.test_dataset_data_root,
        phase                  = "test",
        split_method           = cfg.test_dataset_split_method,
        split_ratio            = 0.8,
        label_mapper           = None,
        vocab_groupping        = cfg.get("test_vocab_groupping"),
        select_channels        = cfg.get("select_channels"),
        preproc_args           = cfg,
        select_vocabs          = cfg.get("select_vocabs"),
        select_vocabs_remap_index = cfg.get("select_vocabs_remap_index", True),
        decode_time            = cfg.decode_time,
        original_sampling_rate = 1000,
    )

    test_dataset._scan_data(
        enable_cache  = cfg.get("enable_dataset_cache", False),
        cache_root    = cfg.get("dataset_cache_root",   None),
        force_rebuild = cfg.get("force_rebuild_cache",  False),
    )

    # ── 4. quick sanity print ─────────────────────────────────────────────────
    print(f"[✓] test_dataset built — {len(test_dataset)} samples")
    sample = test_dataset[0]
    if isinstance(sample, (tuple, list)):
        print(f"    sample[0] shape : {sample[0].shape}")
        print(f"    sample[1] (label): {sample[1]}")
    else:
        print(f"    sample type : {type(sample)}")

    return test_dataset


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("cfg", help="path to your yaml config file")
    # args = parser.parse_args()
    yaml_path = "C:/Users/IDBA/Downloads/UTS_EEG/Dataloader/lo1_hi50_CleanChan.yaml"
    build_test_dataset(yaml_path)