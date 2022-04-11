# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 22:10:37 2021

@author: Jayanth
"""

import logging
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable
import os
import sys
import json
import importlib

import numpy as np
from numpy import ndarray
import transformers

import torch
from torch import nn, Tensor,device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.multiprocessing as multiprocess
from tqdm.autonotebook import trange

import math
import shutil
from collections import OrderedDict
from zipfile import ZipFile
import requests
from tqdm.autonotebook import tqdm

from utils import import_from_string, batch_to_device, http_get

logger = logging.getLogger(__name__)

class Sentinal(nn.Sequential):
    
    def __init__(self,model_name_or_path: str = None, modules : Iterable[nn.Module] = None, device : str = None):
        save_model_to = None
        
        if model_name_or_path is not None and model_name_or_path != "":
            logger.info("Load pretrained SentenceTransformer: {}".format(model_name_or_path))
            model_path = model_name_or_path
            # If model is in not in disk then
            if not os.path.isdir(model_path) and not model_path.startswith('http://') and not model_path.startswith('https://'):
                logger.info("Did not find folder {}".format(model_path))
    
                if '\\' in model_path or model_path.count('/') > 1:
                    raise AttributeError("Path {} not found".format(model_path))
    
                model_path = __DOWNLOAD_SERVER__ + model_path + '.zip'
                logger.info("Search model on server: {}".format(model_path))
            if model_path.startswith('http://') or model_path.startswith('https://'):
                model_url = model_path
                folder_name = model_url.replace("https://", "").replace("http://", "").replace("/", "_")[:250][0:-4] #remove .zip file end

                cache_folder = os.getenv('SENTENCE_TRANSFORMERS_HOME')
                                
                model_path = os.path.join("D:\models",model_name_or_path)

                    
                if not os.path.exists(model_path) or not os.listdir(model_path):
                    model_url = model_url.rstrip("/")
                    logger.info("Downloading sentence transformer model from {} and saving it at {}".format(model_url, model_path))
    
                    model_path_tmp = model_path.rstrip("/").rstrip("\\")+"_part"
                    try:
                        zip_save_path = os.path.join(model_path_tmp, 'model.zip')
                        print("zip path",zip_save_path)
                        http_get(model_url, zip_save_path)
                        with ZipFile(zip_save_path, 'r') as zip:
                            zip.extractall(model_path_tmp)
                        os.remove(zip_save_path)
                        os.rename(model_path_tmp, model_path)
                    except requests.exceptions.HTTPError as e:
                        print(e.reponse.status_code)
                        shutil.rmtree(model_path_tmp)
                        if e.response.status_code == 429:
                            raise Exception("Too many requests were detected from this IP for the model {}. Please contact info@nils-reimers.de for more information.".format(model_name_or_path))
    
                        if e.response.status_code == 404:
                            logger.warning('SentenceTransformer-Model {} not found. Try to create it from scratch'.format(model_url))
                            logger.warning('Try to create Transformer Model {} with mean pooling'.format(model_name_or_path))
    
                            save_model_to = model_path
                            model_path = None
                            transformer_model = Transformer(model_name_or_path)
                            pooling_model = Pooling(transformer_model.get_word_embedding_dimension())
                            modules = [transformer_model, pooling_model]
                        else:
                            raise e
                    except Exception as e:
                        shutil.rmtree(model_path)
                        raise e
                        
                if os.path.exists(model_path):
                    logger.info("Load SentenceTransformer from folder: {}".format(model_path))

                    if os.path.exists(os.path.join(model_path, 'config.json')):
                        with open(os.path.join(model_path, 'config.json')) as fIn:
                            config = json.load(fIn)
                            if config['__version__'] > __version__:
                                logger.warning("You try to use a model that was created with version {}, however, your version is {}. This might cause unexpected behavior or errors. In that case, try to update to the latest version.\n\n\n".format(config['__version__'], __version__))
    
                    with open(os.path.join(model_path, 'modules.json')) as fIn:
                        contained_modules = json.load(fIn)
    
                    modules = OrderedDict()
                    for module_config in contained_modules:
                        module_class = import_from_string(module_config['type'])
                        module = module_class.load(os.path.join(model_path, module_config['path']))
                        modules[module_config['name']] = module


        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        #We created a new model from scratch based on a Transformer model. Save the SBERT model in the cache folder
        if save_model_to is not None:
            self.save(save_model_to)
            
    def encode(self, sentences: Union[str, List[str], List[int]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value == 'token_embeddings':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features)

                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                        last_mask_id = len(attention)-1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0:last_mask_id+1])
                else:   #Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings
    
    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings
    
    def tokenize(self, text: str):
        """
        Tokenizes the text
        """
        return self._first_module().tokenize(text)
    
    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

    def save(self, path):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders
        """
        if path is None:
            return

        os.makedirs(path, exist_ok=True)

        logger.info("Save model to {}".format(path))
        contained_modules = []

        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            model_path = os.path.join(path, str(idx)+"_"+type(module).__name__)
            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            contained_modules.append({'idx': idx, 'name': name, 'path': os.path.basename(model_path), 'type': type(module).__module__})

        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(contained_modules, fOut, indent=2)

        with open(os.path.join(path, 'config.json'), 'w') as fOut:
            json.dump({'__version__': __version__}, fOut, indent=2)
            
    

            
def main():
    a = Sentinal('paraphrase-MiniLM-L6-v2')
    print(a.encode(['hello']))
    print("Hi")
        
main()
    
        


