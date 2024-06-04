
# example of fitting an auxiliary classifier gan (ac-gan) on fashion mnsit
from numpy import zeros
from numpy import ones
import numpy as np
import pandas as pd
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from matplotlib import pyplot
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import os
#from torch.utils.data import Dataset, DataLoader

class Data():
    def __init__(self, split, subject,  dataset,path):
        
        assert split in ['train', 'test']
        
        self.dataset_root = path
        self.dataset_path = os.path.join(self.dataset_root, dataset)
        self.subject = subject        
        self.split = split
        self.x = None
        self.y = None
        


    def load(self):

        self.x = np.load(self.dataset_path+f"/Subject_{self.subject}/"+'X_train.npy', allow_pickle=True)

        if self.split == 'train':
            self.y = np.load(self.dataset_path+f"/Subject_{self.subject}/"+'y_train.npy', allow_pickle=True)
        else:
            self.y = np.load(self.dataset_path+f"/Subject_{self.subject}/"+'y_test.npy', allow_pickle=True)
        
        
        
        
        
        return self.x, self.y
    
    
def y_translator(y,subject,dataset):
    # Define your letters
    letters = set(y)
    y_train_original = y
    # Create a dictionary to map letters to numbers
    letter_to_number = {letter: i for i, letter in enumerate(letters)}

    path = f"{dataset}/dictionary/"

    os.makedirs(path, exist_ok=True)

    np.save(f'{path}Subject_{subject}.npy', letter_to_number) 

    # Test the conversion
    y_train_new=[]
    for letter in y_train_original:
        input_letter = letter
        if input_letter in letter_to_number:
                y_train_new.append(letter_to_number[input_letter])
            #print(f"The number corresponding to {input_letter} is {letter_to_number[input_letter]}")
        else:
            print(f"{input_letter} is not a valid letter.")
    return(np.array(y_train_new))


def load_real_samples(split,subject,dataset):

    test = Data(split,subject=subject,dataset=dataset,path=f'')

    X,y = test.load()
    y = y_translator(y,subject,dataset)
    y = y.reshape(-1, 1)
    return [X,y]


import numpy as np
import torch
from datasets import Dataset
from transformers import BertTokenizer, DataCollatorWithPadding, BertForSequenceClassification, Trainer, TrainingArguments

def load_real_samples(split, subject, dataset):
    data = Data(split, subject=subject, dataset=dataset, path=f'')
    X, y = data.load()
    y = y_translator(y, subject, dataset)
    y = y.reshape(-1, 1)
    return [X, y]

dataset = load_real_samples("train", subject='02', dataset="FALL-mag")
X = dataset[0]
y = dataset[1]

# Convert time series data to strings
X_str = [" ".join(map(str, x.flatten())) for x in X]
y_flat = y.flatten()

# Convert to Hugging Face dataset
hf_dataset = Dataset.from_dict({
    'text': X_str,
    'labels': y_flat.tolist()
})

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
