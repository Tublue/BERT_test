from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# df_essay is the text dataset I used, "text" is pure text data colunm as str 
tokenized_essay = df_essay["text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

proper_tokenized_essay = []
filter_length = 400

# padding
for i, text in enumerate(tokenized_essay):
  temp_text = random_start(text)
  proper_tokenized_essay.append(temp_text + [0]*(MAX_TEXT_LENGTH - len(temp_text)))

# This BERT model need a attention mask
proper_tokenized_essay_np = np.array(proper_tokenized_essay)
attention_mask = np.array(np.where(proper_tokenized_essay_np == 0, 0, 1))

batch_size = 32
essay_feature = np.zeros(shape=(1, 768))
input_ids_total = len(proper_tokenized_essay_np)
for i in range(0, input_ids_total, batch_size):

  if ((i % (batch_size*5)) == 0):
    print('  Tokenized {:,} samples.'.format(i))
  input_ids_end = min(i+batch_size, input_ids_total)
  # print("input_ids_end: ", input_ids_end)
  input_ids = torch.tensor(proper_tokenized_essay_np[i:input_ids_end])
  input_attention_mask = torch.tensor(attention_mask[i:input_ids_end])
  # print(input_ids.shape)
  # print(attention_mask.shape)
  with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=input_attention_mask)
  features = last_hidden_states[0][:,0,:].numpy()
  essay_feature = np.concatenate((essay_feature, features), axis = 0)
