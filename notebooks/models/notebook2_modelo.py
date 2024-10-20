# -*- coding: utf-8 -*-
"""Notebook2_modelo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14owsAoLlH4I2U-52vdQTK-3PAA1q6y1D

#Notebook 2 - Criação do Modelo

  Como essa tarefa se trata de classificação de texto iremos utilizar um modelo BERT. Iremos utilizar um modelo já treinado e realizaremos o ajuste de parametros através da técnica de fine tuning chamada LORA (Low-Rank Adaptation of Large Language Models) descrita [neste artigo](https://arxiv.org/abs/2106.09685). Está ecolha foi feita para aproveitar o conhecimento já existente em modelos pré treinados e economizar recursos. Para tal, será utilizado as bibliotecas disponiveis pelo [Hugging Face](https://huggingface.co/).
"""

from google.colab import drive
import os

drive.mount('/content/drive')  #montanto o drive
os.chdir('/content/drive/MyDrive/IPM_processo_seletivo/questao_2') #diretório onde previamente foram colocados os dados

#verifica se a gpu esta ativa e qual estamos usando
!/opt/bin/nvidia-smi
!nvcc --version

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install datasets
# !pip install tokenizers
# !pip install torchmetrics
# !pip install transformers
# !pip install peft
# !pip install evaluate
#

import torch
import numpy as np
import matplotlib . pyplot as plt
import csv
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertModel,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate

#lendo os dataset
train_data = Dataset.from_csv('train_data.csv')
test_data = Dataset.from_csv('test_data.csv')


data = {"train": train_data , "test": test_data}
final_data = DatasetDict(data)    #cria um dicionário de datasets para treino e teste
print(final_data)

"""Será utilizado do modelo [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) que se trata de um modelo BERT com menos parametros e mais rápido que o Bert original."""

#define o modelo base
base_model = 'distilbert-base-uncased'


# define o mapeamento dos rótulos
id2label = {0: "Non  clickbait", 1: "Clickbait"}
label2id = {"Non  clickbait":0, "Clickbait":1}

# Define o uso da gpu se disponivel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#cria o modelo
model = DistilBertForSequenceClassification.from_pretrained(base_model, num_labels=2, id2label=id2label, label2id=label2id).to(device)
print(model)

"""Acima definimos o modelo e podemos ver sua arquitetura, para realizar o ajuste de parametros vamos treinar as redes que fazer a projeção linear no mecanismo de atenção q_lin, k_lin e v_lin"""

# cria o tokenizador baseado no modelo usado
tokenizer = DistilBertTokenizer.from_pretrained(base_model, add_prefix_space=True)

# adiciona o token de padding para completar sentenças que sejam menores que a max_lenght
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


#função que realiza a tokenização dos datasets
def tokenize_function(examples):

    text = examples["text"]
    #trunca e tokeniza o dataset

    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
       text,
        return_tensors="pt",
        truncation=True,
         padding=True,
        max_length=200
    )

    return tokenized_inputs.to(device)

tokenized_dataset = final_data.map(tokenize_function, batched=True)
print(tokenized_dataset)

#Vizualização da tokenização
for j in range(5):
    # Acessa a linha j do dataset 'test'
    item = tokenized_dataset['test'][j]
    for i in item:
        print(i, ":", item[i])
    print("---------------")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# calcula a acuracia
accuracy = evaluate.load("accuracy")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

#configuração do LORA
peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=10,
                        lora_alpha=16,
                        lora_dropout=0.01,
                        target_modules = ["q_lin","k_lin","v_lin"])
    #só foca nos modulos de auto atenção

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_peft_model(model, peft_config,).to(device)
model.print_trainable_parameters()

""" r: Controla o rank da decomposição das matrizes de peso, afetando o número de parâmetros treináveis e a eficiência do modelo. Valores menores de r reduzem a dimensionalidade das atualizações, tornando o modelo mais eficiente em termos de memória e computação.

lora_alpha: Um fator de escalonamento que ajusta a magnitude das atualizações feitas durante o treinamento, controlando a influência das adaptações de baixo rank no modelo original e ajudando na regularização.

A partir das configuraçẽos selecionadas acima para o ajuste de parametros, vamos treinar apeans 1.2% da arquitetura, isto nos dara um modelo com senso da tarefa especifica sem precisar treinar todos os 67 M de parametros.
"""

#função para realizar o salvamento do modelo de cada época
class SaveModelCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        model.save_pretrained("finetuned_model/"+ base_model + "-lora-clikbait_/" + f"model_epoch_{state.epoch}")

# hiperparametros, foram escolhidos estes parametros pois verificou-se previamente que tem um bom resultado
lr = 5e-4
batch_size = 24
num_epochs = 5

# define os argumentos de treino
training_args = TrainingArguments(
    output_dir=  base_model + "-lora-clikbait",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
)

# define o treinador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

#Verifica o desenpenho do modelo não treinado
trainer.evaluate()

"""A acurácia do modelo não treinado é de 50%, ou seja, o modelo está chutando o resultado ao acaso. A seguir o modelo será treinado por 5 época e veremos os resultados."""

# Adicione o retorno de chamada ao Trainer
# Crie uma instância da classe callback e adicione-a ao Trainer
save_model_callback = SaveModelCallback()
trainer.add_callback(save_model_callback)


trainer.train()

#carrega o modelo treinado para avaliar o desempenho
path_model_trained = "finetuned_model/distilbert-base-uncased-lora-clikbait_/model_epoch_5.0"
model_trained = DistilBertForSequenceClassification.from_pretrained(path_model_trained, num_labels=2, id2label=id2label, label2id=label2id).to(device)

trainer = Trainer(
    model=model_trained,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.evaluate()

"""Pode-se então perceber que após uma unica época o modelo já é capaz de alcançar cerca de 97% de acurácia e após 5 épocas obtivemos 98,84% de acuracia nos dados de teste. Indicando que o modelo e os parametros estão condizentes com a tarefa.

Por fim, a seguir é feita a classificação de 5 frases nunca vistas, reforçando o desempenho positivo do modelo criado.
"""

model.eval()

text_list = ["Check out the marketing infographic",
             "Canada pursues new nuclear research reactor to produce medical isotopes",
             "This Is the Real Reason Doctors Make You Sit on That Tissue Paper",
             "Cuban talk show accuses U.S. diplomat of helping anti-government groups",
             "The 10 Hacks You Need to Stay Healthy This Winter",]

print("Predições do modelo em frases nunca vistas:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)

    logits = model(inputs).logits
    predictions = torch.max(logits,1).indices

    print(text + " - " + id2label[predictions.tolist()[0]])

"""Como dito no primeiro notebook podemos criar um backed utilizando este modelo para denvolver um SAAS capaz de detectar manchetes sensacionalistas. Melhorias no modelo envolveriam a utilização de um dataset mais amplo contendo outras fontes de material sensacionalista."""