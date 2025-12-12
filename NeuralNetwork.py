from huggingface_hub import Padding
from transformers import AutoTokenizer, AutoModel

model_name = "dmis-lab/biobert-base-cased-v1.1"

tokenizer = AutoTokenizer.from_pretrained(model_name) # tranforma texto em numeros
model = AutoModel.from_pretrained(model_name) # realiza o processamneto do texto tokenizado em números

print("Modelo carregado com sucesso!")
