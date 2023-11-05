import os

os.system("wget https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip")
os.system("unzip filtered_paranmt.zip")
os.system("mv filtered.tsv data")