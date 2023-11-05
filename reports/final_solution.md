# Introduction
The Text Detoxification Task involves converting text with toxic elements into neutral language while preserving the original meaning. It plays a crucial role in promoting respectful online communication, addressing concerns related to offensive rhetoric.

# Data analysis
Firstly, I found an average toxicity of the reference sentences: <b>0.54</b>, average toxicity of the translated sentences: <b>0.43</b>, and the similarity between referenced and translated sentences: <b>0.76</b>.
<br> Secondly, I decided to find the average level of toxicity for thirteen popular words (<i>"fuck", "shit", "kill", "you", "idiot", "fool", "want", "wanna",
        "fajhkjg", "feel", "hate", "faggot", "hello"</i>) to better understand how it works.
[Here](../notebooks/1.0-exploration.ipynb) you can learn about it more.

# Model Specification
I used t5-base for the project. Firstly, I tried to use t5-small and t5-large, but they failed. 

# Training Process
I used batch size=54, 3 epochs, and the whole dataset for the training purposes. I used <b>transformers</b> library for learning. 

# Evaluation
I decided to use such metrics for my model: find average similarity of the referenced and translated sentences, and then find the difference between toxicity of the translated and referenced sentences. For this purposes I used pretrained models: [similarity model](https://www.sbert.net/docs/usage/semantic_textual_similarity.html), [toxicity model](https://huggingface.co/cointegrated/rubert-tiny-toxicity). 
# Results
[My model](https://drive.google.com/drive/folders/1TnBgiH7uH5Ff-rf0z2vBbvEtief6zj6o?usp=sharing)
<br> And the result of my metrics are following: 
<br>Average similarity between sentences: <b>0.85</b>
<br>Average difference between toxicity levels the translated and referenced: <b>0.05</b> 