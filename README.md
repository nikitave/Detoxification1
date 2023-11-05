# Detoxification
It's the project which can be used to transform toxic sentences into friendlier ones. 

To start my project you need to do several things:

1. [Download](https://drive.google.com/drive/u/0/folders/1TnBgiH7uH5Ff-rf0z2vBbvEtief6zj6o) weights for the model. You need to put these unarchived weights into the <b>models</b> folder. 
2. Use this [script](./src/data/make_dataset.py) to download data. Put the <b>data</b> folder
3. Install requirements
```bash
pip3 install -r requirements.txt
```
4. Use train.py to train the model.
5. Use test.py to test the model.
6. Use predict_model.py to see how model works with your sentences.
