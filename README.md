# RoBERTa-Namend-Entity-Recognition-Tagging-German-Medical-Data
## Description
This project uses named entity recognition (NER) techniques to extract important information from medical texts written in German. The NER model is trained on annotated German medical text data using the RoBERTa and SimpleTransformer frameworks. The resulting model is then used to identify and classify named entities in medical texts, such as patient names, diagnoses, and treatment methods. The project aims to demonstrate the effectiveness of NER for extracting valuable information from medical texts, and to provide a useful resource for anyone interested in using NER techniques in the medical domain.


## RoBERTa
RoBERTa (short for "Robustly Optimized BERT Pretraining Approach") is a transformer-based language model developed by Facebook's AI Research (FAIR) lab. It was designed to improve upon the original BERT (Bidirectional Encoder Representations from Transformers) model, which had achieved state-of-the-art results on a wide range of natural language processing (NLP) tasks.


## Namend Entity Recognition with RoBERTa
To perform NER with RoBERTa, the model is typically fine-tuned on an annotated dataset of text data that includes named entities and their labels. This can be done using a supervised learning approach, in which the model is trained to predict the named entity labels of each word in the dataset, given the words that come before and after it. The model is then evaluated on a separate test set to measure its performance on the NER task.

Once the RoBERTa model has been fine-tuned for NER, it can be used to label named entities in new, unseen text data. This is done by feeding the new text data into the model and using the model's predictions as the named entity labels for each word. The model's output can be used to extract important information from the text data, such as the names of individuals, organizations, locations, and other entities.


## Usage
**Linux/macOS**
1. Open a terminal and navigate to the directory where you want to create your Python environment.
2. Use the command `python3 -m venv .env` to create a virtual Python environment named ".env".
3. Activate the virtual Python environment with the command source `.env/bin/activate`. You should now see an arrow (e.g. "(.env)") in front of the prompt in your terminal, indicating that the virtual environment is activated.
4. Use the command `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116` to install pytorch. Make sure that you select the cuda version. [Look here to see the documentation](https://pytorch.org/)
5. Use the command `pip install -r requirements.txt` to install the remaining dependencies.
6. Use the command `python -m RoBERTa_NER_Tagger` to run the script

**Windows**
1. Open a command prompt window and navigate to the directory where you want to create your Python environment.
2. Use the command `python -m venv .env` to create a virtual Python environment named ".env".
3. Activate the virtual Python environment with the command `.env\Scripts\activate.bat`. You should now see an arrow (e.g. "(.env)") in front of the prompt in your command prompt window, indicating that the virtual environment is activated.
4. Use the command `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116` to install pytorch. Make sure that you select the cuda version. [Look here to see the documentation](https://pytorch.org/)
5. Use the command `pip install -r requirements.txt` to install the remaining dependencies.
6. Use the command `python -m RoBERTa_NER_Tagger` to run the script
