# RoBERTa-Namend-Entity-Recognition-German-Medical-Data
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
