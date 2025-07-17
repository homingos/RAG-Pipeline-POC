# Local PDF RAG with Llama.cpp on Apple Silicon

This project is a Proof of Concept (PoC) for a Retrieval-Augmented Generation (RAG) system that allows you to chat with your PDF documents completely offline. It is optimized to run locally on Apple Silicon Macs (M1/M2/M3/M4) by leveraging the Metal GPU for accelerated performance via `llama-cpp-python`.

## Features
- **100% Offline & Private:** Your documents and questions never leave your machine.
- **Apple Silicon Optimized:** Uses the Metal framework for fast GPU-accelerated LLM inference.
- **Configurable:** All key parameters are controlled via a simple `config.json` file.
- **Flexible Answering Modes:** Easily switch between a "strict" mode (answers ONLY from the PDF) and a "flexible" mode (allows the model to use its general knowledge).
- **Simple & Self-Contained:** Built with Python and popular libraries, making it easy to understand and extend.

---

## Project Structure
```
.
├── config.json               # Main configuration file for all parameters
├── Attendance Leave Policy.pdf # Your knowledge base PDF (add your own)
├── Mistral-7B-Instruct-v0.2-GGUF/ # Folder for your model file
├── rag_mac_poc.py            # The main Python script
├── requirements.txt          # List of Python dependencies
└── .gitignore                # Specifies files for Git to ignore
```

---

## Setup and Installation

Follow these steps to get the project running on your macOS machine.

### 1. Prerequisites
Ensure you have the following installed:
- **Xcode Command Line Tools:** `xcode-select --install`
- **Homebrew:** [Install Homebrew](https://brew.sh/)
- **Python 3.10+:** `brew install python`

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-name>
```

### 3. Set up the Python Virtual Environment
This isolates the project's dependencies from your system.
```bash
# Create the virtual environment
python3 -m venv venv

# Activate the environment (do this every time you work on the project)
source venv/bin/activate
```
Your terminal prompt should now start with `(venv)`.

### 4. Install Dependencies
Install all required Python packages using the `requirements.txt` file, then install `llama-cpp-python` with Metal support.
```bash
# Upgrade pip
pip install --upgrade pip

# Install packages from requirements.txt
pip install -r requirements.txt

# Install llama-cpp-python with Metal GPU support (this is a special command)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U --force-reinstall --no-cache-dir llama-cpp-python
```

### 5. Download the Language Model
This project is configured to use the **Mistral 7B Instruct GGUF** model.
1.  Download the recommended model file: [**`mistral-7b-instruct-v0.2.Q4_K_M.gguf`**](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf) (Size: ~4.37 GB).
2.  Place the downloaded `.gguf` file inside the `Mistral-7B-Instruct-v0.2-GGUF` directory in your project folder.

---

## Configuration

All settings are controlled by the **`config.json`** file. Before running, you **must update the file paths** to match your system.

```json
{
  "pdf_path": "/path/on/your/mac/to/your.pdf",
  "model_path": "/path/on/your/mac/to/your-model.gguf",
  "retrieval_params": {
    "chunk_size": 1000,
    "chunk_overlap": 100
  },
  "generation_params": {
    "temperature": 0.0,
    "max_tokens": 256,
    "use_strict_prompt": true
  }
}
```
**Key Parameters:**
- `pdf_path` & `model_path`: **Update these to the absolute paths on your computer.**
- `temperature`: `0.0` for factual, deterministic answers. Increase towards `1.0` for more creative responses.
- `use_strict_prompt`:
  - `true`: The model will **refuse** to answer questions not found in the PDF.
  - `false`: The model will use its own knowledge if the answer is not in the PDF.

---

## How to Run
Once the setup and configuration are complete, run the application with this simple command:
```bash
# Make sure your virtual environment is active (venv)
python3 rag_mac_poc.py
```
The first time you run it, it will download the embedding model (~90MB). Subsequent runs will be faster. The first query will also take a moment as the main LLM loads into your GPU memory.

## Usage Example

**With `"use_strict_prompt": true`:**
```
Ask a question: who is the president of india
Thinking...
Answer: I cannot answer this question as the information is not found in the provided document.
```

**With `"use_strict_prompt": false`:**
```
Ask a question: who is the president of india
Thinking...
Answer: The provided document does not contain information about the president of India. Based on my general knowledge, the current president of India is Droupadi Murmu.
```
