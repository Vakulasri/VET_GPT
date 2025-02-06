# Vet GPT: Pre-Trained GPT Model 

## Introduction
The project implements a PyTorch-based GPT model designed to answer questions about "cats and dogs" using data scraped from a veterinary website. The model is built on a transformer architecture and incorporates a question-answering (Q&A) function that retrieves answers based on keyword matching from the input question.

## Execution Environment
This project has been executed in Google Colab for its ease of use, dependency management, and GPU support.

## Data Scraping
The Python script scrapes content related to "Cats" and "Dogs" from the veterinary website [Veterinary Partner](https://veterinarypartner.vin.com/default.aspx?pId=19239&catId=102887). Using the `requests` library, the script fetches HTML content, and with the help of `BeautifulSoup`, it extracts the relevant data by identifying sections marked by `<h2>` tags and collecting following paragraphs (`<p>`) or div elements (`<div>`).

- Extracted content is cleaned by removing dates and unnecessary blank lines.
- The content is saved into a text file (`cats_and_dogs_data.txt`) for further processing.
- The script ensures that each page is only visited once and follows only links from the same domain.
- The script pauses between requests to prevent overwhelming the server.

After scraping, the total length of the extracted content is displayed.

## Main Components

### 1. Imported Libraries
Key libraries include:
- `torch`: For implementing deep learning models.
- `numpy`: For handling numerical operations.
- `tiktoken`: For tokenizing text data.
- `torch.nn`: For building neural network layers.

### 2. Setting Hyperparameters
Several hyperparameters define the model’s structure and training:
- `block_size`, `batch_size`, `max_iters`, `learning_rate`: Establish training parameters.
- `n_embd`, `n_head`, `n_layer`: Define the architecture size and number of attention layers.

### 3. Tokenizing Text
The text data is tokenized using Tiktoken’s GPT-3 tokenizer, converting characters into integer IDs. The mappings for text encoding and decoding are handled through `stoi` (string-to-integer) and `itos` (integer-to-string).

### 4. Data Splitting and Batch Processing
- The dataset is split into training (90%) and validation sets.
- The `get_batch` function creates input and target batches by selecting random indices within the dataset.

### 5. Model Structure
The model consists of several transformer-based components:
- **Head**: Creates a single attention head with key, query, and value projections.
- **FeedForward**: Implements a two-layer feedforward neural network with ReLU activation.
- **MultiHeadAttention**: Combines multiple attention heads, with dropout for regularization.
- **Block**: Represents a transformer block, stacking multi-head attention and feedforward layers.
- **GPTModel**: The primary model class containing embedding tables, positional encoding, transformer blocks, and a final linear layer for token prediction.

### 6. Model Training
- **Loss Calculation**: `estimate_loss()` computes the average losses on both training and validation sets.
- **Optimization Loop**: The training loop iterates over batches, calculates cross-entropy loss, and updates weights using the Adam optimizer.
- **Checkpoints**: `eval_interval` checkpoints monitor model progress during training and validation.

### 7. Generating Text
- The `generate` function creates text by sampling new tokens based on an initial input prompt.

### 8. Q&A System
- **Data Loading**: `load_cleaned_data` imports and prepares the cleaned dataset.
- **Question Processing**: 
  - `clean_question`: Removes special characters and standardizes the question.
  - `extract_keywords_from_question`: Filters out common words to retain only relevant keywords.
- **Answer Retrieval**: `search_answer` searches for matching keywords in the dataset and returns relevant lines as answers.
- **User Interface**: A loop allows users to input questions and displays answers based on keyword matching.

## Model Performance
The model's performance is evaluated during training using `eval_interval` checkpoints, which display the training and validation losses, helping track the model's learning progress.

## Conclusion
This project demonstrates the application of a GPT model for answering questions about "cats and dogs." By combining a transformer architecture for text generation with a keyword-based Q&A system, the project offers a robust solution for generating data-driven responses based on veterinary knowledge.

## How to Run the Project

### Setup Instructions
- Clone the repository.
- Install the required libraries:
   ```bash
   !pip install tiktoken
   
- Run the file Scraping.ipynb.
- Later run the file Pre_Train.ipynb.
