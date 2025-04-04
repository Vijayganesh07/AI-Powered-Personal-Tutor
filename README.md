# 📚 AI-Powered Personal Tutor

An intelligent educational platform that provides personalized learning experiences through document analysis and adaptive questioning.

## Features

- **Knowledge Assessment Quiz**: Evaluates students' current knowledge level through grade-appropriate questions
- **Adaptive Learning**: Adjusts content complexity based on grade level and performance
- **Document Analysis**: Processes and analyzes uploaded PDF documents
- **Interactive Q&A**: Allows students to ask questions about uploaded documents
- **Document-Based Quizzes**: Generates comprehension questions from uploaded materials
- **Performance Tracking**: Monitors and adapts to student progress

## Prerequisites

- Python 3.10
- Ollama installed and running locally (for LLM inference)
- Sufficient disk space for document storage and vector database
- Required Ollama models:
  - llama3.2:3b (for LLM inference)
  - nomic-embed-text (for text embeddings)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vijayganesh-07/AI-powered-personal-tutor.git
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install and start Ollama:
   - Follow the installation instructions at [Ollama's official website](https://ollama.ai/download)
   - Start the Ollama service:
```bash
ollama serve
```

5. Pull required Ollama models:
```bash
# Pull the LLM model for inference
ollama pull llama3.2:3b

# Pull the embedding model
ollama pull nomic-embed-text
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Follow the interface steps:
   - Select your grade level
   - Complete the knowledge assessment quiz
   - Upload and process your learning materials
   - Ask questions or take document-based quizzes

## How It Works

1. **Knowledge Assessment**:
   - Students select their grade level
   - Complete a grade-appropriate quiz
   - System adapts content based on performance

2. **Document Processing**:
   - Upload PDF documents
   - System processes and indexes content
   - Creates searchable vector database

3. **Interactive Learning**:
   - Ask questions about uploaded documents
   - Get personalized answers based on grade level
   - Generate comprehension quizzes from documents

4. **Adaptive System**:
   - Adjusts language complexity
   - Provides detailed explanations
   - Maintains appropriate academic rigor

## Technical Details

- **Vector Database**: ChromaDB for document storage and retrieval
- **LLM**: Local Ollama instance with llama3.2:3b model
- **Embeddings**: Ollama's nomic-embed-text model for text embeddings
- **Document Processing**: PyMuPDF for PDF handling
- **Re-ranking**: Sentence Transformers for improved answer relevance
- **UI Framework**: Streamlit for interactive interface

## Ollama Configuration

The application uses two main Ollama models:

1. **llama3.2:3b**:
   - Used for generating quiz questions
   - Providing adaptive responses
   - Creating document-based assessments
   - Minimum 8GB RAM recommended

2. **nomic-embed-text**:
   - Used for creating text embeddings
   - Enables semantic search functionality
   - Powers document retrieval
   - Lighter resource requirements

### Model Management

To update models:
```bash
# Update LLM model
ollama pull llama3.2:3b

# Update embedding model
ollama pull nomic-embed-text
```

To check model status:
```bash
ollama list
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ollama for providing local LLM inference
- ChromaDB for vector database functionality
- Streamlit for the web interface framework
- LangChain for document processing utilities


🚨 NOTE: **Requires `Python > 3.10` with  `SQLite > 3.35`**


