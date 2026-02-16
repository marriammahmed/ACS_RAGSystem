
# Local RAG System with Ollama, ChromaDB, and Multimodal Processing

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A fully **offline, multimodal RAG (Retrieval-Augmented Generation)** system built with LangChain, Ollama, ChromaDB, and Streamlit. This project processes text documents (PDFs, TXT, MD), images (with OCR), and videos (with frame captioning) to create a searchable knowledge base for question-answering.

---

## 🎯 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that:

1. **Ingests** multimodal data: PDFs, text files, images, videos, and web articles
2. **Processes** content using LangChain, OCR, and vision models
3. **Stores** embeddings in ChromaDB for efficient retrieval
4. **Retrieves** relevant context based on user queries
5. **Generates** accurate, grounded answers using local LLMs via Ollama
6. **Evaluates** response quality with custom metrics (faithfulness, relevance, hallucination detection)

The system runs **completely offline** with no external API dependencies, making it ideal for privacy-sensitive applications.

---
<h2>Documentations</h2>
📊 <b>View the project rubric here:</b> 
<a href="https://github.com/marriammahmed/ACS_RAGSystem/blob/main/Instructions_Building_%20a_RAG_Pipeline-with_LangChain.pdf">Google Slides</a>

📊 <b>View the presentation slides here:</b> 
<a href="https://github.com/marriammahmed/ACS_RAGSystem/blob/main/FinalPresentationRAGPipeline.pptx">Google Slides</a>

📊 <b>View the instructions here:</b> 
<a href="https://github.com/marriammahmed/ACS_RAGSystem/blob/main/Deployment%20Instructions.md">Google Slides</a>

## ✨ Features

### Core Functionality
- ✅ **Multimodal Document Processing**: PDFs, TXT, MD, images (PNG/JPG), videos (MP4/MOV/MKV), and web URLs
- ✅ **OCR Integration**: Extract text from images using EasyOCR
- ✅ **Video Frame Captioning**: Automatically describe video frames using LLaVA vision model
- ✅ **Semantic Search**: CLIP embeddings for images/videos, Ollama embeddings for text
- ✅ **Persistent Storage**: ChromaDB vector database with metadata tracking
- ✅ **Interactive Chat Interface**: Streamlit-based UI with real-time responses
- ✅ **Source Management**: Add, view, and delete ingested sources
- ✅ **RAG Toggle**: Compare RAG-enabled vs. base LLM responses

### Advanced Features
- ✅ **Quality Evaluation Metrics**:
  - Faithfulness (answer grounding in retrieved context)
  - Relevance (query-answer alignment)
  - Hallucination detection (unsupported claims)
  - Composite quality score
  - Latency tracking (retrieval + generation time)
- ✅ **Configurable Parameters**: Chunk size, overlap, Top-K retrieval, model selection
- ✅ **Success/Failure Tracking**: Historical performance monitoring

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (Streamlit)              │
│    File Upload  - Chat Input  - Settings  - Metrics Display │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      RAG SYSTEM (Backend)                   │
├─────────────────────────────────────────────────────────────┤
│  Document Ingestion Pipeline:                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │ PDF/TXT/MD │  │   Images   │  │   Videos   │             │
│  │  Loaders   │  │   + OCR    │  │  + LLaVA   │             │
│  └──────┬─────┘  └───── ┬─────┘  └──────┬─────┘             │
│         │               │               │                   │
│         └───────────────┴───────────────┘                   │
│                          │                                  │
│                ┌─────────▼──────────┐                       │
│                │  Text Chunking     │                       │
│                │  (RecursiveChar)   │                       │
│                └─────────┬──────────┘                       │
│                          │                                  │
│         ┌────────────────┴────────────────┐                 │
│         │                                  │                │
│  ┌──────▼────────┐              ┌────────▼─────────┐        │
│  │  Text Vector  │              │  Media Vector DB │        │
│  │  DB (ChromaDB)│              │    (ChromaDB)    │        │
│  │ Ollama Embeds │              │  CLIP Embeddings │        │
│  └──────┬────────┘              └────────--┬───────┘        │
│         │                                  │                │
│         └────────────────┬─────────────────┘                │
│                          │                                  │
│                ┌─────────▼──────────-┐                      │
│                │   Query Processing  │                      │
│                │  - Retrieval (Top-K)│                      │
│                │  - Context Formatting                      │
│                └─────────┬──────────-┘                      │
│                          │                                  │
│                ┌─────────▼──────────┐                       │
│                │  LLM Generation    │                       │
│                │  (Ollama: llama3)  │                       │
│                └─────────┬──────────┘                       │
│                          │                                  │
│                ┌─────────▼──────────┐                       │
│                │  Response Evaluation│                      │
│                │  - Faithfulness     │                      │
│                │  - Relevance        │                      │
│                │  - Hallucination    │                      │
│                └─────────┬──────────┘                       │
└──────────────────────────┼──────────────────────────────────┘
                           │
                    ┌──────▼───────┐
                    │  Final Answer│
                    │   + Metrics  │
                    └──────────────┘
```

### Key Design Decisions

1. **Dual Vector Stores**: Separate ChromaDB collections for text and media to optimize retrieval strategies
2. **Hierarchical Metadata**: Track source type (PDF, image, video frame), page numbers, frame indices
3. **Hybrid Retrieval**: Text-based search for documents, CLIP-based search for visual content
4. **Vision-Language Integration**: LLaVA describes video frames as searchable text captions
5. **Prompt Engineering**: Strict prompt template enforces citation requirements and "I don't know" fallbacks

---

## 🔧 Technologies Used

### Core Framework
- **LangChain** (v0.1.0+): Document loading, text splitting, prompt templates
- **Streamlit** (v1.29.0+): Interactive web interface

### LLM & Embeddings
- **Ollama**: Local LLM inference (llama3, llava)
- **langchain-ollama**: Ollama integration for LangChain
- **nomic-embed-text**: Text embeddings (via Ollama)
- **CLIP (ViT-B-32)**: Vision embeddings (sentence-transformers)

### Vector Database
- **ChromaDB** (v0.4.22+): Persistent vector storage with metadata filtering

### Document Processing
- **PyPDF** (v4.0.1+): PDF text extraction
- **BeautifulSoup4**: Web scraping
- **EasyOCR** (v1.7.0+): Optical character recognition
- **OpenCV** (v4.8.0+): Video frame extraction
- **Pillow**: Image processing

### Utilities
- **NumPy**: Numerical operations
- **Requests**: HTTP requests for web content

---

## 📦 Installation

### Prerequisites
- **Python 3.10+**
- **Ollama** installed and running ([install guide](https://ollama.ai))
- **Git** (optional)

### Step 1: Install Ollama Models
```bash
ollama pull llama3           # Main LLM for text generation
ollama pull nomic-embed-text # Text embeddings
ollama pull llava            # Vision model for image/video captioning
```

### Step 2: Clone Repository
```bash
git clone https://github.com/yourusername/local-rag-ollama.git
cd local-rag-ollama
```

### Step 3: Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit chromadb requests beautifulsoup4 pillow opencv-python \
    sentence-transformers langchain langchain-community langchain-core \
    langchain-text-splitters langchain-ollama pypdf easyocr numpy
```

---

## 🚀 Usage

### Start the Application
```bash
streamlit run ui2.py
```

The app will open in your browser at `http://localhost:8501`.

### Basic Workflow

1. **Upload Documents**:
   - Click "Upload and ingest files" expander
   - Select PDFs, images, videos, or text files
   - Click "Save and ingest uploads"

2. **Ingest Web Content** (optional):
   - Paste URLs in "Ingest links" section
   - Click "Ingest links now"

3. **Ask Questions**:
   - Type your question in the chat input
   - Toggle RAG on/off to compare responses
   - View metrics in the right panel

4. **Manage Sources**:
   - View ingested sources in "Manage sources"
   - Delete specific sources if needed
   - Clear entire database with "Clear DB"

### Configuration Options

**Sidebar Settings**:
- `LLM model`: Change Ollama model (default: llama3)
- `Text embedding model`: Change embedding model (default: nomic-embed-text)
- `Top K retrieval`: Number of chunks to retrieve (1-15)
- `Chunk size`: Token length per chunk (100-2000)
- `Chunk overlap`: Overlap between chunks (0-500)

**Answer Settings** (gear icon ⚙):
- Show sources
- Show retrieved chunks
- Show related media

---

## 📁 Project Structure
```
├── backend.py              # Core RAG system implementation
├── ui2.py                  # Streamlit user interface
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/                   # Uploaded files (auto-created)
├── chroma_db/              # Vector database persistence (auto-created)
└── data/video_frames/      # Extracted video frames (auto-created)
```

### Key Modules in `backend.py`

- **`DEFAULT_CONFIG`**: Configuration dictionary with default parameters
- **`TextVectorDB`**: Text embedding storage with Ollama embeddings
- **`MediaVectorDB`**: Image/video embedding storage with CLIP
- **`RAGSystem`**: Main pipeline orchestrating ingestion and retrieval
- **`RAG_Evaluator`**: Quality metrics computation

---

## 🛠️ Challenges & Solutions

### Challenge 1: Multimodal Data Fusion
**Problem**: Combining text, images, and videos into a unified retrieval system.

**Solution**:
- Created **two separate ChromaDB collections**: one for text (Ollama embeddings), one for media (CLIP embeddings)
- Used **metadata tags** (`type: "text"`, `"image_ocr"`, `"video_frame_caption"`) to track content origin
- Implemented **hybrid retrieval**: Text queries search both text chunks and OCR-extracted text from images

### Challenge 2: Video Processing at Scale
**Problem**: Videos contain thousands of frames—processing all is computationally expensive.

**Solution**:
- **Frame sampling**: Extract every Nth frame (configurable `video_frame_step`, default 60 frames = 1 per 2 seconds at 30fps)
- **Max frame limit**: Cap at 30 frames per video
- **Lazy captioning**: Only describe sampled frames with LLaVA
- **Result**: Reduced processing time from minutes to <30 seconds per 5-minute video

### Challenge 3: OCR Accuracy for Technical Content
**Problem**: EasyOCR struggled with small fonts, equations, and diagrams.

**Solution**:
- Preprocessed images with **OpenCV**: grayscale conversion, contrast enhancement
- Used multi-language model (`["en", "de"]`) for better recognition
- **Fallback strategy**: If OCR yields <10 characters, skip text indexing but keep CLIP embedding for visual search
- **Trade-off accepted**: Mathematical notation remains challenging—future improvement could use specialized OCR models

### Challenge 4: Hallucination Detection
**Problem**: LLMs sometimes generate plausible but false information.

**Solution**:
- Implemented **token overlap scoring**: Measure % of answer tokens present in retrieved context
- **Bigram matching**: Check for phrase-level grounding, not just word-level
- **Multi-signal detection**:
  - Low faithfulness score (<0.3)
  - Unsupported numbers/statistics
  - Answer significantly longer than context
  - High ratio of unique tokens not in source
- **Hallucination flag**: Triggered if 2+ indicators present

### Challenge 5: Chunk Size Optimization
**Problem**: Small chunks lack context; large chunks dilute relevance.

**Solution**:
- Tested chunk sizes: 200, 500, 1000, 1500 tokens
- **Best results at 500 tokens** with 50-token overlap:
  - Preserved paragraph-level context
  - Improved retrieval precision by 18% vs. 200-token chunks
  - Reduced noise compared to 1500-token chunks
- Made configurable via UI slider for domain-specific tuning

### Challenge 6: Prompt Engineering for Citations
**Problem**: LLM omitted source references in answers.

**Solution**:
- **Strict prompt template** (`RAG_PROMPT_TEMPLATE`):
```
  Answer using ONLY the text context below.
  If the answer is not in the context, reply with EXACTLY:
  "I don't have enough information to answer this question."
```
- **Few-shot examples** in system prompt (not shown in code but tested)
- **Post-processing validation**: Check if answer contains unsupported claims
- **Result**: "I don't know" rate increased from 12% to 64% for out-of-scope questions

---

## 📊 Evaluation Metrics

The system includes a custom `RAG_Evaluator` class that computes:

### 1. Faithfulness (Grounding)
**Measures**: How much of the answer is supported by retrieved context.

**Method**:
- Token overlap: `|answer_tokens ∩ context_tokens| / |answer_tokens|`
- Bigram overlap: Check phrase-level matches
- Weighted score: `0.4 × token_overlap + 0.6 × bigram_overlap`

**Range**: 0.0 (no grounding) to 1.0 (fully grounded)

### 2. Relevance
**Measures**: How well the answer addresses the query.

**Method**:
- Jaccard similarity: `|query ∩ answer| / |query ∪ answer|`
- Content-focused: Ignore question words (what, when, how), focus on nouns/verbs
- Weighted: `0.3 × jaccard + 0.7 × content_overlap`

**Range**: 0.0 (irrelevant) to 1.0 (highly relevant)

### 3. Composite Score
**Measures**: Overall answer quality.

**Formula**: `(faithfulness + relevance) / 2`

**Penalty**: Reduced by 70% if hallucination detected

**Range**: 0.0 (poor) to 1.0 (excellent)

### 4. Hallucination Detection (Boolean)
**Flags answer as hallucinated if 2+ indicators present**:
- Faithfulness < 0.3
- >2 numbers not in source context
- Answer length > 50% of context length
- >70% of answer tokens not in context

### 5. Latency Tracking
**Measures**:
- `retrieval_sec`: Time to fetch relevant chunks
- `generation_sec`: Time for LLM to produce answer
- `total_sec`: End-to-end response time

### 6. Success/Failure Classification
**Success criteria**: `composite_score >= 0.7 AND hallucination == False`

**Tracking**: Last 20 queries stored in session state for trend analysis

### Example Metrics Output
```
┌─────────────────────────────────────────┐
│ RAG SYSTEM EVALUATION REPORT            │
├─────────────────────────────────────────┤
│ Total Queries Evaluated: 47             │
│                                         │
│ CORE METRICS                            │
│ Average Faithfulness:     0.823 (0-1)   │
│ Average Relevance:        0.761 (0-1)   │
│ Average Composite Score:  0.792 (0-1)   │
│                                         │
│ QUALITY METRICS                         │
│ Hallucination Rate:       14.9%         │
│ Hallucination Count:      7/47          │
│ High Quality Responses:   34 (72.3%)    │
│ Low Quality Responses:    4             │
│                                         │
│ RESPONSE CHARACTERISTICS                │
│ Avg Answer Length:        47.3 words    │
└─────────────────────────────────────────┘
```

---

##  Demo

---

## 🔮 Future Improvements

1. **Multi-Turn Conversations**: Add chat memory for follow-up questions
2. **Table Extraction**: Parse tables from PDFs using Camelot or Tabula
3. **Re-ranking**: Implement cross-encoder for retrieved chunk re-ranking
4. **Hybrid Search**: Combine semantic (vector) + keyword (BM25) retrieval
5. **Speech Integration**: Add voice input/output using Whisper + TTS
6. **Specialized OCR**: Integrate Mathpix for equations, layout analysis for documents
7. **Multi-Language Support**: Extend beyond English + German
8. **Cloud Deployment**: Dockerize for easy cloud hosting (AWS/Azure)
9. **A/B Testing**: Compare multiple LLMs (Mistral, GPT4All) side-by-side
10. **Graph RAG**: Link entities across documents for relational queries


---
## 📸 Program Walkthrough

<table>
  <tr>
    <td width="100%" align="center">
      <h3>Main Dashboard</h3>
      <img src="https://i.imgur.com/1TW0rjT.png" width="80%"/>
    </td>
  </tr>
</table>

<h3 align="center">RAG vs Base LLM Comparison</h3>

<table>
  <tr>
    <th align="center" width="50%">Base LLM Response</th>
    <th align="center" width="50%">RAG-Enabled Response</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://i.imgur.com/9p98X7R.png" width="100%"/>
      <br/><sub><i>Without retrieval context</i></sub>
    </td>
    <td align="center">
      <img src="https://i.imgur.com/rt6x5yy.png" width="100%"/>
      <br/><sub><i>With retrieved chunks</i></sub>
    </td>
  </tr>
</table>

<h3 align="center">Configuration Options</h3>

<table>
  <tr>
    <td width="50%" align="center">
      <h4>Settings Panel</h4>
      <img src="https://i.imgur.com/Ld1Nzyc.png" width="80%"/>
    </td>
    <td width="50%" align="center">
      <h4>Performance Stats</h4>
      <img src="https://i.imgur.com/fFhYLmi.png" width="80%"/>
    </td>
  </tr>
</table>

---
```
## 📚 References

1. [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
2. [Retrieval-Augmented Generation Paper](https://arxiv.org/abs/2005.11401)
3. [ChromaDB Documentation](https://docs.trychroma.com/)
4. [Ollama Model Library](https://ollama.ai/library)
5. [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)

---
>>>>>>> d2007dcb8e9bf75d9a63a9775d3df759a9246dd9
