# Gen AI Customer Service Chatbot

## Overview

This project is an advanced AI-powered Customer Service Chatbot developed as part of the ElevanceSkills AI Internship Program. The chatbot is designed to provide intelligent customer support using Natural Language Processing (NLP), Generative AI, multilingual communication, sentiment analysis, and knowledge retrieval techniques.

## Features

### 1. Sentiment Analysis

* Detects customer emotions from messages.
* Classifies sentiment as Positive, Negative, or Neutral.
* Generates context-aware responses based on customer mood.

### 2. Dynamic Knowledge Base

* Retrieves information from a vector database.
* Supports continuous knowledge expansion and updates.
* Improves response quality using stored information.

### 3. Multilingual Support

* Supports multiple languages.
* Automatically detects user language.
* Provides responses in the user's preferred language.

### 4. Multimodal AI Support

* Handles both text and image-based interactions.
* Enables richer customer support experiences.

### 5. Research Expert Module

* Uses scientific research and document retrieval techniques.
* Provides summarized explanations for complex topics.
* Supports educational and technical query handling.

## Project Structure

```text
customer_service_chatbot
│
├── app.py
├── config.py
├── requirements.txt
├── data/
│   └── chroma_db/
├── modules/
│   ├── sentiment.py
│   ├── knowledge_base.py
│   ├── multilingual.py
│   ├── multimodal.py
│   └── arxiv_expert.py
│
└── .streamlit/
    └── config.toml
```

## Technologies Used

* Python
* Streamlit
* Natural Language Processing (NLP)
* ChromaDB
* Hugging Face Transformers
* Generative AI Models
* Machine Learning
* Vector Databases

## Installation

```bash
git clone https://github.com/shivamschandel/gen-ai-customer-service-chatbot.git
cd gen-ai-customer-service-chatbot
pip install -r requirements.txt
streamlit run app.py
```

## Internship Tasks Covered

* Sentiment Analysis Integration
* Dynamic Knowledge Base Expansion
* Domain Expert Chatbot Development
* Multilingual Chatbot Support
* Multimodal AI Chatbot Features

## Future Enhancements

* Advanced Medical Question Answering
* Real-Time Knowledge Updates
* Voice Interaction Support
* Improved Retrieval-Augmented Generation (RAG)

## Internship Information

This project was developed as part of the ElevanceSkills Artificial Intelligence Internship Program. The objective was to build an advanced customer service chatbot capable of handling real-world user interactions using modern AI and NLP techniques.

## Author

**Shivam Singh**
BCA Student | AI & Data Science Enthusiast
