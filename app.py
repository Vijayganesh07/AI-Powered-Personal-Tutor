import os
import tempfile
import json
import datetime

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Initialize session state for quiz and documents
if 'quiz_state' not in st.session_state:
    st.session_state.quiz_state = {
        'grade': None,
        'current_question': 0,
        'score': 0,
        'total_questions': 5,
        'questions': [],
        'answers': [],
        'quiz_completed': False,
        'documents_processed': False,
        'doc_quiz_mode': False,  # Flag to indicate document-based quiz
        'knowledge_quiz_completed': False  # New flag for knowledge quiz completion
    }

# Create different system prompts based on grade level and score
def get_adaptive_system_prompt(grade: int, score: int, max_score: int = 5) -> str:
    """Returns a system prompt adapted to the user's grade and quiz performance."""
    # Calculate performance as a percentage
    performance = score / max_score

    # Base complexity on grade and performance
    if grade <= 3 or performance < 0.4:
        # Simple language for young children or poor performers, but still detailed
        return """
        You are an AI tutor for young or beginning students. When answering questions:
        1. Use simple words and clear sentences that a 6-8 year old would understand
        2. Explain concepts with everyday examples and familiar comparisons
        3. Avoid technical terms - if you must use them, explain them very simply
        4. Break down complex ideas into smaller, manageable pieces
        5. Use friendly, encouraging language
        6. Provide DETAILED explanations despite the simple language
        7. Include all relevant information from the context
        8. Use visual language and concrete examples
        
        Only use information from the context provided to answer the question.
        Focus on simplifying language and concepts, NOT reducing the thoroughness of your answer.
        """
    elif grade <= 6 or performance < 0.6:
        # Simple language for elementary students or below-average performers
        return """
        You are an AI tutor for elementary school students or learners needing simplified explanations. When answering questions:
        1. Use straightforward language with minimal technical terms
        2. Provide clear, DETAILED explanations with concrete examples
        3. Define any specialized vocabulary when you use it
        4. Use analogies to familiar concepts when introducing new ideas
        5. Maintain a supportive, encouraging tone
        6. Cover all important aspects of the topic thoroughly
        7. Organize information in a logical way with clear transitions
        8. Avoid complex sentences but don't oversimplify the content itself
        
        Only use information from the context provided to answer the question.
        Focus on accessible language while maintaining comprehensive content coverage.
        """
    elif grade <= 9 or performance < 0.8:
        # Moderate language for middle school students or average performers
        return """
        You are an AI tutor for middle school students or intermediate learners. When answering questions:
        1. Use clear, conversational language appropriate for teenagers
        2. Provide DETAILED explanations with specific examples
        3. Introduce appropriate technical terms with brief, clear definitions
        4. Use helpful analogies and visual descriptions
        5. Organize information with headings and bullet points when appropriate
        6. Connect new concepts to foundational knowledge
        7. Address nuances and exceptions when relevant
        8. Maintain a balance between technical accuracy and accessibility
        
        Only use information from the context provided to answer the question.
        Ensure your explanations are thorough while keeping the language accessible.
        """
    else:
        # Advanced language for high school students or high performers
        return """
        You are an AI tutor for advanced students. When answering questions:
        1. Use precise, academic language appropriate for the subject
        2. Provide DETAILED explanations with depth and nuance
        3. Use proper terminology and concepts appropriate to the field
        4. Include specific examples, counterexamples, and edge cases
        5. Discuss underlying principles and connections to broader concepts
        6. Organize information logically with clear structure
        7. Address potential misconceptions or alternative perspectives
        8. Maintain rigor and accuracy while ensuring clarity
        
        Only use information from the context provided to answer the question.
        Present comprehensive, sophisticated explanations while ensuring they remain clear and understandable.
"""


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded PDF file by converting it to text chunks.

    Takes an uploaded PDF file, saves it temporarily, loads and splits the content
    into text chunks using recursive character splitting.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing the PDF file

    Returns:
        A list of Document objects containing the chunked text from the PDF

    Raises:
        IOError: If there are issues reading/writing the temporary file
    """
    temp_file = None
    try:
        # Store uploaded file as a temp file
        temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()  # Close the file before using it

        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        return text_splitter.split_documents(docs)
    finally:
        # Clean up the temporary file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except PermissionError:
                # If we can't delete the file, just log it and continue
                st.warning("Could not delete temporary file. It will be cleaned up later.")


def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage."""
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a vector collection for semantic search."""
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Document processed and ready for questions!")


def query_collection(prompt: str, n_results: int = 10):
    """Queries the vector collection with a given prompt to retrieve relevant documents."""
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def call_llm(context: str, prompt: str, grade: int, score: int):
    """Calls the language model with context and prompt to generate a response.

    Adapts the response complexity based on grade and quiz score.
    """
    system_prompt = get_adaptive_system_prompt(grade, score)
    
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def re_rank_cross_encoders(query_text: str, documents: list[str]) -> tuple[str, list[int]]:
    """Re-ranks documents using a cross-encoder model."""
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(query_text, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids


def generate_document_based_quiz(max_retries: int = 3) -> list[dict]:
    """Generate quiz questions directly from the document content."""
    
    # Query the collection to get a representative sample of document content
    with st.spinner("Analyzing document content to create quiz..."):
        # Use generic queries to retrieve representative content
        topics_query = "What are the main topics in this document"
        results = query_collection(topics_query, n_results=15)
        
        if not results or len(results.get("documents", [[]])[0]) == 0:
            st.error("Unable to extract enough content from the document for a quiz.")
            return []
        
        document_samples = results.get("documents")[0]
        document_content = " ".join(document_samples[:10])  # Limit to avoid context length issues
        
        quiz_gen_prompt = f"""
        Based on the following document content, create 5 quiz questions that test understanding of key concepts.
        
        Document content:
        {document_content}
        
        Generate 5 multiple-choice questions in JSON format with 4 options per question.
        """
        
        gen_system_prompt = """You are an educational quiz generator. Generate exactly 5 questions based on the provided document content.
        Return your response in the following JSON format:
        [
            {
                "question": "What is the main topic discussed in the document?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Option B"
            },
            {
                "question": "According to the document, what is an important characteristic of X?",
                "options": ["Characteristic 1", "Characteristic 2", "Characteristic 3", "Characteristic 4"],
                "correct_answer": "Characteristic 2"
            }
        ]
        Make sure to:
        1. Generate exactly 5 questions that test understanding of the document content
        2. Each question must have exactly 4 options
        3. The correct_answer must match one of the options exactly
        4. Use proper JSON format with double quotes
        5. Include questions that test comprehension, not just fact recall
        6. DO NOT include any explanations, comments, or additional text before or after the JSON array
        7. Ensure all strings are properly escaped
        8. Use only valid JSON characters
        """
    
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Generating document quiz (attempt {attempt+1}/{max_retries})..."):
                response = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {
                            "role": "system",
                            "content": gen_system_prompt
                        },
                        {
                            "role": "user",
                            "content": quiz_gen_prompt
                        }
                    ]
                )
                
                # Clean the response to ensure it's valid JSON
                content = response['message']['content']
                # Remove any markdown code block markers if present
                content = content.replace('```json', '').replace('```', '').strip()
                
                # Try to find the JSON array in the response
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx]
                    try:
                        # Clean up the JSON content
                        json_content = json_content.replace('\n', ' ').replace('\r', '')
                        json_content = ' '.join(json_content.split())  # Remove extra whitespace
                        
                        questions = json.loads(json_content)
                        
                        # Validate the questions
                        if not isinstance(questions, list):
                            st.warning(f"Attempt {attempt+1}: LLM did not return a valid list. Response: {content[:200]}...")
                            continue
                            
                        if len(questions) < 3:  # Accept at least 3 questions if not 5
                            st.warning(f"Attempt {attempt+1}: LLM generated only {len(questions)} questions. Response: {content[:200]}...")
                            continue
                            
                        valid_questions = []
                        for q in questions:
                            try:
                                if not all(k in q for k in ['question', 'options', 'correct_answer']):
                                    st.warning(f"Attempt {attempt+1}: Question missing required fields: {q}")
                                    continue
                                if len(q['options']) != 4:
                                    st.warning(f"Attempt {attempt+1}: Question has wrong number of options: {q}")
                                    continue
                                if q['correct_answer'] not in q['options']:
                                    st.warning(f"Attempt {attempt+1}: Correct answer not in options: {q}")
                                    continue
                                valid_questions.append(q)
                            except Exception as e:
                                st.warning(f"Attempt {attempt+1}: Error validating question: {str(e)}")
                                continue
                        
                        if len(valid_questions) < 3:  # Accept at least 3 valid questions
                            st.warning(f"Attempt {attempt+1}: Only found {len(valid_questions)} valid questions. Response: {content[:200]}...")
                            continue
                        
                        max_questions = min(5, len(valid_questions))
                        st.success(f"Successfully generated {max_questions} document-based quiz questions!")
                        return valid_questions[:max_questions]
                    except json.JSONDecodeError as e:
                        st.warning(f"Attempt {attempt+1}: Error parsing JSON response: {str(e)}")
                        st.warning(f"Response content: {content[:200]}...")
                else:
                    st.warning(f"Attempt {attempt+1}: No valid JSON array found in LLM response. Response: {content[:200]}...")
        
        except Exception as e:
            st.warning(f"Attempt {attempt+1}: Error generating questions: {str(e)}")
    
    # If we get here, we've exhausted our retries
    st.error("Failed to generate document quiz questions after multiple attempts.")
    return []


def generate_grade_appropriate_questions(grade: int, max_retries: int = 3) -> list[dict]:
    """Generate questions appropriate for the student's grade level using LLM."""
    grade_prompts = {
        1: "Generate 5 simple questions for grade 1 students about basic math, reading, and science. Make them very simple and easy to understand.",
        2: "Generate 5 questions for grade 2 students about basic math, reading comprehension, and simple science concepts.",
        3: "Generate 5 questions for grade 3 students about multiplication, reading comprehension, and basic science.",
        4: "Generate 5 questions for grade 4 students about division, reading comprehension, and science concepts.",
        5: "Generate 5 questions for grade 5 students about fractions, reading comprehension, and science topics.",
        6: "Generate 5 questions for grade 6 students about algebra basics, reading comprehension, and science concepts.",
        7: "Generate 5 questions for grade 7 students about pre-algebra, reading comprehension, and science topics.",
        8: "Generate 5 questions for grade 8 students about algebra, reading comprehension, and science concepts.",
        9: "Generate 5 questions for grade 9 students about algebra, geometry basics, and science topics.",
        10: "Generate 5 questions for grade 10 students about geometry, algebra, and science concepts.",
        11: "Generate 5 questions for grade 11 students about advanced algebra, trigonometry, and science topics.",
        12: "Generate 5 questions for grade 12 students about calculus, advanced math, and science concepts."
    }
    
    question_prompt = grade_prompts.get(grade, "Generate 5 general educational questions.")
    
    gen_system_prompt = """You are an educational quiz generator. Generate exactly 5 questions in the following JSON format:
    [
        {
            "question": "What is 2 + 2?",
            "options": ["3", "4", "5", "6"],
            "correct_answer": "4"
        },
        {
            "question": "Which planet is closest to the Sun?",
            "options": ["Earth", "Mars", "Venus", "Mercury"],
            "correct_answer": "Mercury"
        }
    ]
    Make sure to:
    1. Generate exactly 5 questions
    2. Each question must have exactly 4 options
    3. The correct_answer must match one of the options exactly
    4. Use proper JSON format with double quotes
    5. Include a mix of math, science, and reading questions appropriate for the grade level
    6. DO NOT include any explanations, comments, or additional text before or after the JSON array
    """
    
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Generating quiz questions (attempt {attempt+1}/{max_retries})..."):
                response = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {
                            "role": "system",
                            "content": gen_system_prompt
                        },
                        {
                            "role": "user",
                            "content": question_prompt
                        }
                    ]
                )
                
                # Clean the response to ensure it's valid JSON
                content = response['message']['content']
                # Remove any markdown code block markers if present
                content = content.replace('```json', '').replace('```', '').strip()
                
                # Try to find the JSON array in the response
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx]
                    questions = json.loads(json_content)
                    
                    # Validate the questions
                    if not isinstance(questions, list):
                        st.warning(f"Attempt {attempt+1}: LLM did not return a valid list. Retrying...")
                        continue
                        
                    if len(questions) != 5:
                        st.warning(f"Attempt {attempt+1}: LLM generated {len(questions)} questions instead of 5. Retrying...")
                        continue
                        
                    valid_questions = []
                    for q in questions:
                        if not all(k in q for k in ['question', 'options', 'correct_answer']):
                            continue
                        if len(q['options']) != 4:
                            continue
                        if q['correct_answer'] not in q['options']:
                            continue
                        valid_questions.append(q)
                    
                    if len(valid_questions) < 5:
                        st.warning(f"Attempt {attempt+1}: Only found {len(valid_questions)} valid questions. Retrying...")
                        continue
                        
                    st.success("Successfully generated quiz questions!")
                    return valid_questions[:5]  # Take the first 5 valid questions
                else:
                    st.warning(f"Attempt {attempt+1}: No valid JSON array found in LLM response. Retrying...")
            
        except Exception as e:
            st.warning(f"Attempt {attempt+1}: Error in question generation: {str(e)}. Retrying...")
    
    # If we get here, we've exhausted our retries
    st.error("Failed to generate valid questions after multiple attempts. Please try again later.")
    
    # Create minimal default questions as a last resort
    return [
        {
            "question": f"What grade level are you in?", 
            "options": [f"{grade-1}", f"{grade}", f"{grade+1}", f"{grade+2}"],
            "correct_answer": f"{grade}"
        },
        {
            "question": "Which subject do you find most interesting?", 
            "options": ["Math", "Science", "Reading", "History"],
            "correct_answer": "Math"  # This is arbitrary
        },
        {
            "question": "How comfortable are you with using technology?", 
            "options": ["Very comfortable", "Somewhat comfortable", "Not very comfortable", "Not comfortable at all"],
            "correct_answer": "Very comfortable"  # This is arbitrary
        },
        {
            "question": "How do you prefer to learn new things?", 
            "options": ["Reading", "Watching videos", "Hands-on activities", "Listening to explanations"],
            "correct_answer": "Hands-on activities"  # This is arbitrary
        },
        {
            "question": "Which of these skills would you like to improve the most?", 
            "options": ["Problem solving", "Memorization", "Critical thinking", "Creative thinking"],
            "correct_answer": "Problem solving"  # This is arbitrary
        }
    ]


def assess_performance(grade: int, score: int) -> str:
    """Generate a performance assessment based on grade and score."""
    try:
        assessment_prompt = f"""
        Generate a detailed assessment for a grade {grade} student who scored {score}/5.
        The assessment should:
        1. Be appropriate for their grade level
        2. Include specific strengths and areas for improvement
        3. Provide constructive feedback
        4. Use language appropriate for their grade level
        5. Include specific suggestions for improvement
        """
        
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "system",
                    "content": "You are an educational assessment expert. Provide detailed, grade-appropriate feedback."
                },
                {
                    "role": "user",
                    "content": assessment_prompt
                }
            ]
        )
        
        return response['message']['content']
    except Exception as e:
        # Fallback assessment if generation fails
        if score >= 4:
            return f"Great job! You scored {score}/5, which is excellent for a grade {grade} student. Keep up the good work!"
        elif score >= 2:
            return f"Good effort! You scored {score}/5. With a bit more practice, you'll master these concepts."
        else:
            return f"You scored {score}/5. Don't worry! Learning takes time and practice. Let's try again soon."


def main():
    st.title("📚 AI-Powered Personal Tutor")
    
    # Initialize session state for quiz and documents if not exists
    if 'quiz_state' not in st.session_state:
        st.session_state.quiz_state = {
            'grade': None,
            'current_question': 0,
            'score': 0,
            'total_questions': 5,
            'questions': [],
            'answers': [],
            'quiz_completed': False,
            'documents_processed': False,
            'doc_quiz_mode': False,
            'knowledge_quiz_completed': False
        }
    
    # Knowledge Assessment Quiz
    if not st.session_state.quiz_state['knowledge_quiz_completed']:
        # Grade Selection
        if not st.session_state.quiz_state['grade']:
            st.markdown("### Welcome to Your Personal AI Tutor!")
            st.markdown("Let's start by understanding your current knowledge level to provide the best learning experience.")
            
            # Create a container for better alignment
            with st.container():
                grade = st.selectbox("Select your grade level:", range(1, 13))
                if st.button("Begin Assessment", type="primary"):
                    st.session_state.quiz_state['grade'] = grade
                    questions = generate_grade_appropriate_questions(grade)
                    st.session_state.quiz_state['questions'] = questions
                    st.rerun()
        
        # Quiz Interface
        elif not st.session_state.quiz_state['quiz_completed'] and st.session_state.quiz_state['questions']:
            current_question = st.session_state.quiz_state['current_question']
            questions = st.session_state.quiz_state['questions']
            
            progress = (current_question + 1) / len(questions)
            st.progress(progress)
            
            st.markdown(f"### Question {current_question + 1} of {len(questions)}")
            st.markdown(f"**{questions[current_question]['question']}**")
            
            answer = st.radio("Select your answer:", questions[current_question]['options'])
            
            if st.button("Submit Answer", type="primary"):
                if answer == questions[current_question]['correct_answer']:
                    st.session_state.quiz_state['score'] += 1
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Wrong! The correct answer was: {questions[current_question]['correct_answer']}")
                
                # Store the answer feedback in session state
                st.session_state.quiz_state['show_feedback'] = True
                st.session_state.quiz_state['current_answer'] = answer
                st.rerun()
            
            # Show feedback and Next Question button if answer was submitted
            if st.session_state.quiz_state.get('show_feedback', False):
                if st.session_state.quiz_state['current_answer'] == questions[current_question]['correct_answer']:
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Wrong! The correct answer was: {questions[current_question]['correct_answer']}")
                
                if st.button("Next Question", type="primary"):
                    st.session_state.quiz_state['current_question'] += 1
                    st.session_state.quiz_state['show_feedback'] = False
                    if st.session_state.quiz_state['current_question'] >= len(questions):
                        st.session_state.quiz_state['quiz_completed'] = True
                        st.session_state.quiz_state['knowledge_quiz_completed'] = True
                    st.rerun()
        
        # Quiz Results
        elif st.session_state.quiz_state['quiz_completed']:
            score = st.session_state.quiz_state['score']
            grade = st.session_state.quiz_state['grade']
            total_questions = len(st.session_state.quiz_state['questions'])
            
            st.markdown("### 🎉 Assessment Complete!")
            st.markdown(f"**Grade Level:** {grade} | **Score:** {score}/{total_questions}")
            
            if st.button("Continue to Learning", type="primary"):
                st.rerun()
    
    # Document Learning Interface
    else:
        # Document Upload Area
        with st.sidebar:
            st.markdown("### 📚 Learning Materials")
            uploaded_file = st.file_uploader(
                "Upload your PDF document", type=["pdf"], accept_multiple_files=False
            )

            if uploaded_file:
                if st.button("Process Document", type="primary"):
                    with st.spinner("Processing your document..."):
                        normalize_uploaded_file_name = uploaded_file.name.translate(
                            str.maketrans({"-": "_", ".": "_", " ": "_"})
                        )
                        all_splits = process_document(uploaded_file)
                        add_to_vector_collection(all_splits, normalize_uploaded_file_name)
                        st.session_state.quiz_state['documents_processed'] = True
                        st.success("Document processed successfully!")
            
            if st.session_state.quiz_state['documents_processed']:
                if st.button("📝 Generate Quiz from Document"):
                    st.session_state.quiz_state['doc_quiz_mode'] = True
                    st.session_state.quiz_state['current_question'] = 0
                    st.session_state.quiz_state['score'] = 0
                    
                    doc_questions = generate_document_based_quiz()
                    if doc_questions:
                        st.session_state.quiz_state['questions'] = doc_questions
                        st.rerun()
                    else:
                        st.error("Failed to generate quiz. Please try again.")
        
        # Document Quiz Interface
        if st.session_state.quiz_state['doc_quiz_mode'] and st.session_state.quiz_state['questions']:
            current_question = st.session_state.quiz_state['current_question']
            questions = st.session_state.quiz_state['questions']
            
            progress = (current_question + 1) / len(questions)
            st.progress(progress)
            
            st.markdown("### 📝 Document Comprehension Quiz")
            st.markdown(f"**Question {current_question + 1} of {len(questions)}**")
            st.markdown(f"**{questions[current_question]['question']}**")
            
            answer = st.radio("Select your answer:", questions[current_question]['options'])
            
            if st.button("Submit Answer", type="primary"):
                if answer == questions[current_question]['correct_answer']:
                    st.session_state.quiz_state['score'] += 1
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Wrong! The correct answer was: {questions[current_question]['correct_answer']}")
                
                # Store the answer feedback in session state
                st.session_state.quiz_state['show_feedback'] = True
                st.session_state.quiz_state['current_answer'] = answer
                st.rerun()
            
            # Show feedback and Next Question button if answer was submitted
            if st.session_state.quiz_state.get('show_feedback', False):
                if st.session_state.quiz_state['current_answer'] == questions[current_question]['correct_answer']:
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Wrong! The correct answer was: {questions[current_question]['correct_answer']}")
                
                if st.button("Next Question", type="primary"):
                    st.session_state.quiz_state['current_question'] += 1
                    st.session_state.quiz_state['show_feedback'] = False
                    if st.session_state.quiz_state['current_question'] >= len(questions):
                        st.session_state.quiz_state['doc_quiz_mode'] = False
                        st.success(f"🎉 Quiz Complete! Your score: {st.session_state.quiz_state['score']}/{len(questions)}")
                        if st.button("Back to Document Learning"):
                            st.rerun()
                    st.rerun()
        
        # Document Q&A Interface (only show if not in quiz mode)
        elif not st.session_state.quiz_state['doc_quiz_mode']:
            if not st.session_state.quiz_state['documents_processed']:
                st.markdown("### Welcome to Your Learning Session!")
                st.markdown("Upload a document to begin exploring and learning.")
                return
            
            st.markdown("### Ask Questions About Your Document")
            st.markdown("Your answers will be tailored to your knowledge level.")
            
            user_question = st.text_area("What would you like to know?", placeholder="Type your question here...")
            
            if st.button("Find Answer", type="primary"):
                if user_question:
                    with st.spinner("Searching for information..."):
                        results = query_collection(user_question)
                        
                        if not results or len(results.get("documents", [[]])[0]) == 0:
                            st.error("Sorry, I couldn't find relevant information in the document.")
                        else:
                            context = results.get("documents")[0]
                            relevant_text, relevant_text_ids = re_rank_cross_encoders(user_question, context)
                            
                            st.markdown("### Answer")
                            response = call_llm(
                                context=relevant_text, 
                                prompt=user_question, 
                                grade=st.session_state.quiz_state['grade'], 
                                score=st.session_state.quiz_state['score']
                            )
                            st.write_stream(response)

                            with st.expander("View Source Information"):
                                st.markdown("**Relevant sections from your document:**")
                                for i, text_id in enumerate(relevant_text_ids):
                                    st.markdown(f"**Passage {i+1}:**")
                                    st.write(context[text_id])


if __name__ == "__main__":
    main()
