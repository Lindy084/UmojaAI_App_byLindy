import os
import base64
import requests
import streamlit as st
from dotenv import load_dotenv
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

# ------------------------
# Setup
# ------------------------
load_dotenv()
st.set_page_config(
    page_title="UmojaAI · Streamlit",
    page_icon="images/logo.png",  # path to your logo
    layout="wide"
)

st.sidebar.image("images/logo.png", width=180)
st.set_page_config(page_title="UmojaAI · Streamlit", page_icon="🤖", layout="wide")

# ------------------------
# API Configuration
# ------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_openrouter(model: str, messages: list, **kwargs):
    """Call the OpenRouter API safely"""
    if not OPENROUTER_API_KEY:
        raise ValueError("❌ API Key not found. Please set OPENROUTER_API_KEY in your Streamlit secrets.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }
    payload = {"model": model, "messages": messages}
    payload.update(kwargs or {})
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    content = None
    if isinstance(data, dict):
        choices = data.get("choices") or []
        if choices and "message" in choices[0]:
            content = choices[0]["message"].get("content")
    return content, data

# ------------------------
# UI
# ------------------------
UI = {
    "en": {
        "title": "Welcome to UmojaAI_App By Lindiwe Ndlazi",
        "tagline": "UmojaAI provides intelligent, accessible AI tools designed to empower individuals to reach their full potential, educate communities with clear and practical knowledge, and foster a truly inclusive environment where people from all backgrounds can learn, create, and thrive together.",
        "tabs": ["🤖 Chatbot", "📝 Summarizer", "🌐 Translator", "🖼️ Image Captioning", "📚 AI Knowledge Quiz", "🌍 Story", "📞 Contact"],
        "contact": {
            "email": "ndlazilindy22@gmail.com",
            "phone": "+27 617150448",
            "socials": {
                "linkedin": "https://www.linkedin.com/in/ndlazi-lindiwe-76baa6229"
            }
        }
    },
    "zu": {
        "title": "Siyakwamukela ku-UmojaAI",
        "tagline": "Amathuluzi e-AI aqinisa, afundisa, futhi afaka bonke.",
        "tabs": ["🤖 Ingxoxo", "📝 Isifinyezo", "🌐 Umhumushi", "🖼️ Ukuchaza Izithombe", "📝 Imibuzo yokuhlola", "🌍 Indaba", "📞 Xhumana Nathi"],
        "contact": {
            "email": "ndlazilindy22@gmail.com",
            "phone": "+27 617150448",
            "socials": {
                "linkedin": "https://www.linkedin.com/in/ndlazi-lindiwe-76baa6229"
            }
        }
    },
    "isiXhosa": {
        "title": "Wamkelekile e-UmojaAI",
        "tagline": "Izixhobo ze-AI ezinamandla, ezifundisayo nezibandakanya wonke umntu.",
        "tabs": ["🤖 Ingxoxo", "📝 Isifinyezo", "🌐 Umhumushi", "🖼️ Ukuchaza Izithombe", "📝 Imibuzo yokuhlola", "🌍 Indaba", "📞 Xhumana Nathi"],
        "contact": {
            "email": "ndlazilindy22@gmail.com",
            "phone": "+27 617150448",
            "socials": {
                "linkedin": "https://www.linkedin.com/in/ndlazi-lindiwe-76baa6229"
            }
        }
    },
    "seSotho": {
        "title": "Rea u amohela ho UmojaAI",
        "tagline": "Lisebelisoa tsa AI tse matlafatsang, tse rutang le tse kenyelletsang bohle.",
        "tabs": ["🤖 Puisano", "📝 Kakaretso", "🌐 Phetolelo", "🖼️ Tlhaloso ea Setšoantšo", "📝 Dipotso tsa Teko", "🌍 Pale", "📞 Iteanye le Rona"],
        "contact": {
            "email": "ndlazilindy22@gmail.com",
            "phone": "+27 617150448",
            "socials": {
                "linkedin": "https://www.linkedin.com/in/ndlazi-lindiwe-76baa6229"
            }
        }
    },
    "Tswana": {
        "title": "O amogetswe mo UmojaAI",
        "tagline": "Didiriswa tsa AI tse nonotshang, tse rutang le go akaretsa botlhe.",
        "tabs": ["🤖 Puisano", "📝 Kakaretso", "🌐 Phetolelo", "🖼️ Tlhaloso ea Setšoantšo", "📝 Dipotso tsa Teko", "🌍 Pale", "📞 Iteanye le Rona"],
        "contact": {
            "email": "ndlazilindy22@gmail.com",
            "phone": "+27 617150448",
            "socials": {
                "linkedin": "https://www.linkedin.com/in/ndlazi-lindiwe-76baa6229"
            }
        }
    },
    "Tsonga": {
        "title": "Amukela eka UmojaAI",
        "tagline": "Switirhisiwa swa AI leswi tiyisaka, leswi dyondzisaka naswona swi akareta hinkwavo.",
        "tabs": ["🤖 Xinghano", "📝 Nkatsakanyo", "🌐 Muhundzuluxi", "🖼️ Nhlamuselo ya Xifaniso", "📝 Swivutiso swa AI", "🌍 Rungula", "📞 Ikhontakete"],
        "contact": {
            "email": "ndlazilindy22@gmail.com",
            "phone": "+27 617150448",
            "socials": {
                "linkedin": "https://www.linkedin.com/in/ndlazi-lindiwe-76baa6229"
            }
        }
    },
    "siSwati": {
        "title": "Uyemukelwa ku-UmojaAI",
        "tagline": "Amathuluzi e-AI lakhulisa, afundzisa futsi afaka wonkhe muntfu.",
        "tabs": ["🤖 Incokati", "📝 Umfingakiso", "🌐 Umhumushi", "🖼️ Kuchaza Semandla", "📝 Imibuto ye-AI", "🌍 Indzaba", "📞 Xhumana Nati"],
        "contact": {
            "email": "ndlazilindy22@gmail.com",
            "phone": "+27 617150448",
            "socials": {
                "linkedin": "https://www.linkedin.com/in/ndlazi-lindiwe-76baa6229"
            }
        }
    },
    "Venda": {
        "title": "U amukedzwa kha UmojaAI",
        "tagline": "Zwishumiswa zwa AI zwi khwaṱhisedzaho, zwi funzishaho na u shumisa muṅwe na muṅwe.",
        "tabs": ["🤖 Pfunzo", "📝 Tshumelo", "🌐 Mutoloi", "🖼️ Ndivho ya Tshithu", "📝 Mibuzo ya AI", "🌍 Ndzumbululo", "📞 Iteanani Na Rena"],
        "contact": {
            "email": "ndlazilindy22@gmail.com",
            "phone": "+27 617150448",
            "socials": {
                "linkedin": "https://www.linkedin.com/in/ndlazi-lindiwe-76baa6229"
            }
        }
    },
    "isiNdebele": {
        "title": "Wamukelekile e-UmojaAI",
        "tagline": "Amathuluzi e-AI aqinisa, afundisa futhi afaka bonke abantu.",
        "tabs": ["🤖 Ingxoxo", "📝 Isifinyezo", "🌐 Umhumushi", "🖼️ Ukuchaza Izithombe", "📝 Imibuzo ye-AI", "🌍 Indaba", "📞 Xhumana Nathi"],
        "contact": {
            "email": "ndlazilindy22@gmail.com",
            "phone": "+27 617150448",
            "socials": {
                "linkedin": "https://www.linkedin.com/in/ndlazi-lindiwe-76baa6229"
            }
        }
    },
    "Sepedi": {
        "title": "O amogetšwe go UmojaAI",
        "tagline": "Didirišwa tša AI tšeo di matlafatšago, tšeo di rutago le go akaretša bohle.",
        "tabs": ["🤖 Poledišano", "📝 Kakaretšo", "🌐 Mofetoleli", "🖼️ Tlhaloso ya Seswantšho", "📝 Dipotšišo tša AI", "🌍 Kanegelo", "📞 Ikgokaganye le Rena"],
        "contact": {
            "email": "ndlazilindy22@gmail.com",
            "phone": "+27 617150448",
            "socials": {
                "linkedin": "https://www.linkedin.com/in/ndlazi-lindiwe-76baa6229"
            }
        }
    },
    "Afrikaans": {
        "title": "Welkom by UmojaAI",
        "tagline": "AI-hulpmiddels wat bemagtig, opvoed en insluitend is.",
        "tabs": ["🤖 Gesprek", "📝 Opsomming", "🌐 Vertaler", "🖼️ Beeldbeskrywing", "📝 KI-Quiz", "🌍 Storie", "📞 Kontak Ons"],
        "contact": {
            "email": "ndlazilindy22@gmail.com",
            "phone": "+27 617150448",
            "socials": {
                "linkedin": "https://www.linkedin.com/in/ndlazi-lindiwe-76baa6229"
            }
        }
    }
}


lang = st.sidebar.selectbox("Language", list(UI.keys()), format_func=lambda k: {"en": "English", "zu": "isiZulu"}.get(k, k), index=0)
ui = UI.get(lang, UI["en"])

st.title(ui["title"])
st.write(ui["tagline"])
st.markdown("---")

tabs = st.tabs(ui["tabs"])

# ------------------------
# 🤖 Chatbot
# ------------------------
with tabs[0]:
    st.subheader("Chatbot Buddy")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "You are Chatbot Buddy, a friendly assistant who answers anything clearly and simply."}
        ]
    for msg in [m for m in st.session_state.chat_history if m["role"] != "system"]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])
    user_prompt = st.chat_input("Ask me anything...")
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)
        try:
            reply_text, raw = call_openrouter(
                model="openai/gpt-3.5-turbo",
                messages=st.session_state.chat_history,
                max_tokens=200,
                temperature=0.7,
            )
            reply_text = reply_text or "⚠️ No reply from model."
            st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
            with st.chat_message("assistant"):
                st.write(reply_text)
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"API error: {e}")

# ------------------------
# 📝 Summarizer
# ------------------------
with tabs[1]:
    st.subheader("Text Summarizer")
    
    # Option 1: Paste text
    long_text = st.text_area("Paste long text here…", height=180)
    
    # Option 2: Upload file
    uploaded_file = st.file_uploader("Or upload a text file (.txt)", type=["txt"], key="summarizer_file")
    if uploaded_file is not None:
        # Read uploaded file
        long_text = uploaded_file.read().decode("utf-8")

    if st.button("Summarize"):
        if not long_text.strip():
            st.warning("Please paste some text or upload a file to summarize.")
        else:
            with st.spinner("Summarizing…"):
                try:
                    prompt = f"Summarize this clearly:\n\n{long_text}"
                    summary, raw = call_openrouter(
                        model="openai/gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful summarizer."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=300,
                    )

                    st.write(summary)

                    # TXT download
                    st.download_button(
                        label="📥 Download as TXT",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )

                    # PDF download
                    pdf_buffer = BytesIO()
                    c = canvas.Canvas(pdf_buffer, pagesize=letter)
                    text_object = c.beginText(40, 750)
                    text_object.setFont("Helvetica", 12)

                    for line in summary.split("\n"):
                        text_object.textLine(line)

                    c.drawText(text_object)
                    c.showPage()
                    c.save()
                    pdf_buffer.seek(0)

                    st.download_button(
                        label="📥 Download as PDF",
                        data=pdf_buffer,
                        file_name="summary.pdf",
                        mime="application/pdf"
                    )

                except Exception as e:
                    st.error(f"API error: {e}")

# ------------------------
# 🌐 Translator
# ------------------------
with tabs[2]:
    st.subheader("Language Translator")
    
    # Option 1: Paste text
    text = st.text_area("Enter text to translate…", height=140)
    
    # Option 2: Upload file
    uploaded_file = st.file_uploader("Or upload a text file (.txt)", type=["txt"], key="translator_file")
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

    target = st.selectbox(
        "Target language",
        ["English", "isiZulu", "Afrikaans", "French", "Spanish",
         "Tshivenda","Sesotho","Sepedi","Tsonga","Setswana",
         "isiXhosa","isiSwati"]
    )

    if st.button("Translate"):
        if not text.strip():
            st.warning("Please enter text or upload a file to translate.")
        else:
            with st.spinner("Translating…"):
                try:
                    prompt = f"Translate this to {target}:\n{text}"
                    translated, raw = call_openrouter(
                        model="openai/gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a precise translator."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                        max_tokens=300,
                    )

                    st.write(translated)

                    # TXT download
                    st.download_button(
                        label="📥 Download as TXT",
                        data=translated,
                        file_name="translation.txt",
                        mime="text/plain"
                    )

                    # PDF download
                    pdf_buffer = BytesIO()
                    c = canvas.Canvas(pdf_buffer, pagesize=letter)
                    text_object = c.beginText(40, 750)
                    text_object.setFont("Helvetica", 12)

                    for line in translated.split("\n"):
                        text_object.textLine(line)

                    c.drawText(text_object)
                    c.showPage()
                    c.save()
                    pdf_buffer.seek(0)

                    st.download_button(
                        label="📥 Download as PDF",
                        data=pdf_buffer,
                        file_name="translation.pdf",
                        mime="application/pdf"
                    )

                except Exception as e:
                    st.error(f"API error: {e}")


# ------------------------
# 🖼️ Image Captioning
# ------------------------
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

with tabs[3]:
    st.subheader("Image Captioning")

    file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"], key="image_caption_file")

    if file is not None:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)

        if st.button("Generate Caption"):
            with st.spinner("Generating caption…"):
                try:
                    # Load BLIP model
                    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

                    # Generate caption
                    inputs = processor(image, return_tensors="pt")
                    out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)

                    st.write(caption)

                    # Download buttons
                    st.download_button(
                        label="📥 Download Caption as TXT",
                        data=caption,
                        file_name="caption.txt",
                        mime="text/plain"
                    )

                    # PDF
                    from io import BytesIO
                    from reportlab.lib.pagesizes import letter
                    from reportlab.pdfgen import canvas

                    pdf_buffer = BytesIO()
                    c = canvas.Canvas(pdf_buffer, pagesize=letter)
                    text_object = c.beginText(40, 750)
                    text_object.setFont("Helvetica", 12)

                    for line in caption.split("\n"):
                        text_object.textLine(line)

                    c.drawText(text_object)
                    c.showPage()
                    c.save()
                    pdf_buffer.seek(0)

                    st.download_button(
                        label="📥 Download Caption as PDF",
                        data=pdf_buffer,
                        file_name="caption.pdf",
                        mime="application/pdf"
                    )

                except Exception as e:
                    st.error(f"Error generating caption: {e}")
                    # ------------------------
                    # ------------------------
# ------------------------
# 📚 AI Knowledge Quiz with Topics
# ------------------------
with tabs[4]:  # Tab 4 for Quiz
    st.subheader("📚 AI Knowledge Quiz")

    # Step 1: User details for certificate/results
    st.markdown("### Enter Your Details")
    name = st.text_input("Full Name & Surname")
    email = st.text_input("Email Address")

    st.markdown("---")
    st.markdown("### Quiz Instructions")
    st.write("Select the correct answer for each question. A score of 60% or higher is required to pass.")

    # Step 2: Choose topic
    selected_topic = st.selectbox(
        "Choose an AI topic for your quiz:",
        ["General AI", "Machine Learning", "NLP"]  # Add more topics if needed
    )

    # Step 3: Define quizzes by topic (at least 15 questions each)
    AI_QUIZ_TOPICS = {
        "General AI": [
            {"question": "What does AI stand for?", "options": ["Artificial Intelligence", "Automated Internet", "Artificial Integration", "Algorithmic Input"], "answer": "Artificial Intelligence"},
            {"question": "Which language is commonly used for AI development?", "options": ["Python", "HTML", "CSS", "SQL"], "answer": "Python"},
            {"question": "Who is considered the father of AI?", "options": ["Alan Turing", "Bill Gates", "Steve Jobs", "Elon Musk"], "answer": "Alan Turing"},
            {"question": "Which of these is NOT AI?", "options": ["Microsoft Word", "GPT", "ChatGPT", "Computer Vision"], "answer": "Microsoft Word"},
            {"question": "What is the main goal of AI?", "options": ["Simulate human intelligence", "Create websites", "Design databases", "Print documents"], "answer": "Simulate human intelligence"},
            {"question": "Which AI application is used for voice recognition?", "options": ["Siri", "Excel", "Photoshop", "PowerPoint"], "answer": "Siri"},
            {"question": "Which AI system can play chess?", "options": ["Deep Blue", "Word", "Excel", "PowerPoint"], "answer": "Deep Blue"},
            {"question": "What is an intelligent agent?", "options": ["Software that perceives and acts", "A human assistant", "A robot only", "A website"], "answer": "Software that perceives and acts"},
            {"question": "Which AI type learns from experience?", "options": ["Machine Learning", "HTML", "CSS", "SQL"], "answer": "Machine Learning"},
            {"question": "Which AI term refers to machines understanding human language?", "options": ["NLP", "SQL", "CSS", "API"], "answer": "NLP"},
            {"question": "Which sector uses AI for self-driving cars?", "options": ["Automotive", "Education", "Healthcare", "Banking"], "answer": "Automotive"},
            {"question": "Which AI is used for recommendation systems?", "options": ["Collaborative Filtering", "Photoshop", "Excel", "Word"], "answer": "Collaborative Filtering"},
            {"question": "What is the Turing Test?", "options": ["Test machine intelligence", "Test software speed", "Test coding skills", "Test websites"], "answer": "Test machine intelligence"},
            {"question": "Which is an example of weak AI?", "options": ["Siri", "GPT", "Autonomous AGI", "All-knowing AI"], "answer": "Siri"},
            {"question": "Which AI approach mimics human brain structure?", "options": ["Neural Networks", "SQL", "HTML", "CSS"], "answer": "Neural Networks"}
        ],
        "Machine Learning": [
            {"question": "What is supervised learning?", "options": ["Learning with labeled data", "Learning without data", "Learning from mistakes only", "Learning randomly"], "answer": "Learning with labeled data"},
            {"question": "What is unsupervised learning?", "options": ["Learning patterns without labels", "Learning from tests", "Learning only supervised", "Learning with labels"], "answer": "Learning patterns without labels"},
            {"question": "Which of these is a type of neural network?", "options": ["Convolutional Neural Network", "Random Forest", "Decision Tree", "K-Means"], "answer": "Convolutional Neural Network"},
            {"question": "What is overfitting?", "options": ["Model fits training data too well", "Model performs poorly on training", "Data missing", "Perfect model"], "answer": "Model fits training data too well"},
            {"question": "Which library is used for ML in Python?", "options": ["scikit-learn", "HTML", "CSS", "Word"], "answer": "scikit-learn"},
            {"question": "Which algorithm is used for classification?", "options": ["Logistic Regression", "Linear Regression", "K-Means", "Pandas"], "answer": "Logistic Regression"},
            {"question": "Which algorithm is used for clustering?", "options": ["K-Means", "Linear Regression", "Decision Tree", "Logistic Regression"], "answer": "K-Means"},
            {"question": "Which is an ensemble method?", "options": ["Random Forest", "KNN", "Linear Regression", "SVM"], "answer": "Random Forest"},
            {"question": "What is feature selection?", "options": ["Choosing important variables", "Removing data", "Adding noise", "Random selection"], "answer": "Choosing important variables"},
            {"question": "What is cross-validation?", "options": ["Technique to check model performance", "Testing HTML", "Printing PDF", "Creating database"], "answer": "Technique to check model performance"},
            {"question": "Which metric measures classification accuracy?", "options": ["Accuracy", "Loss", "Learning rate", "Epoch"], "answer": "Accuracy"},
            {"question": "Which metric measures error in regression?", "options": ["MSE", "Accuracy", "Precision", "Recall"], "answer": "MSE"},
            {"question": "What is a learning rate?", "options": ["Step size for model update", "Learning speed for humans", "Database speed", "Website loading"], "answer": "Step size for model update"},
            {"question": "Which ML type uses labeled data?", "options": ["Supervised", "Unsupervised", "Reinforcement", "None"], "answer": "Supervised"},
            {"question": "Which ML type uses reward signals?", "options": ["Reinforcement", "Supervised", "Unsupervised", "Regression"], "answer": "Reinforcement"}
        ],
        "NLP": [
            {"question": "What is NLP?", "options": ["Natural Language Processing", "New Learning Protocol", "Neural Logic Programming", "Network Language Processing"], "answer": "Natural Language Processing"},
            {"question": "Which Python library is used for NLP?", "options": ["NLTK", "Pandas", "NumPy", "Matplotlib"], "answer": "NLTK"},
            {"question": "What is tokenization?", "options": ["Splitting text into tokens", "Joining text", "Printing data", "Analyzing PDF"], "answer": "Splitting text into tokens"},
            {"question": "What is stemming?", "options": ["Reducing words to root form", "Printing words", "Adding suffix", "Text highlighting"], "answer": "Reducing words to root form"},
            {"question": "What is lemmatization?", "options": ["Base word extraction", "Adding prefix", "Counting words", "Token splitting"], "answer": "Base word extraction"},
            {"question": "What is named entity recognition?", "options": ["Identifying entities in text", "Printing entities", "Summarizing text", "Counting characters"], "answer": "Identifying entities in text"},
            {"question": "Which NLP task converts speech to text?", "options": ["Speech Recognition", "Text Summarization", "Translation", "POS tagging"], "answer": "Speech Recognition"},
            {"question": "Which NLP task summarizes text?", "options": ["Text Summarization", "Translation", "Speech Recognition", "NER"], "answer": "Text Summarization"},
            {"question": "Which NLP task translates text?", "options": ["Translation", "Text Summarization", "Speech Recognition", "NER"], "answer": "Translation"},
            {"question": "What is POS tagging?", "options": ["Part-of-Speech tagging", "PDF tagging", "Word counting", "Token splitting"], "answer": "Part-of-Speech tagging"},
            {"question": "Which model is used for NLP tasks?", "options": ["GPT", "CNN", "Random Forest", "K-Means"], "answer": "GPT"},
            {"question": "Which method converts text into numbers for ML?", "options": ["Vectorization", "Stemming", "Lemmatization", "Tokenization"], "answer": "Vectorization"},
            {"question": "What is sentiment analysis?", "options": ["Detecting emotion in text", "Printing emotions", "Counting words", "Token splitting"], "answer": "Detecting emotion in text"},
            {"question": "Which library is for NLP deep learning?", "options": ["Transformers", "Pandas", "NumPy", "Matplotlib"], "answer": "Transformers"},
            {"question": "Which NLP technique detects topics in text?", "options": ["Topic Modeling", "Vectorization", "Tokenization", "POS tagging"], "answer": "Topic Modeling"}
        ]
    }

    # Load quiz for the selected topic
    quiz = AI_QUIZ_TOPICS[selected_topic]

    # Step 4: Take user answers
    user_score = 0
    user_responses = []
    for idx, q in enumerate(quiz):
        st.markdown(f"**Q{idx+1}: {q['question']}**")
        user_answer = st.radio("Choose an answer:", q["options"], key=f"{selected_topic}_q{idx}")
        user_responses.append((q['question'], user_answer, q["answer"]))
        if user_answer == q["answer"]:
            user_score += 1

    if st.button("Submit Quiz"):
        st.markdown("---")
        # Calculate percentage
        score_percent = round((user_score / len(quiz)) * 100, 2)
        st.markdown(f"### Your Score: {score_percent}% ({user_score}/{len(quiz)})")

        passing_score = 60  # 60% threshold

        if score_percent >= passing_score:
            st.success("🎉 Congratulations! You passed!")
            passed = True
        else:
            st.error("😢 Sorry, you did not pass.")
            passed = False

        # -------------------------
        # PDF Generation (certificate or results)
        # -------------------------
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from io import BytesIO
        import datetime, random

        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter

        # Background & borders
        c.setFillColorRGB(1, 0.992, 0.925)  # cream-white
        c.rect(0, 0, width, height, fill=1, stroke=0)
        c.setStrokeColorRGB(0.4, 0.26, 0.13)
        c.setLineWidth(5)
        c.rect(20, 20, width-40, height-40)
        c.setLineWidth(2)
        c.rect(35, 35, width-70, height-70)
        if passed:
    # Certificate content
    c.setFont("Times-Bold", 34)
    c.setFillColorRGB(0.0, 0.2, 0.0)
    c.drawCentredString(width/2, height - 150, "Certificate of Achievement")
    c.setFont("Helvetica", 16)
    c.setFillColorRGB(0, 0, 0)
    c.drawCentredString(width/2, height - 200, "This certificate is proudly awarded to")
    c.setFont("Times-Bold", 26)
    c.drawCentredString(width/2, height - 250, name)
    c.setFont("Helvetica", 16)
    c.drawCentredString(width/2, height - 300,
                        f"For successfully demonstrating knowledge in {selected_topic}")
    c.drawCentredString(width/2, height - 330, f"Score: {score_percent}% ({user_score}/{len(quiz)})")
    c.drawCentredString(width/2, height - 360,
                        f"Date: {datetime.date.today().strftime('%B %d, %Y')}")
    c.drawCentredString(width/2, height - 380, "Issued in South Africa")
    
   # Certificate ID
            cert_id = f"UBC-{datetime.date.today().strftime('%Y%m%d')}-{random.randint(1000,9999)}"
            c.setFont("Helvetica-Oblique", 10)
            c.drawRightString(width - 40, 40, f"Certificate ID: {cert_id}")

            # Signature line
            c.line(width/2 - 100, height - 450, width/2 + 100, height - 450)
            c.setFont("Helvetica-Bold", 12)
            c.drawCentredString(width/2, height - 465, "LM Ndlazi")
            c.setFont("Helvetica", 10)
            c.drawCentredString(width/2, height - 480, "Programme Lead, UmojaAI – Career Bridge Initiative")

        else:
            # Quiz results
            c.setFont("Helvetica-Bold", 20)
            c.drawCentredString(width/2, height - 100, f"{selected_topic} Quiz Results")
            c.setFont("Helvetica", 14)
            c.drawString(50, height - 150, f"Name: {name}")
            c.drawString(50, height - 170, f"Email: {email}")
            c.drawString(50, height - 200, f"Score: {score_percent}% ({user_score}/{len(quiz)})")
            c.drawString(50, height - 230, f"Passing Score: {passing_score}%")
            c.drawString(50, height - 260, "Detailed Results:")
            y_pos = height - 290
            for q_text, ua, ca in user_responses:
                c.drawString(60, y_pos, f"Q: {q_text}")
                y_pos -= 20
                c.drawString(70, y_pos, f"Your Answer: {ua}")
                y_pos -= 20
                c.drawString(70, y_pos, f"Correct Answer: {ca}")
                y_pos -= 30

        c.save()
        pdf_buffer.seek(0)


        # Download PDF
        if passed:
            st.download_button(
                label="📥 Download Certificate",
                data=pdf_buffer,
                file_name=f"{name}_certificate.pdf",
                mime="application/pdf"
            )
        else:
            st.download_button(
                label="📥 Download Results",
                data=pdf_buffer,
                file_name=f"{name}_{selected_topic}_quiz_results.pdf",
                mime="application/pdf"
            )

# ------------------------
# 🌍 Story
# ------------------------
with tabs[5]:
    st.subheader("Why I Built This")

    # Create two columns: left for image, right for text
    col1, col2 = st.columns([1, 3])  # Adjust ratio as needed

    with col1:
        st.image("images/liindii.jpg", width=200, caption="Lindiwe Ndlazi")
 # Replace with your image path

    with col2:
        st.markdown(
            """
As a professional who began my career in Human Resource Management (HRM) and later transitioned into the field of Information Technology and Artificial Intelligence, 
I experienced firsthand the challenges of accessing practical learning resources and career development tools. 
This journey inspired me to create UmojaAI — a platform that leverages AI to empower individuals, particularly those from underrepresented or underserved backgrounds, 
by providing accessible, language-driven tools for learning, skill development, and career advancement.

Through this project, my mission is to make technology more inclusive, practical, and empowering, ensuring that anyone, regardless of their background or starting point, 
can harness the power of AI to learn, grow, and succeed in today’s digital and technology-driven world.
            """
        )

    
    # ------------------------
# 📞 Contact
# ------------------------
with tabs[6]:
    st.subheader("Contact Us")
    contact = ui.get("contact", {})

    st.write(f"📧 Email: {contact.get('email', 'N/A')}")
    st.write(f"📞 Phone: {contact.get('phone', 'N/A')}")

    socials = contact.get("socials", {})
    linkedin_url = socials.get("linkedin", "")
    if linkedin_url:
        st.write(f"🔗 LinkedIn: [{linkedin_url}]({linkedin_url})")

    
   
   
    
    
   
    
































