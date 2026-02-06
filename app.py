import os
import base64
import requests
import streamlit as st
from dotenv import load_dotenv
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import datetime
import random


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
# 📚 AI Knowledge Quiz App
# ------------------------
st.set_page_config(page_title="AI Knowledge Quiz", layout="wide")

st.title("📚 AI Knowledge Quiz")

# Step 1: User details
st.subheader("Enter Your Details")
name = st.text_input("Full Name & Surname")
email = st.text_input("Email Address")

st.markdown("---")
st.subheader("Quiz Instructions")
st.write("Select the correct answer for each question. A score of 60% or higher is required to pass.")

# ------------------------
# Step 2: Topics & Questions
# ------------------------
topics = {
    "AI Basics": [
        {"question": "What does AI stand for?", "options": ["Artificial Intelligence", "Automated Internet", "Artificial Integration", "Algorithmic Input"], "answer": "Artificial Intelligence"},
        {"question": "Which language is commonly used for AI development?", "options": ["Python", "HTML", "CSS", "SQL"], "answer": "Python"},
        {"question": "What is supervised learning?", "options": ["Learning with labeled data", "Learning without data", "Learning from mistakes only", "Learning without supervision"], "answer": "Learning with labeled data"},
        {"question": "What is an algorithm?", "options": ["Step-by-step instructions", "A programming language", "A type of AI model", "A dataset"], "answer": "Step-by-step instructions"},
        {"question": "Which AI technique is used for predictions?", "options": ["Machine Learning", "Manual Programming", "Graphic Design", "SQL Queries"], "answer": "Machine Learning"},
        {"question": "Which AI is designed to mimic human thinking?", "options": ["Cognitive AI", "Reactive AI", "Strong AI", "Narrow AI"], "answer": "Cognitive AI"},
        {"question": "Which AI is specialized in one task?", "options": ["Narrow AI", "Strong AI", "General AI", "Super AI"], "answer": "Narrow AI"},
        {"question": "What does NLP stand for?", "options": ["Natural Language Processing", "New Learning Protocol", "Neural Logic Programming", "Network Language Processing"], "answer": "Natural Language Processing"},
        {"question": "Which is an example of AI in daily life?", "options": ["Chatbots", "Cars", "Books", "Pencils"], "answer": "Chatbots"},
        {"question": "Which data is used for supervised learning?", "options": ["Labeled data", "Unlabeled data", "Random data", "No data"], "answer": "Labeled data"},
        {"question": "What is reinforcement learning?", "options": ["Learning by reward and punishment", "Learning with labeled data", "Learning without data", "Learning by copying"], "answer": "Learning by reward and punishment"},
        {"question": "Which AI model is used for image recognition?", "options": ["CNN", "RNN", "LSTM", "Decision Tree"], "answer": "CNN"},
        {"question": "Which is an AI programming library?", "options": ["TensorFlow", "Excel", "Photoshop", "PowerPoint"], "answer": "TensorFlow"},
        {"question": "Which AI technique generates new data?", "options": ["Generative AI", "Supervised AI", "Reactive AI", "Narrow AI"], "answer": "Generative AI"},
        {"question": "Which AI mimics human intelligence fully?", "options": ["Strong AI", "Narrow AI", "Weak AI", "Reactive AI"], "answer": "Strong AI"},
    ],
    "Machine Learning": [
        {"question": "Which algorithm is used for classification?", "options": ["Decision Tree", "K-Means", "PCA", "Linear Regression"], "answer": "Decision Tree"},
        {"question": "Which algorithm is used for regression?", "options": ["Linear Regression", "Naive Bayes", "KNN", "SVM"], "answer": "Linear Regression"},
        {"question": "What is overfitting?", "options": ["Model fits training data too well", "Model predicts perfectly on new data", "Model ignores training data", "Model underperforms"], "answer": "Model fits training data too well"},
        {"question": "What is underfitting?", "options": ["Model is too simple", "Model is too complex", "Model predicts perfectly", "Model memorizes data"], "answer": "Model is too simple"},
        {"question": "Which is unsupervised learning?", "options": ["Clustering", "Linear Regression", "Logistic Regression", "Decision Tree"], "answer": "Clustering"},
        {"question": "Which is a supervised learning algorithm?", "options": ["SVM", "K-Means", "PCA", "DBSCAN"], "answer": "SVM"},
        {"question": "What is reinforcement learning?", "options": ["Learning by reward and punishment", "Learning from labeled data", "Clustering data", "Reducing dimensionality"], "answer": "Learning by reward and punishment"},
        {"question": "Which metric measures classification performance?", "options": ["Accuracy", "Loss", "Mean Squared Error", "R-Squared"], "answer": "Accuracy"},
        {"question": "Which metric measures regression performance?", "options": ["MSE", "Accuracy", "F1 Score", "Precision"], "answer": "MSE"},
        {"question": "Which method prevents overfitting?", "options": ["Regularization", "Clustering", "Gradient Boosting", "PCA"], "answer": "Regularization"},
        {"question": "What is a neural network?", "options": ["Layers of interconnected nodes", "Database", "Spreadsheet", "Image filter"], "answer": "Layers of interconnected nodes"},
        {"question": "Which is a deep learning framework?", "options": ["PyTorch", "Excel", "SQL", "HTML"], "answer": "PyTorch"},
        {"question": "Which activation function outputs 0 or 1?", "options": ["Sigmoid", "ReLU", "Tanh", "Softmax"], "answer": "Sigmoid"},
        {"question": "Which ML technique reduces dimensionality?", "options": ["PCA", "KNN", "SVM", "Decision Tree"], "answer": "PCA"},
        {"question": "Which algorithm groups similar data points?", "options": ["K-Means", "Linear Regression", "Decision Tree", "Random Forest"], "answer": "K-Means"},
    ],
    "Responsible AI": [
        {"question": "What is bias in AI?", "options": ["Unfair outcomes", "Correct predictions", "Faster processing", "More accuracy"], "answer": "Unfair outcomes"},
        {"question": "Which principle promotes fairness?", "options": ["Fairness", "Speed", "Scalability", "Complexity"], "answer": "Fairness"},
        {"question": "What is explainability?", "options": ["Ability to understand AI decisions", "AI speed", "AI size", "AI color"], "answer": "Ability to understand AI decisions"},
        {"question": "What is transparency in AI?", "options": ["Open understanding of AI logic", "AI secret codes", "Hidden processes", "Closed systems"], "answer": "Open understanding of AI logic"},
        {"question": "Which tool audits AI for bias?", "options": ["IBM AI Fairness 360", "Photoshop", "Excel", "TensorFlow"], "answer": "IBM AI Fairness 360"},
        {"question": "Which is NOT a Responsible AI principle?", "options": ["Dishonesty", "Fairness", "Transparency", "Accountability"], "answer": "Dishonesty"},
        {"question": "Which helps mitigate bias?", "options": ["Bias detection & mitigation strategies", "Random guessing", "Ignoring data", "Speed optimization"], "answer": "Bias detection & mitigation strategies"},
        {"question": "What is accountability in AI?", "options": ["Being responsible for AI outcomes", "Faster computation", "Memory usage", "Code optimization"], "answer": "Being responsible for AI outcomes"},
        {"question": "Which AI can harm society if unchecked?", "options": ["Unethical AI", "Supervised AI", "Reactive AI", "Narrow AI"], "answer": "Unethical AI"},
        {"question": "Which ensures AI follows laws?", "options": ["Regulations & policies", "Neural networks", "Gradient descent", "Data cleaning"], "answer": "Regulations & policies"},
        {"question": "What is privacy in AI?", "options": ["Protection of user data", "AI speed", "AI complexity", "Model size"], "answer": "Protection of user data"},
        {"question": "Which mitigates discrimination in AI?", "options": ["Fairness-aware algorithms", "Random models", "Overfitting", "Unsupervised clustering"], "answer": "Fairness-aware algorithms"},
        {"question": "What is Responsible AI design?", "options": ["Building ethical AI", "Building fast AI", "Building large AI", "Building complex AI"], "answer": "Building ethical AI"},
        {"question": "Which promotes trust in AI?", "options": ["Transparency & fairness", "Complexity", "Hidden logic", "Speed"], "answer": "Transparency & fairness"},
        {"question": "What is inclusive AI?", "options": ["AI that serves all users fairly", "AI for one group only", "AI that excludes people", "AI that predicts incorrectly"], "answer": "AI that serves all users fairly"},
    ]
}

# Step 3: Select Topic
selected_topic = st.selectbox("Select AI Topic", list(topics.keys()))
quiz = topics[selected_topic]

# Step 4: Display Questions
st.markdown("---")
st.subheader(f"{selected_topic} Quiz")
user_answers = []
for i, q in enumerate(quiz):
    user_answer = st.radio(f"{i+1}. {q['question']}", q['options'], key=f"q{i}")
    user_answers.append(user_answer)

# Step 5: Submit & Scoring
if st.button("Submit Quiz"):
    user_score = sum([1 for i, q in enumerate(quiz) if user_answers[i] == q['answer']])
    score_percent = round(user_score / len(quiz) * 100, 2)
    passing_score = 60
    user_responses = [(q['question'], user_answers[i], q['answer']) for i, q in enumerate(quiz)]
    
    # Pass/Fail message
    if user_score >= int(len(quiz) * 0.6):
        st.success(f"🎉 Congratulations {name}! You passed the {selected_topic} quiz with {score_percent}%.")
    else:
        st.error(f"😔 Sorry {name}, you did not pass the {selected_topic} quiz. Your score is {score_percent}%.")

    # ------------------------
    # PDF Generation
    # ------------------------
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    # Background & borders
    c.setFillColorRGB(1, 0.992, 0.925)
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setStrokeColorRGB(0.4, 0.26, 0.13)
    c.setLineWidth(5)
    c.rect(20, 20, width-40, height-40)
    c.setLineWidth(2)
    c.rect(35, 35, width-70, height-70)

    # Header
    c.setFont("Times-Bold", 22)
    c.setFillColorRGB(0.0, 0.2, 0.0)
    c.drawCentredString(width/2, height - 70, "UmojaAI – Career Bridge Initiative")
    c.setFont("Helvetica-Oblique", 11)
    c.setFillColorRGB(0, 0, 0)
    c.drawCentredString(width/2, height - 90, "An AI-powered career guidance and digital skills initiative")

    # Watermark
    c.saveState()
    c.setFont("Helvetica-Bold", 50)
    c.setFillColorRGB(0.9, 0.9, 0.9)
    c.translate(width/2, height/2)
    c.rotate(30)
    c.drawCentredString(0, 0, "UmojaAI – Career Bridge ")
    c.restoreState()

    if user_score >= int(len(quiz) * 0.6):
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
        c.drawCentredString(width/2, height - 300, f"For successfully demonstrating knowledge in {selected_topic}")
        c.drawCentredString(width/2, height - 330, f"Score: {score_percent}% ({user_score}/{len(quiz)})")
        c.drawCentredString(width/2, height - 360, f"Date: {datetime.date.today().strftime('%B %d, %Y')}")
        c.drawCentredString(width/2, height - 380, "Issued in South Africa")
        cert_id = f"UBC-{datetime.date.today().strftime('%Y%m%d')}-{random.randint(1000,9999)}"
        c.setFont("Helvetica-Oblique", 10)
        c.drawRightString(width - 40, 40, f"Certificate ID: {cert_id}")
if passed:
    # -------------------------
    # Signature (Aligned like Certificate ID)
    # -------------------------
    sig_x = 60
    sig_y = 55  # aligned visually with Certificate ID height

    # Signature line
    c.setStrokeColorRGB(0.4, 0.26, 0.13)  # brown
    c.setLineWidth(1.2)
    c.line(sig_x, sig_y, sig_x + 160, sig_y)

    # Signature name (handwritten style)
    c.setFont("Times-Italic", 12)
    c.setFillColorRGB(0.0, 0.2, 0.0)  # dark green
    c.drawString(sig_x + 10, sig_y + 8, "LM Ndlazi")

    # Signature title
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(
        sig_x,
        sig_y - 14,
        "Programme Lead · UmojaAI – Career Bridge Initiative"
    )

else:
    # Quiz results content
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

# Download button
if user_score >= int(len(quiz) * 0.6):
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
        file_name=f"{name}_quiz_results.pdf",
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

    
   
   
    
    
   
    






































