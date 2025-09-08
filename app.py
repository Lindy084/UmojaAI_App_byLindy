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
        "tabs":   ["🤖 Ingxoxo", "📝 Isifinyezo", "🌐 Umhumushi", "🖼️ Ukuchaza Izithombe", "📝 Imibuzo yokuhlola", "🌍 Indaba", "📞 Xhumana Nathi"],
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
            "email": "contact@umojaai.com",
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
        "tabs": 
        ["🤖 Ingxoxo", "📝 Isifinyezo", "🌐 Umhumushi", "🖼️ Ukuchaza Izithombe", "📝 Imibuzo ye-AI", "🌍 Indaba", "📞 Xhumana Nathi"],

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
            "phone": "061715048",
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
# 📚 AI Knowledge Quiz
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

    # Step 2: Pre-defined multiple-choice questions
    quiz = [
        {"question": "What does AI stand for?", "options": ["Artificial Intelligence", "Automated Internet", "Artificial Integration", "Algorithmic Input"], "answer": "Artificial Intelligence"},
        {"question": "Which language is commonly used for AI development?", "options": ["Python", "HTML", "CSS", "SQL"], "answer": "Python"},
        {"question": "What is supervised learning?", "options": ["Learning with labeled data", "Learning without data", "Learning from mistakes only", "Learning without supervision"], "answer": "Learning with labeled data"},
        {"question": "Which of these is a type of neural network?", "options": ["Convolutional Neural Network", "Random Forest", "Decision Tree", "K-Means"], "answer": "Convolutional Neural Network"},
        {"question": "What is NLP?", "options": ["Natural Language Processing", "New Learning Protocol", "Neural Logic Programming", "Network Language Processing"], "answer": "Natural Language Processing"},
        {"question": "Which Python library is used for data manipulation?", "options": ["Pandas", "NumPy", "Matplotlib", "Seaborn"], "answer": "Pandas"},
        {"question": "What is overfitting in machine learning?", "options": ["Model performs well on training but poorly on new data", "Model underperforms on training", "Data not enough", "Model is perfect"], "answer": "Model performs well on training but poorly on new data"},
        {"question": "Which of the following is a cloud service provider?", "options": ["AWS", "GitHub", "Python", "Excel"], "answer": "AWS"},
        {"question": "What is a chatbot?", "options": ["AI program that can converse with humans", "A website", "A type of database", "A programming language"], "answer": "AI program that can converse with humans"},
        {"question": "Which of these is a type of machine learning?", "options": ["Supervised", "Randomized", "Sequential", "Structured"], "answer": "Supervised"},
        {"question": "What is a dataset?", "options": ["Collection of data", "A programming language", "A neural network", "A text editor"], "answer": "Collection of data"},
        {"question": "Which of these is used for data visualization?", "options": ["Matplotlib", "SQL", "Python", "AWS"], "answer": "Matplotlib"},
        {"question": "What does API stand for?", "options": ["Application Programming Interface", "Artificial Programming Intelligence", "Automated Process Integration", "Applied Protocol Interface"], "answer": "Application Programming Interface"},
        {"question": "Which of these is an AI framework?", "options": ["TensorFlow", "Bootstrap", "Django", "Flask"], "answer": "TensorFlow"},
        {"question": "What is deep learning?", "options": ["Subset of machine learning using neural networks", "Basic programming", "Database querying", "Data visualization"], "answer": "Subset of machine learning using neural networks"},
        {"question": "What is reinforcement learning?", "options": ["Learning by trial and error with rewards", "Learning with labeled data", "Learning without data", "Learning randomly"], "answer": "Learning by trial and error with rewards"},
        {"question": "Which of these is an AI language model?", "options": ["GPT", "HTML", "CSS", "SQL"], "answer": "GPT"},
        {"question": "What is a model in machine learning?", "options": ["Algorithm trained on data to make predictions", "A type of website", "A text file", "A programming language"], "answer": "Algorithm trained on data to make predictions"},
        {"question": "Which of the following is NOT AI?", "options": ["Microsoft Word", "GPT", "Computer Vision", "ChatGPT"], "answer": "Microsoft Word"},
        {"question": "Which metric measures classification accuracy?", "options": ["Accuracy", "Loss", "Learning rate", "Epoch"], "answer": "Accuracy"},
        {"question": "What is bias in AI?", "options": ["Systematic error in predictions", "Random mistakes", "Perfect predictions", "Data cleaning"], "answer": "Systematic error in predictions"},
    ]

    # Step 3: Take user answers
    user_score = 0
    user_responses = []
    for idx, q in enumerate(quiz):
        st.markdown(f"**Q{idx+1}: {q['question']}**")
        user_answer = st.radio("Choose an answer:", q["options"], key=f"q{idx}")
        user_responses.append((q['question'], user_answer, q["answer"]))
        if user_answer == q["answer"]:
            user_score += 1

    if st.button("Submit Quiz"):
        st.markdown("---")
        st.markdown(f"### Your Score: {user_score}/{len(quiz)}")
        passing_score = int(0.6 * len(quiz))  # 60% pass threshold

        if user_score >= passing_score:
            st.success("🎉 Congratulations! You passed!")
            passed = True
        else:
            st.error("😢 Sorry, you did not pass.")
            passed = False

        
         # Step 4: Generate PDF certificate or results
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from io import BytesIO
        import datetime, random

        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter

        # Gradient-like background
        c.saveState()
        for i in range(0, int(height), 5):
            c.setFillColorRGB(0.93, 0.95 - i/2000, 1)  # Light gradient from blue to white
            c.rect(0, i, width, 5, fill=1, stroke=0)
        c.restoreState()

        # Decorative double border
        c.setStrokeColor(colors.darkblue)
        c.setLineWidth(4)
        c.rect(20, 20, width-40, height-40)
        c.setLineWidth(2)
        c.rect(30, 30, width-60, height-60)

        # Institution header
        c.setFont("Helvetica-Bold", 20)
        c.setFillColor(colors.darkblue)
        c.drawCentredString(width/2, height - 50, "UmojaAI Inclusive")
        c.setFont("Helvetica-Oblique", 12)
        c.setFillColor(colors.black)
        c.drawCentredString(width/2, height - 70, "Empowering Everyone Through AI")

        # Watermark placeholder
        c.setFont("Helvetica-Bold", 50)
        c.setFillColorRGB(0.9,0.9,0.9)
        c.saveState()
        c.translate(width/2 - 150, height/2 - 50)
        c.rotate(30)
        c.drawCentredString(0, 0, "UmojaAI")
        c.restoreState()
        c.setFillColor(colors.black)

        if passed:
            # Certificate design
            c.setFont("Helvetica-Bold", 32)
            c.setFillColor(colors.darkblue)
            c.drawCentredString(width/2, height - 130, "Certificate of Achievement")
            c.setFont("Helvetica", 16)
            c.setFillColor(colors.black)
            c.drawCentredString(width/2, height - 180, "This certificate is proudly awarded to")
            c.setFont("Helvetica-Bold", 24)
            c.drawCentredString(width/2, height - 230, name)
            c.setFont("Helvetica", 16)
            c.drawCentredString(width/2, height - 280, "For successfully passing the AI Knowledge Quiz")
            c.drawCentredString(width/2, height - 310, f"Score: {user_score}/{len(quiz)}")
            c.drawCentredString(width/2, height - 340, f"Date: {datetime.date.today().strftime('%B %d, %Y')}")

            # Certificate ID
            cert_id = f"UMJ-{datetime.date.today().strftime('%Y%m%d')}-{random.randint(1000,9999)}"
            c.setFont("Helvetica-Oblique", 10)
            c.drawRightString(width - 40, 40, f"Certificate ID: {cert_id}")

            # Signature line with your name
            c.line(80, 80, 280, 80)
            c.setFont("Helvetica-BoldOblique", 12)
            c.drawString(120, 82, "LM Ndlazi")  # Director Signature above line
            c.setFont("Helvetica", 10)
            c.drawString(120, 68, "Director, UmojaAI Inclusive")
        else:
            # Results PDF
            c.setFont("Helvetica-Bold", 20)
            c.drawCentredString(width/2, height - 100, "Quiz Results")
            c.setFont("Helvetica", 14)
            c.drawString(50, height - 150, f"Name: {name}")
            c.drawString(50, height - 170, f"Email: {email}")
            c.drawString(50, height - 200, f"Score: {user_score}/{len(quiz)}")
            c.drawString(50, height - 230, f"Passing Score: {passing_score}/{len(quiz)}")
            c.drawString(50, height - 260, "Detailed Results:")
            y_pos = height - 290
            for q_text, ua, ca in user_responses:
                c.drawString(60, y_pos, f"Q: {q_text}")
                y_pos -= 20
                c.drawString(70, y_pos, f"Your Answer: {ua}")
                y_pos -= 20
                c.drawString(70, y_pos, f"Correct Answer: {ca}")
                y_pos -= 30

        c.showPage()
        c.save()
        pdf_buffer.seek(0)

        # Step 5: Download button
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
        st.image("images/Lindi.jpg", width=200, caption="Lindiwe Ndlazi")
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
    st.write(
        f"LinkedIn: [{socials.get('linkedin','')}](https://{socials.get('linkedin','')})"
    )
   


    )
   
   
    
    
   
    

















