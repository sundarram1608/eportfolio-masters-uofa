import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components
import base64
import textwrap
from PIL import Image


icon_path = Path(__file__).parent / "images" / "page_logo.jpeg"
icon = Image.open(icon_path)

st.set_page_config(
                    page_title="E-Portfolio - Sundar Ram Subramanian",
                    page_icon=icon,
                    layout="wide"
                )


# -----------------------------
# Helpers
# -----------------------------
ASSETS = Path(__file__).parent / "images"

FILES_DIR = Path(__file__).parent / "images"  # same folder you use


def file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def render_image(path: Path):
    b64 = file_to_base64(path)
    ext = path.suffix.lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    img_html = f"""
    <div style="display:flex; justify-content:center;">
      <img src="data:{mime};base64,{b64}" style="max-width:95%; height:auto;" />
    </div>
    """
    components.html(img_html, height=900, scrolling=True)

def img_path(filename: str) -> str:
    p = ASSETS / filename
    return str(p) if p.exists() else ""

def pill(text: str) -> str:
    return textwrap.dedent(f"""
    <span style="display:inline-block;
        padding:6px 10px;
        margin:4px 6px 0 0;
        border-radius:999px;
        border:1px solid rgba(0,0,0,.12);
        background:rgba(0,0,0,.04);
        font-size:0.85rem;">{text}</span>
    """).strip()

def to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""

    ext = path.suffix.lower()
    mime_map = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    mime = mime_map.get(ext, "application/octet-stream")

    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def img_uri(filename: str) -> str:
    return to_data_uri(ASSETS / filename)


def show_transcript_pdf(pdf_path: Path, max_pages: int = 6):
    """
    Reliable preview: render PDF pages as images using pypdfium2.
    This avoids browser issues with iframes/data-URLs.
    """
    if not pdf_path.exists():
        st.error(f"Transcript PDF not found: {pdf_path}")
        return

    try:
        import pypdfium2 as pdfium
    except Exception:
        st.error("Missing dependency for PDF preview: pypdfium2")
        st.code("pip install pypdfium2")
        return

    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)
    # st.info(f"Showing first {min(max_pages, n_pages)} page(s) of the transcript. Download below for full PDF.")

    for i in range(min(max_pages, n_pages)):
        page = pdf[i]
        pil_image = page.render(scale=2).to_pil()  # scale=2 is readable
        st.image(pil_image, caption=f"Page {i+1}", use_container_width=True)


# ---- NEW TAB VIEWER for viewing transcripts and degree ----
doc = st.query_params.get("doc", None)

if isinstance(doc, list):
    doc = doc[0] if doc else None

if doc == "transcript":
    # st.title("Unofficial Transcript")

    pdf_path = FILES_DIR / "unofficial_transcript.pdf"

    # Always allow download
    if pdf_path.exists():
        st.download_button(
            "Download PDF",
            data=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
        )
    # Preview (renders pages)
    show_transcript_pdf(pdf_path, max_pages=6)


    st.stop()

if doc == "degree":
    img_path = FILES_DIR / "degree.jpeg"  # change to .jpeg/.png if needed
    # st.title("Degree")
    if img_path.exists():
        render_image(img_path)
        # st.download_button("Download Image", data=img_path.read_bytes(),
        #                    file_name=img_path.name, mime="image/jpeg")
    else:
        st.error(f"File not found: {img_path}")
    st.stop()


# -----------------------------
# Data
# -----------------------------
PROFILE = {
            "capsname": "SUNDAR RAM SUBRAMANIAN",
            "name": "Sundar Ram Subramanian",
            "portfolio_title": "ACADEMIC E-PORTFOLIO",
            "program": "Masters in Information Science (Machine Learning Specialization)",
            "college": "College of Information Science",
            "year": "Batch of 2024 - 2026",
            "status": "Pursuing",
            "email": "sundarram1997@arizona.edu",
            # "linkedin": "https://www.linkedin.com/in/sundar-ram-subramanian",
            # "github": "https://github.com/your-handle",
        }

COURSES = [
            {
                "semester": "Fall 2025",
                "course_name": "INFO 555: Applied NLP",
                "grade": "A",
                "skills": ["Data Acquisition", "Text Preprocessing", "Tokenization", "Stemming", "Lemmatization", "POS Tagging", "Named Entity Recognition", "Word Embeddings", "Word2Vec", "GloVe", "TF-IDF", "Text Classification", "Sentiment Analysis", "Text Summarization", "Topic Modeling", "Sequence Processing", "RNN", "GRU", "LSTM", "Encoder-Decoder", "Attention", "self-Attention", "Transformers", "BERT", "GPT", "BLEU Score", "Pre-Trained LLMs", "LLM Fine-Tuning", "Prompting Techniques", "Memory Augmentation", "RAG", "Tools", "Agentic AI"],
                "project":{
                                "title": "Semeval Task Solutions using NLP Techniques",
                                "description": "Several Semeval tasks including Semantic Textual Similarity, Offenseval, Measeval and ComVE were solved using NLP techniques. Several python libraries like gensim for GloVe embeddings, scikit-learn for classification models, Keras for Deep Learning, and Hugging face transformers for finetuning of pre-trained language models of BERT were leveraged.",
                            }
            },
            {
                "semester": "Fall 2025",
                "course_name": "INFO 556: Text Retrieval and Web Search",
                "grade": "A",
                "skills": ["Music Information Retrieval", "Digital Signal Processing", "Deep Learning Retrieval", "Inverted Index", "Text Preprocessing", "Boolean Model", "Vector Space Model", "Probabilistic Model", "TF-IDF", "BM25", "Query likelihood", "Language Modeling", "Pseudo feedback", "Implicit feedback", "Rocchio Method", "Mixture Model", "MapReduce","Page Rank", "Content-based filtering","Collaborative filtering", "HITS Algorithm", "User-centered IR Evaluation", "Research Design", "Research Review"],
                "project":{
                                "title": "Carnatic Music Raga Identification using Music Information Retrieval + Deep Learning",
                                "description": "This project developed a deep learning based Carnatic Music Raga Retrieval System. It integrates Music Information Retrieval (MIR) techniques for extracting characteristic Raga features from digital audio, with modern deep neural networks for Raga classification. Patented dataset Indian Art Music Raga Recognition datasets of UPF has been leveraged for feature extraction and to train and evaluate models for raga classification of 8 Melakarta ragas. Python’s Librosa library is used for feature extraction from audio file and a Fusion Deep Learning Model (CNN + LSTM) with SoftMax output has been built for Raga Identification and Classification."
                            }
            },                        
            {
                "semester": "Fall 2025",
                "course_name": "INFO 510: Bayesian Modeling & Inference",
                "grade": "A",
                "skills": ["Prior", "Posterior", "Likelihood", "Conjugate Priors", "MCMC", "Gibbs Sampling", "Metropolis-Hastings","Graphs", "Gaussian Mixture Model","Hierarchical Models"],
                "project":{
                                "title": "Bayesian Analysis of Price Determinants and Quality Indicators in the U.S. Airbnb Market",
                                "description": "This project analyses the US Airbnb 2023 dataset of Kaggle using Multi level Bayesian Hierarchical models including Bayesian hierarchical linear & logistic regression models to model uncertainty, prior beliefs, and hierarchical clusters in the dataset. The analysis helps identify factors influencing Airbnb listing prices and user-engagement.",
                            }
            },
            {
                "semester": "Fall 2025",
                "course_name": "INFO 531: Data Warehousing and Analytics in the Cloud",
                "grade": "A",
                "skills": ["Data Warehousing","Database Design", "Normalization (1NF, 2NF, 3NF)", "Data Modeling (Conceptual, Logical, Physical)", "Referential Integrity", "SQL", "Cloud Computing", "Apache Spark", "Azure Synapse Analytics", "Relational Data warehouse", "Azure Synapse Pipeline", "Azure Stream Analytics", "Azure Databricks"],
                "project":{
                                "title": "Database Creation & UI Development",
                                "description": "This project involved Design and development of a comprehensive Data Management and Analytics System on the modified Northwind dataset. The system integrates a well-structured backend built using python, a robust relational database powered by MYSQL, and a streamlit powered intuitive front-end interface that supports a complete analytical workflow, including data ingestion, validation, CRUD operations, and visualization.",
                            }
            },
            ###########
            {
                "semester": "Spring 2025",
                "course_name": "INFO 521: Introduction to Machine Learning",
                "grade": "A",
                "skills": ["Linear Regression", "Ridge", "Lasso", "Multiple Linear Regression", "Polynomial Regression", "Classification","SVM","Random Forest", "Catboost", "Light GBM", "Bias-Variance trade off", "Resampling", "Hyper-parameter tuning"],
                "project":{
                                "title": "German Bank Loan Default Prediction",
                                "description": "This project predicts loan defaults using historical data from customers of a German bank. Several independent models were built and tuned for hyper parametes along with identification of feature importance in different models to select the best model. The dataset includes various attributes of customers who have taken loans, such as account balances, credit history, loan details, and personal demographics.",
                            }
            },
            {
                "semester": "Spring 2025",
                "course_name": "INFO 557: Neural Networks",
                "grade": "A",
                "skills": ["Linear Model", "Activation functions", "Back Propagation", "MSE", "BCE", "Generalization", "Optimization", "SGD", "BGD", "Regulatization", "Dropouts", "Early Stopping", "CNN", "RNN", "GRU", "LSTM"],
                "project":{
                                "title": "Assesment",
                                "description": "Several assignments and quizzes on Back Propagation, Convolutional Neural Networks and Recurrent neural networks were assessed",
                            }
            },
            {
                "semester": "Fall 2024",
                "course_name": "INFO 526: Data Analysis and Visualzation",
                "grade": "A",
                "skills": ["Summary Statistics", "Distribution Plots", "Sonification", "Venn Diagrams", "Concept Maps", "Flow Diagrams","Polar Plots", "Dashboard Visualization"],
                "project":{
                                "title": "Assessment",
                                "description": "Weekly assignments and quizzes on Data visualization techniques",
                            }
            },            
            {
                "semester": "Fall 2024",
                "course_name": "INFO 523: Data Mining and Discovery",
                "grade": "A",
                "skills": ["EDA","Outliers", "Normalization", "Discretization", "Hypothesis Testing", "PCA", "Regression", "Random Forest", "Boosting", "Bagging", "K-means", "K-Medoids","Hierarchical Clustering", "BIRCH", "DBSCAN", "OPTICS", "Association", "Data Warehousing","ETL"],
                "project":{
                                "title": "Mining Fatal Police Shooting Data",
                                "description": "This Course included several coding assignments on Fatal Police Shooting data for exploratory data analysis, data pre-processing, data cleaning, transformation, Classification model building (signs of mental illness in victims) and Pattern Mining using various clustering techniques.",
                            }            
            },                        
            {
                "semester": "Summer 2024",
                "course_name": "INFO 505: Foundations of Information",
                "grade": "A",
                "skills": ["Web Scraping", "Data Collection", "Data Wrangling", "Data Visualization"],
                "project":{
                                "title": "Job Market analysis and  web scraping of TMDB",
                                "description": "This project has 2 sections. The first section involved scraping job market data from various online sources to explore and analyze the open positions relatedto the jobs in the field of data or machine learning in a particular region. The secong part involved web scraping of The Movie Database (TMDB) to build a database of movie related information",
                            }
            }
        ]

# -----------------------------
# Styling (LinkedIn-like profile block)
# -----------------------------
st.markdown(
            """
            <style>
            /* Make the app feel cleaner */
            .block-container { padding-top: 0rem; padding-bottom: 2.5rem; }
            
            /* Fixed header wrapper */
            .fixed-header-wrap {
                                position: fixed;
                                top: 0;
                                left: 0;
                                right: 0;
                                z-index: 999999;
                                background: white;
                                border-bottom: 1px solid rgba(0,0,0,.12);
                                padding: 10px 18px;
                                }
                                
            .fixed-header-grid h2 { font-size: 1.3em;
                                    font-weight: 700;}

            /* Layout: logo | centered title | spacer */
            .fixed-header-grid {
                                display: grid;
                                grid-template-columns: 360px 1fr 180px;
                                align-items: center;
                                width: min(1200px, 100%);
                                margin: 0 auto; /* center the header contents */
                                }    
                                            
            /* University logo */
            .uni-logo {
                        max-height: 70px;
                        width: auto;
                        margin-left: 12px;
                        }

            /* Profile card */
            .profile-card {
                            border: 1px solid rgba(0,0,0,.10);
                            border-radius: 16px;
                            overflow: hidden;
                            background: white;
                            }

            /* Cover */
            .cover {
                    height: 210px;
                    background: linear-gradient(135deg, rgba(0,109,255,.20), rgba(110,231,255,.25));
                    position: relative;
                    }
                    
            .cover img {
                        width: 100%;
                        height: 210px;
                        object-fit: cover;
                        display: block;
                        }

            /* Avatar overlaps cover */
            .avatar-wrap {
                            position: relative;
                            padding: 0 22px;
                            }
            .avatar {
                        width: 160px;
                        height: 160px;
                        border-radius: 999px;
                        border: 5px solid white;
                        object-fit: cover;
                        position: absolute;
                        top: -80px;
                        left: 50px;
                        background: #f3f4f6;
                        }

            /* Info section */
            .info {
                    padding: 84px 22px 18px 22px; /* space for avatar overlap */
                    }
            .name {
                    font-size: 36px !important;
                    font-weight: 900;
                    margin: 0;
                    }
            .meta {
                    color: rgba(0,0,0,.70);
                    margin-top: 6px;
                    line-height: 1.35;
                    }
            .links a {
                        margin-right: 12px;
                        font-weight: 700;
                        text-decoration: none;
                        }
            .links a:hover { text-decoration: underline; }

            /* Course expander inner cards */
            .small-muted {
                            color: rgba(0,0,0,.65);
                            font-size: 0.95rem;
                            }
            .external-link::before {
                                    content: "↗ ";
                                    font-size: 0.9em;
                                    margin-right: 2px;
                                    }
/* Give the page room for your fixed header (important on mobile too) */
            .block-container {
                                padding-top: 110px;   /* was 0rem - causes overlap/collapse feeling */
                                padding-bottom: 2.5rem;
                              }
            
            /* Make header less “rigid” */
            .fixed-header-wrap {
                                position: sticky;   /* better than fixed on mobile */
                                top: 0;
                              }
            
            /* Responsive overrides */
            @media (max-width: 768px) {
            
            /* Make Streamlit content not hug the edges */
            .block-container {
              padding-left: 0.75rem;
              padding-right: 0.75rem;
              padding-top: 90px;
            }
          
            /* Header becomes compact */
            .fixed-header-grid {
              grid-template-columns: 56px 1fr 0px; /* no right spacer */
              width: 100%;
              gap: 10px;
            }
          
            .fixed-header-grid h2 {
              font-size: 1.0em;
              line-height: 1.2;
              margin: 0;
            }
          
            .uni-logo {
              max-height: 40px;
              margin-left: 0;
            }
          
            /* Cover + avatar scale down */
            .cover { height: 140px; }
            .cover img { height: 140px; }
          
            .avatar {
              width: 110px;
              height: 110px;
              top: -55px;
              left: 18px;
              border-width: 4px;
            }
          
            .info {
              padding: 66px 16px 16px 16px; /* less padding, fits mobile */
            }
          
            .name {
              font-size: 26px !important;
            }
          
            /* Pills wrap more nicely */
            span[style*="border-radius:999px"] {
              font-size: 0.78rem !important;
              padding: 5px 8px !important;
            }
          }

                                    
            </style>
            """,
                unsafe_allow_html=True,
            )

# -----------------------------
# Header row with university logo at left
# -----------------------------
st.markdown(
f"""<div class="fixed-header-wrap">
  <div class="fixed-header-grid">
    <div><img src="{img_uri('university_logo.png')}" class="uni-logo"/></div>
    <div><h2>{PROFILE["portfolio_title"]}: {PROFILE["capsname"]} </h2></div>
    <div></div>
  </div>
</div>""",
unsafe_allow_html=True
)

# -----------------------------
# LinkedIn-style profile + cover layout
# -----------------------------
cover_uri = img_uri("cover.jpeg")
profile_uri = img_uri("profile_picture.jpeg")

cover_html = f'<img src="{cover_uri}" />' if cover_uri else ""
avatar_html = f'<img class="avatar" src="{profile_uri}" />' if profile_uri else ""

transcript_uri = img_uri("unofficial_transcript.pdf")
degree_uri = img_uri("degree.jpeg")  # or degree.jpeg if that's your file

st.markdown(
            f"""
            <div class="profile-card">
            <div class="cover">
                {cover_html if cover_html else ""}
            </div>
            <div class="avatar-wrap">
                {avatar_html if avatar_html else '<div class="avatar"></div>'}
            </div>
            <div class="info">
                <h2 class="name">{PROFILE["name"]}</h2>
                <div class="meta">
                <div><b>{PROFILE["program"]}</b></div>
                <div>{PROFILE["college"]}</div>  
                <div>{PROFILE["year"]}</div>
                <br>
                <div><b>Status:</b> {PROFILE["status"]}</div>
                </div>
                <div class="links" style="margin-top:12px;">
                <div><a href="?doc=transcript" target="_blank" rel="noopener noreferrer" class="external-link">View Transcript</a></div>
                <br>
                <div><a href="mailto:{PROFILE["email"]}">Email</a></div>
                </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
                # <a href="{PROFILE["linkedin"]}" target="_blank">LinkedIn</a>
                # <a href="{PROFILE["github"]}" target="_blank">GitHub</a>
                # <a href="?doc=degree" target="_blank" rel="noopener noreferrer" class="external-link">View Degree</a>


st.write("")  # spacer

# -----------------------------
# Courses section with expanders
# -----------------------------
st.markdown("#### Courses Completed")

for course in COURSES:
    with st.expander(course["course_name"], expanded=True):

        # Top row inside expander: Grade + Skills
        st.markdown(f"**Semester:** {course['semester']}")        
        st.markdown(f"**Grade:** {course['grade']}")
        skills_html = " ".join([pill(s) for s in course["skills"]])
        st.markdown(f"""<div><strong>Skills, Tools & Concepts:</strong>&nbsp&nbsp&nbsp{skills_html}</div>""",unsafe_allow_html=True)
        st.markdown(f"""<div><br><strong>Project/ Assessment:</strong>&nbsp&nbsp {course["project"]["title"]}</div>""",unsafe_allow_html=True)
        st.markdown(
            f"""<div><p class="small-muted">{course["project"]["description"]}</p></div>""",
            unsafe_allow_html=True
        )
# Optional: footer
st.markdown("---")
st.markdown(
    f"<div class='small-muted' style='text-align:center;'>© {PROFILE['name']} • Built with Streamlit</div>",
    unsafe_allow_html=True,
)
