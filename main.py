import streamlit as st
import os
import hashlib
import sqlite3  # For the database
from typing import Optional, Callable
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
import pandas as pd
import altair as alt
import tempfile  # For handling uploaded files
# Set your Groq API key
os.environ["GROQ_API_KEY"] = "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" # Replace with your key


# --- Database Setup (SQLite) ---
# This sets up a simple SQLite database to store user info.  For a production app,
# consider a more robust database like PostgreSQL, and use a proper ORM (like
# SQLAlchemy) for security and maintainability.
DATABASE_NAME = "users.db"

def get_db_connection():
    """Gets a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row  # So we can access columns by name
    return conn

def init_db():
    """Initializes the database (creates the user table if it doesn't exist)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            name TEXT,
            email TEXT
        )
    """)
    conn.commit()
    conn.close()

# --- Security ---
# VERY IMPORTANT:  Never store passwords in plaintext!  Always hash them
# using a strong algorithm like bcrypt or Argon2.  For this example, we're
# using SHA256, which is better than plaintext, but not ideal for real-world
# password storage.  Use bcrypt or Argon2.  Streamlit does not include these
# by default, so you would need to add them to your requirements.txt.
# (e.g.,  `pip install bcrypt`)

def hash_password(password: str) -> str:
    """Hashes a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password: str, password_hash: str) -> bool:
    """Checks if a password matches a hash (using SHA256)."""
    return hash_password(password) == password_hash

# --- User Management Functions ---

def get_user(username: str) -> Optional[dict]:
    """Retrieves a user from the database by username."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user if user else None  # Return None if no user found

def create_user(username: str, password: str, name: str, email: str) -> bool:
    """Creates a new user in the database. Returns True on success, False on failure."""
    conn = get_db_connection()
    cursor = conn.cursor()
    password_hash = hash_password(password)
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, name, email) VALUES (?, ?, ?, ?)",
            (username, password_hash, name, email),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:  # Handle username already exists
        conn.close()
        return False

# --- Authentication Decorator ---
# This is a more advanced technique, but very useful for securing your app.
# It's a function that takes another function as input (the function to be
# protected) and returns a *new* function that includes the login check.
#
# In Streamlit, we check for the session state.
def login_required(func: Callable) -> Callable:
    """Decorator to require login for a Streamlit page/function."""
    def wrapper(*args, **kwargs):
        if "user" in st.session_state:
            func(*args, **kwargs)
        else:
            st.error("Please log in to access this page.")
    return wrapper

# --- Langchain and Data Processing ---

# Load embeddings model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Load and process CSV data from file object
def load_and_process_data(uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        loader = CSVLoader(file_path=temp_file_path, csv_args={'fieldnames': ['Account Number', 'Customer Name', 'Account Type','Balance','Last Transaction Date','Transaction Type','Transaction Amount','Branch']})
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        os.remove(temp_file_path)  # Clean up the temporary file
        return docs
    return None

# Create vectorstore (now takes data as an argument)
@st.cache_resource
def create_vectorstore(_docs, _embeddings):
    if _docs:
        return FAISS.from_documents(_docs, _embeddings)
    return None

# Initialize Groq LLM
@st.cache_resource
def load_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile"
    )

llm = load_llm()

# Create Retrieval QA chain (now takes vectorstore as argument)
@st.cache_resource
def create_qa_chain(_llm, _vectorstore):
    if _vectorstore:
        return RetrievalQA.from_chain_type(
            llm=_llm,
            chain_type="stuff",
            retriever=_vectorstore.as_retriever()
        )
    return None

# --- Streamlit UI ---

def show_login():
    """Displays the login page."""
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = get_user(username)
        if user and check_password(password, user["password_hash"]):
            st.session_state["user"] = {  # Store user info in session state
                "id": user["id"],
                "username": user["username"],
                "name": user["name"],
                "email": user["email"],
            }
            st.success("Logged in successfully!")
            st.session_state["page"] = "home"  # Set the page
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.markdown("---")
    st.subheader("Register")
    new_name = st.text_input("Name")
    new_email = st.text_input("Email")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Register"):
        if create_user(new_username, new_password, new_name, new_email):
            st.success("Registered successfully! Please log in.")
        else:
            st.error("Username already exists.")

@login_required
def show_home():
    """Displays the home page (accessible only after login)."""
    st.title("GEN-AI BOT")
    st.header(f"Welcome, {st.session_state['user']['name']}!")


    # File uploader
    uploaded_file = st.file_uploader("Upload your bank_data.csv file", type=["csv"])

    if uploaded_file is not None:
        # Load and process the uploaded data
        data = load_and_process_data(uploaded_file)

        # Create the vectorstore
        vectorstore = create_vectorstore(data, embeddings)

        # Create the QA chain
        qa = create_qa_chain(llm, vectorstore)

        # Load DataFrame for display and charting (from uploaded file)
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading uploaded CSV: {e}")
            df = None

        # Display a sample of the data
        st.subheader("Glance at the Data:")
        if df is not None:
            st.dataframe(df.head())
        else:
            st.info("No data to display.")

        st.markdown("---")

        # Dynamic Charts
        if df is not None:
            st.subheader("Ask for Custom Charts:")
            chart_prompt = st.text_input("Enter a prompt to generate a custom chart (e.g., 'Show a bar chart of balance by branch'):")
            if chart_prompt:
                with st.spinner("Generating chart..."):
                    try:
                        if "balance by branch" in chart_prompt.lower():
                            balance_by_branch = df.groupby('Branch')['Balance'].sum().reset_index()
                            chart = alt.Chart(balance_by_branch).mark_bar().encode(
                                x='Branch:N',
                                y='Balance:Q',
                                tooltip=['Branch', 'Balance']
                            ).properties(title='Total Balance by Branch')
                            st.altair_chart(chart, use_container_width=True)
                        elif "account type distribution" in chart_prompt.lower():
                            account_type_counts = df['Account Type'].value_counts().reset_index()
                            account_type_counts.columns = ['Account Type', 'Count']
                            chart = alt.Chart(account_type_counts).mark_bar().encode(
                                x='Account Type:N',
                                y='Count:Q',
                                tooltip=['Account Type', 'Count']
                            ).properties(title='Distribution of Account Types')
                            st.altair_chart(chart, use_container_width=True)
                        elif "balance per customer" in chart_prompt.lower():
                            chart = alt.Chart(df).mark_bar().encode(
                                x=alt.X('Balance:Q', title='Balance Amount'),
                                y=alt.Y('Customer Name:N', sort='-x', title='Customer Name'),
                                tooltip=['Customer Name', 'Balance']
                            ).properties(title='Balance Amount per Customer')
                            st.altair_chart(chart, use_container_width=True)
                        elif "average balance by account type" in chart_prompt.lower():
                            avg_balance_per_type = df.groupby('Account Type')['Balance'].mean().reset_index()
                            chart = alt.Chart(avg_balance_per_type).mark_bar().encode(
                                x='Account Type:N',
                                y='Balance:Q',
                                tooltip=['Account Type', alt.Tooltip('Balance', format=',.2f')]
                            ).properties(title='Average Balance per Account Type')
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.warning("Sorry, I can't generate that specific chart yet. Try prompts like 'Show balance by branch', 'Show account type distribution', 'Show balance per customer', or 'Show average balance by account type'.")
                    except Exception as e:
                        st.error(f"Error generating chart: {e}")
        st.markdown("---")

        # Chat Interface (only if file is uploaded and QA chain is ready)
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Upload your bank data to start asking questions."}]

        if qa:
            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Your question"):
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = qa.run(prompt)
                        st.markdown(response)

                        # --- SQL Query Generation Byproduct ---
                        try:
                            from langchain.prompts import PromptTemplate
                            from langchain.chains import LLMChain

                            sql_generation_prompt_template = """Given the following CSV schema:
                            {csv_schema}

                            Generate a SQL query to answer the question: {question}
                            """
                            sql_generation_prompt = PromptTemplate(template=sql_generation_prompt_template, input_variables=["csv_schema", "question"])
                            sql_generation_chain = LLMChain(llm=llm, prompt=sql_generation_prompt)

                            csv_schema = "Account Number (INTEGER), Customer Name (TEXT), Account Type (TEXT), Balance (REAL), Last Transaction Date (TEXT), Transaction Type (TEXT), Transaction Amount (REAL), Branch (TEXT)"
                            sql_query = sql_generation_chain.run(csv_schema=csv_schema, question=prompt)
                            st.subheader("Generated SQL Query:")
                            st.code(sql_query, language="sql")
                            st.session_state["messages"].append({"role": "assistant", "content": response + f"\n\n**Generated SQL Query:**\n```sql\n{sql_query}\n```"})

                        except Exception as e:
                            st.error(f"Error generating SQL query: {e}")
                            st.session_state["messages"].append({"role": "assistant", "content": response + f"\n\n**Error generating SQL query:** {e}"})

                    st.session_state["messages"].append({"role": "assistant", "content": response})
        elif uploaded_file is not None:
            st.info("Processing uploaded file...")

    else:
        st.info("Please log in to access the data analysis features.")

def main():
    """Main function to run the Streamlit app."""
    init_db()  # Initialize the database
    if "user" not in st.session_state:
        st.session_state["page"] = "login"
    if st.session_state["page"] == "login":
        show_login()
    elif st.session_state["page"] == "home":
        show_home()

if __name__ == "__main__":
    main()
