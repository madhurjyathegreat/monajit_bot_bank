import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
import pandas as pd
import altair as alt


# Install pyngrok (run this in a separate cell first)
# !pip install pyngrok

# --- Setup and Initialization ---

# Set your Groq API key
os.environ["GROQ_API_KEY"] = "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" # Replace with your key

# Load embeddings model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Load and process CSV data
@st.cache_data
def load_and_process_data(file_path):
    loader = CSVLoader(file_path=file_path, csv_args={'fieldnames': ['Account Number', 'Customer Name', 'Account Type','Balance','Last Transaction Date','Transaction Type','Transaction Amount','Branch']})
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

# Load data and create vectorstore
file_path = "bank_data.csv"  # Ensure this path is accessible within the Streamlit app
data = load_and_process_data(file_path)

@st.cache_resource
def create_vectorstore(_docs, _embeddings):
    return FAISS.from_documents(_docs, _embeddings)

vectorstore = create_vectorstore(data, embeddings)

# Initialize Groq LLM
@st.cache_resource
def load_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile"
    )

llm = load_llm()

# Create Retrieval QA chain
@st.cache_resource
def create_qa_chain(_llm, _vectorstore):  # Changed 'vectorstore' to '_vectorstore'
    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever()
    )

qa = create_qa_chain(llm, vectorstore)

# --- Streamlit UI ---

st.title("Bank Data Chatbot")
st.write("Ask questions about your bank data.")

# Display a sample of the data

st.subheader("Glance at the Data:")
df=pd.read_csv('bank_data.csv')
st.dataframe(df)


# Charts
# Charts
if df is not None:
    st.subheader("Data Visualizations")

    # Bar chart of Balance per Customer
    chart_balance_per_customer = alt.Chart(df).mark_bar().encode(
        x=alt.X('Balance:Q', title='Balance Amount'),
        y=alt.Y('Customer Name:N', sort='-x', title='Customer Name'),  # Sort by balance descending
        tooltip=['Customer Name', 'Balance']
    ).properties(
        title='Balance Amount per Customer'
    )
    st.altair_chart(chart_balance_per_customer, use_container_width=True)

    st.markdown("---")

    # Bar chart of Account Type distribution (as before)
    account_type_counts = df['Account Type'].value_counts().reset_index()
    account_type_counts.columns = ['Account Type', 'Count']
    chart_account_type = alt.Chart(account_type_counts).mark_bar().encode(
        x='Account Type:N',
        y='Count:Q',
        tooltip=['Account Type', 'Count']
    ).properties(
        title='Distribution of Account Types'
    )
    st.altair_chart(chart_account_type, use_container_width=True)

    # Bar chart of Average Balance per Account Type (as before)
    avg_balance_per_type = df.groupby('Account Type')['Balance'].mean().reset_index()
    chart_avg_balance = alt.Chart(avg_balance_per_type).mark_bar().encode(
        x='Account Type:N',
        y='Balance:Q',
        tooltip=['Account Type', alt.Tooltip('Balance', format=',.2f')]
    ).properties(
        title='Average Balance per Account Type'
    )
    st.altair_chart(chart_avg_balance, use_container_width=True)
else:
    st.info("No data available to generate charts.")


# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you with your bank data?"}]

# Display chat messages from history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Your question"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa.run(prompt)
            st.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
