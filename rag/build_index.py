from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# load docs
text = open("../data/ckd_docs.txt").read()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([text])

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(docs, embeddings)
db.save_local("../vectorstore/faiss_index")