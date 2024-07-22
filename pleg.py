from tkinter import *
from tkinter import ttk, messagebox, filedialog
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Function to read text files from a directory
def read_documents_from_files(directory_path, filenames, filetype):
    documents = []
    for filename in filenames:
        filepath = os.path.join(directory_path, filename)
        if filetype == ".txt":
            with open(filepath, 'r', encoding='utf-8') as file:
                documents.append(file.read())
        elif filetype == ".pdf":
            doc = fitz.open(filepath)
            full_text = []
            for page in doc:
                full_text.append(page.get_text())
            documents.append('\n'.join(full_text))
    return documents

def select_directory():
    directory_path = filedialog.askdirectory()
    if directory_path:
        directory_entry.delete(0, END)
        directory_entry.insert(0, directory_path)
        list_files(directory_path)

def list_files(directory_path):
    file_listbox.delete(0, END)
    global available_files
    file_type = filetype_var.get()
    available_files = [f for f in os.listdir(directory_path) if f.endswith(file_type)]
    if not available_files:
        messagebox.showerror("Error", f"No {file_type} files found in the directory {directory_path}.")
    else:
        for filename in available_files:
            file_listbox.insert(END, filename)

def compare_files():
    selected_indices = file_listbox.curselection()
    if not selected_indices:
        messagebox.showwarning("Warning", "No files selected for comparison.")
        return

    selected_files = [available_files[i] for i in selected_indices]
    directory_path = directory_entry.get()
    file_type = filetype_var.get()
    
    documents = read_documents_from_files(directory_path, selected_files, file_type)
    preprocessed_docs = [preprocess_text(doc) for doc in documents]

    # Check for empty documents
    non_empty_docs = [doc for doc in preprocessed_docs if doc.strip() != ""]
    if not non_empty_docs:
        messagebox.showerror("Error", "All selected documents are empty after preprocessing.")
        return
    
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(non_empty_docs)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        cosine_sim_percentages = cosine_sim * 100
        display_similarity_scores(cosine_sim_percentages, selected_files)
    except ValueError as e:
        messagebox.showerror("Error", str(e))

def display_similarity_scores(cosine_sim, filenames):
    result_text.delete(1.0, END)
    num_docs = len(filenames)
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            result_text.insert(END, f"Similarity between {filenames[i]} and {filenames[j]}: {cosine_sim[i, j]:.2f}%\n")

root = Tk()
root.title("Text Similarity Application")
root.geometry("700x600")

# Define the style
style = ttk.Style()
style.configure("TLabel", background="#f0f0f0", foreground="#333")
style.configure("TButton", background="green", foreground="black", padding=6)
style.configure("TEntry", padding=5)
style.configure("TFrame", background="brown")
style.configure("TListbox", background="white", foreground="black")

# Frame for directory selection
frame = ttk.Frame(root, padding="10 10 10 10")
frame.grid(row=0, column=0, sticky=(N, S, E, W))

# Directory selection
directory_label = ttk.Label(frame, text="Directory:")
directory_label.grid(row=0, column=0, padx=5, pady=5, sticky=W)
directory_entry = ttk.Entry(frame, width=50)
directory_entry.grid(row=0, column=1, padx=5, pady=5)
directory_button = ttk.Button(frame, text="Browse", command=select_directory)
directory_button.grid(row=0, column=2, padx=5, pady=5)

# File type selection
filetype_var = StringVar(value=".txt")
filetype_label = ttk.Label(frame, text="File Type:")
filetype_label.grid(row=1, column=0, padx=5, pady=5, sticky=W)
filetype_txt_radio = ttk.Radiobutton(frame, text=".txt", variable=filetype_var, value=".txt", command=lambda: list_files(directory_entry.get()))
filetype_txt_radio.grid(row=1, column=1, padx=5, pady=5, sticky=W)
filetype_pdf_radio = ttk.Radiobutton(frame, text=".pdf", variable=filetype_var, value=".pdf", command=lambda: list_files(directory_entry.get()))
filetype_pdf_radio.grid(row=1, column=2, padx=5, pady=5, sticky=W)

# File list
file_listbox = Listbox(frame, selectmode=MULTIPLE, width=60, height=10)
file_listbox.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

# Compare button
compare_button = ttk.Button(frame, text="Compare Selected Files", command=compare_files)
compare_button.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

# Result text
result_text = Text(frame, width=80, height=20, background="#e0e0e0", foreground="#000")
result_text.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

# Add padding to all widgets
for widget in frame.winfo_children():
    widget.grid_configure(padx=10, pady=5)

root.mainloop()


