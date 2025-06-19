# from langchain_community.document_loaders import PyPDFLoader

# def load_pdf(file_path):
#     loader = PyPDFLoader(file_path)
#     return loader.load()


from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import TesseractBlobParser

def load_pdf(file_path):
    # loader = PyMuPDFLoader(file_path=file_path, extract_images=True)
    # documents = loader.load()
    # return documents

    loader = PyMuPDFLoader(
    file_path=file_path,
    mode="page",
    extract_tables="markdown",
    images_inner_format="html-img",
    images_parser=TesseractBlobParser()
    )

    docs = loader.load()
    return docs




