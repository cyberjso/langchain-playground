import requests

from tempfile import NamedTemporaryFile

def pdf_path_sample():
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    response = requests.get(pdf_url)
    response.raise_for_status()

    with NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name