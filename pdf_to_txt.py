from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, XMLConverter, HTMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import BytesIO
import os
import getpass
import glob
import yaml

# loading config params
project_root =  "/mnt/md0/user/swidnickira68812/bertv1"
with open(project_root+"/config.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

path_files = params["converter"]["pdf_path"]

def convert_pdf(directory_pdf,mode="default"):

    #Count numbers of resumes
    def count_numbers_of_cvs(directory_pdf):
        numbers_of_cvs = 0
        for path in os.listdir(directory_pdf):
            if os.path.isfile(os.path.join(directory_pdf, path)):
                numbers_of_cvs += 1
        return numbers_of_cvs

    # Convert each resume from PDF to TXT file
    def convert_pdf(path, format='text', codec='utf-8', password=''):
        rsrcmgr = PDFResourceManager()
        retstr = BytesIO()
        laparams = LAParams()
        if format == 'text':
            device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        elif format == 'html':
            device = HTMLConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        else:
            raise ValueError('provide format, either text or html!')
        fp = open(path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                      check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue().decode()
        fp.close()
        device.close()
        retstr.close()
        return text

    if mode == "bertv4":
        # Save path of job description
        jobdescription_path = glob.glob(directory_pdf+"stellenbeschreibung/"+'*.pdf')

        # Check if folder only consists of job description
        if len(jobdescription_path) == 0:
            raise ValueError("Please put the job description for similarity check into the folder: \n", directory_pdf+"stellenbeschreibung")
        if len(jobdescription_path) > 1:
            raise ValueError("Please remove all pdf files and only leave the job description for similarity check inside the folder: \n", directory_pdf+"stellenbeschreibung")
    
        print("\n Found job description data for converting process \n")

        # Convert as first .txt file the job description 
        print(jobdescription_path[0])
        cv_in_txt = convert_pdf(jobdescription_path[0], 'text')
        file = open(directory_pdf + '0.txt', 'w')
        print("\n- - - job description have been converted to .txt file with the name: 0.txt")
        
    # Save file paths of resumes
    filepaths = glob.glob(directory_pdf+'*.pdf')


    if len(filepaths) == 0:
        raise ValueError("Please put some resume pdf files into the given directory path:",
                         directory_pdf)

    else:
        # Convert all resumes in an iteration as long as all .txt files are generated
        for idx, x in enumerate(filepaths):
            cv_in_txt = convert_pdf(filepaths[idx], 'text')
            file = open(directory_pdf + str(int(idx)+1) + '.txt', 'w')
            file.write(cv_in_txt)
            file.close()

        print("\n- - - PDF files has been all converted")

