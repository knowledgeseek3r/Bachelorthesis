import csv
from os import path
import textsimilarity
import yaml

# loading config params
project_root =  "/mnt/md0/user/swidnickira68812/bertv1"
with open(project_root+"/config.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

path_save = params["converter_txt2csv"]["csv_path"]

#Converts txt file resume (=sentences) to csv file resume with defined column heading
def create_csv(sentences, column_heading):

    with open(path_save,'w') as result_file:
        wr = csv.writer(result_file, dialect='excel')

        wr.writerow([column_heading])

        for x in sentences:
            wr.writerow([x])

create_csv(textsimilarity.get_sentences(""), "lebenslauf")