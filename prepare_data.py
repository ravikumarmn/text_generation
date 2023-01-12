import pandas as pd
import glob
import config
import re

def get_all_file_path(path,file_format = "*.csv"):
    dir = path + file_format
    csv_file_names = glob.glob(dir)
    return csv_file_names


def create_dataframe(file_names):
    df = list()
    for file_path in file_names:
        d = pd.read_csv(file_path)
        df.append(d)
    return pd.concat(df)


def clean_text(text):
    # Remove puncuation,stopwords,only words,lowercase,lematization
    text = re.sub(r'(\#\w+)'," ",text)
    text = re.sub(r"br","",text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub('[^a-zA-Z]+',' ',text)
    text = " ".join([wordnet_lemmatizer.lemmatize(x) for x in text.split() ])
    text = text.lower()
    return text.strip()    


if __name__=="__main__":
    csv_file_names = get_all_file_path(config.data_dir)
    df = create_dataframe(csv_file_names)
    df['label'] = df['sentiment'].map(config.labels)

    print()
