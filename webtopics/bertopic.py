import numpy as np
import pandas as pd
from bertopic import BERTopic

data = pd.read_parquet('../data/marketing_sample_walmart.parq.gzip')


products = data['Product Name'].to_list()
display(products[:10])


model = BERTopic(
    top_n_words=5,
    n_gram_range=(1,3),
    nr_topics='auto'
)
topics, probs = model.fit_transform(products)