import pickle
files= open('/content/gdrive/MyDrive/Stocktwits/stocktwits_US_TSLA.p','rb')
stocktwits= pickle.load(files)

req_date = stocktwits['date'].str.slice(stop=10)

stocktwits['req_date'] = req_date

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop = stopwords.words('english')

stocktwits['stocktwit'] = stocktwits['stocktwit'].apply(lambda x: " ".join(x.lower() for x in x.split()))

stocktwits['stocktwit'] = stocktwits['stocktwit'].str.replace('[^\w\s]','')

stocktwits['stocktwit'] = stocktwits['stocktwit'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

import pandas as pd

freq = pd.Series(' '.join(stocktwits['stocktwit']).split()).value_counts()[-10:]

freq = list(freq.index)
stocktwits['stocktwit'] = stocktwits['stocktwit'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

# remove too long words
stocktwits['stocktwit'] = stocktwits['stocktwit'].apply(lambda x: " ".join(x for x in x.split() if len(x) < 15))

from textblob import TextBlob

stocktwits['sentiment'] = stocktwits['stocktwit'].apply(lambda x: TextBlob(x).sentiment[0])

stocktwits['followers'].mean()

stocktwits = stocktwits[['req_date', 'stocktwit', 'sentiment', 'followers']].copy()

stocktwits['weight'] = stocktwits1['followers'].apply(lambda x: 0.7 if x > 2357 else 0.3)

date_grp = stocktwits.groupby(['req_date'])

sorted_list_date = list(stocktwits['req_date'].value_counts().index.sort_values())

new_stocktwit = {'date': [], 'sentiment': [], 'count': []}

for date in sorted_list_date:
  new_stocktwit['date'].append(date)
  batch_date = date_grp.get_group(date)
  new_stocktwit['count'].append(batch_date.shape[0])
  big_v = batch_date[batch_date['weight']>0.5]['sentiment'].sum()*0.7/batch_date[batch_date['weight']>0.5].shape[0]
  potato = batch_date[batch_date['weight']<0.5]['sentiment'].sum()*0.3/batch_date[batch_date['weight']<0.5].shape[0]
  new_stocktwit['sentiment'].append(big_v + potato)

intermediate_stocktwit = pd.DataFrame(new_stocktwit)

int_table = pd.read_excel('/content/gdrive/MyDrive/Stocktwits/int.xlsx')

int_stocktwit = intermediate_stocktwit[intermediate_stocktwit['date'].isin(list(int_table['date']))]

df_stocktwit = int_stocktwit.merge(int_table, how='inner', on='date')

tsla = pd.read_csv('/content/gdrive/MyDrive/Stocktwits/TSLA.csv')

tsla.rename(columns={"Date": "date"},inplace=True)

df_stocktwit = df_stocktwit.merge(tsla, how='inner', on='date')

df_stocktwit['sentiment'] -= df_stocktwit['sentiment'].min()
df_stocktwit['count'] -= df_stocktwit['count'].min()
df_stocktwit['interest rate'] -= df_stocktwit['interest rate'].min()
df_stocktwit['daily return'] -= df_stocktwit['daily return'].min()

df_stocktwit['sentiment'] /= df_stocktwit['sentiment'].max()
df_stocktwit['count'] /= df_stocktwit['count'].max()
df_stocktwit['interest rate'] /= df_stocktwit['interest rate'].max()
df_stocktwit['daily return'] /= df_stocktwit['daily return'].max()

df_stocktwit['sentiment'] *= 2
df_stocktwit['count'] *= 2
df_stocktwit['interest rate'] *= 2
df_stocktwit['daily return'] *= 2

df_stocktwit['sentiment'] -= 1
df_stocktwit['count'] -= 1
df_stocktwit['interest rate'] -= 1
df_stocktwit['daily return'] -= 1

data_all = []
for i in range(df_stocktwit.shape[0]-49):
  np_test = df_stocktwit.loc[i:i+49,'sentiment':'daily return'].to_numpy()
  np_test = np_test.T
  data_all.append(np_test)

np_all = np.array(data_all)

tgt = pd.read_csv('/content/gdrive/MyDrive/Stocktwits/TSLA_1.csv')

df_stocktwit_w_tgt = df_stocktwit.merge(tgt, how='inner', on='date')

tgt = df_stocktwit_w_tgt['target'].to_numpy()

np_train = np_all[:600]

np_val = np_all[600:]

tgt_train = tgt[:600]
tgt_val = tgt[600:]

from torch.utils.data import Dataset, DataLoader

class TslaDataset(Dataset):
    def __init__(self, data1, label, transform=None):
        self.tomodel = data1
        self.y = label
        self.transform = transform
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        tomodel = torch.tensor(self.tomodel[idx], dtype = torch.float32)
        y = torch.tensor(self.y[idx])
        return tomodel, y

