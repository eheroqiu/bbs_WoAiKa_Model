
# coding: utf-8

# In[1]:

import pandas as pd
import jieba as jb
import numpy as np
#爬虫得到的论坛数据
f=pd.read_csv('C:/Users/Qiu-qiu/ccc_title/pufa_title.csv')
#加入中文停用词筛选器
ty=pd.read_csv('D:/anaconda/Lib/site-packages/jieba/chinese_stopword.txt')
ty_list=np.append(np.array(ty), ['','浦发','信用卡','卡','白','银行','浦发银行','浦发信用卡','楼主'])
#将停用词做成词典
stopwords={}
for word in ty_list:
    stopwords[word.strip().decode('utf-8')]=word.strip().decode('utf-8')
stopwords[u'']=u''
stopwords[u' ']=u' '
stopwords[u' ']=u' '

#有些评论数据不太干净, 所以需要清理下
title=f.drop_duplicates(['title'])['title']



# In[2]:

#添加自定义分词词典 要注意的是词典必须以UTF-8的字符编码格式存储
jb.load_userdict('D:/anaconda/Lib/site-packages/jieba/sqdbccc_dict.txt')
jb.load_userdict('D:/anaconda/Lib/site-packages/jieba/ccc_dict.txt')
jb.load_userdict('D:/anaconda/Lib/site-packages/jieba/web_words_dict.txt')

#分词
title_Cut_After=[]
for i in range(len(title)):
    seg=jb.cut_for_search(title.iloc[i])
    aft=''
    for sg in seg:
        if sg in stopwords:
            continue
        aft=aft+','+sg
    if aft!='':
        title_Cut_After.append(aft.encode('utf8'))


# In[3]:

pd.DataFrame(title_Cut_After)[:5000].to_csv('C:/Users/Qiu-qiu/ccc_title/before_tagged.csv',encoding ='utf-8')


# In[3]:

#将分词后的结果转换成 corpus语料库dict_text dict_title
from gensim import corpora
title_af=[[word for word in title_Cut_After_xx.split(',') if word not in ty_list] for title_Cut_After_xx in title_Cut_After]
#store the dict , for tuture reference
dict_title=corpora.Dictionary(title_af)
dict_title.save('C:/Users/Qiu-qiu/Documents/NLP/dict/pf_title.dict')


# In[4]:

#语料库对象 corpus 
from  gensim.models import ldamodel
corpus_title=[dict_title.doc2bow(title) for title in title_af]
lda_title=ldamodel.LdaModel(corpus=corpus_title, num_topics=50,id2word=dict_title)


# In[9]:

for i in range(0,50):
    print lda_title.print_topic(i,topn=10)


# In[ ]:



