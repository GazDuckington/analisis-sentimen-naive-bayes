import re
import pandas as pd
from newspaper import Article
from GoogleNews import GoogleNews

gn = GoogleNews(lang='id', period='7d')

gn.search('jokowi')
isi = []
tk = []
for i in range(0,2):
    gn.get_page(i)
    u = gn.get_links()
    t = gn.get_texts()

for n in u:
    art = Article(n)
    art.download()
    art.parse()
    c = art.text.lower()
    isi.append(c)
    d = re.sub(pattern = "[^\w\s]", repl = "", string = c).split()
    tk.append(d)

df1 = pd.DataFrame({'url':[x for x in u],
    'title':[z for z in t],
    'content':[l for l in isi],
    'tokens': [g for g in tk]})
df1.to_csv('gnews2.csv')
