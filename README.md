# PENTING
*Notebook* ini dibuat menggunakan *virtual environment* dengan versi **Python 3.8**. 
## VSCode *Interpreter*
Jika, mengalami kesulitan dalam mengatur interpretasi dalam vscode:
### opsi 1
- buka pengaturan dalam *file* `.vscode/settings.json`
- ubah nilai dari `python.pythonPath` menjadi lokasi binari python yang digunakan 
### opsi 2
Hapus `.vscode` dan bangun lingkungan secara manual
## *Dependencies*
Instalasi dapat dilakukan dengan perintah berikut:
```bash
pip install -r requirements.txt
```
Jika terjadi kegagalan instalasi dengan menggunakan *requirements.txt*. Install secara manual:
```bash
pip install nltk sastrawi pandas numpy
```
Setelah instalasi, jangan lupa untuk mengunduh data nltk "punkt" dan "stopwords"
```
nltk.download('punkt')
nltk.download('stopwords')
```
# TENTANG PROYEK
gunakan ```help(nama_fungsi)``` dalam python untuk informasi mengenai masing-masing fungsi
## *preprocessing*
- fungsi-fungsi normalisasi kalimat. daftar fungsi:
- rem_url(string)
- rem_num(string)
- tokenize(string)
- rem_punc(string)
- rem_stop(string)
- stemm(string)
- freqs(string)
- normalisasi(string)
## *sentiment*
fungsi-fungsi kalkulasi naive-bayes. daftar fungsi
- kamus_freq(teks, label)
- train_nbc(kamus_freq, train_x, train_y)
- predict_nbc(string, logprior, loglikelihood)
- test_nbc(test_x, test_y, logprior, loglikelihood)

## *tools*
*script* yang digunakan untuk mengumpulkan dan mamnipulasi data. *docstring* tidak tersedia.
- googlenews.py 
    <br>news scrapper, dependecies: newspaper3k, googlenews, pandas, re
- twitter.py
    <br>twitter scraper, dependency: twint
- twittranlate.py
    <br>translasi hasil `twitter.py`, dependencies: pandas, translatepy
- prepcsv.py
    <br>**jangan digunakan** lingkungan interaktif untuk memanipulasi hasil web scrapping