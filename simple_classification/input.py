from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
text = 'It is a sunny day'
# 可以考虑后面选用一个更好的embedding方法
# CountVectorizer  得到的是一维，长度 13656 的向量
text_vector = vectorizer.transform([text]).toarray()