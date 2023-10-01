# Created by Ansari at 7/28/2019
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import pandas as pd
from keras.preprocessing import sequence, text

data = pd.read_csv ('data/quora_duplicate_questions.tsv', sep='\t')
y = data.is_duplicate.values

tk = text.Tokenizer (nb_words=200000)

max_len = 40
tk.fit_on_texts (list (data.question1.values.astype (str)) + list (data.question2.values.astype (str)))
x1 = tk.texts_to_sequences (data.question1.values.astype (str))
x1 = sequence.pad_sequences (x1, maxlen=max_len)

x2 = tk.texts_to_sequences (data.question2.values.astype (str))
x2 = sequence.pad_sequences (x2, maxlen=max_len)

new_model = load_model ("./output/weights_845.h5")

x1_test = x1[363861:]
x2_test = x2[363861:]
y_test = y[363861:]


#checkpoint = ModelCheckpoint ('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)
print(new_model.metrics_names)
loss, acc = new_model.evaluate([x1_test, x2_test,x1_test, x2_test,x1_test, x2_test], y=y_test, batch_size=100)
print(f'Loss:{loss}')
print(f'Acc:{acc}')