# Figure Skating Scoring

Программа для автоматического анализа и оценивания видеозаписи выступления фигуриста с помощью нейронных сетей на основе статьи [Score Figure Skating Sport Videos](https://arxiv.org/pdf/1802.02774.pdf).

Архитектура модели:
<img src="https://drive.google.com/file/d/1X8yBeqDjvkMYHdF4dI-9Ka_Ci-0j4fxe/view?usp=sharing">

Модель состоит из четырёх частей:
1. Получение признаков действий C3D из видеозаписи
2. S-LSTM, использующий механизм self-attention и позволяющий
получить более компактное представление признаков, по которым
делается предсказание
3. M-LSTM, использующий 3 уровня свёрточных сетей, к выходным
данным которых применяется LSTM с пропуском состояний, что
позволяет получить 3 предсказания
4. Объединение 4-х предсказаний, на основе которых последний
полносвязный слой выдаёт финальное предсказание оценок TES и PCS

Датасет видеозаписей выступления фигуристов, представленный уже в виде признаков C3D можно загрузить по этой [ссылке](https://drive.google.com/file/d/1FQ0-H3gkdlcoNiCe8RtAoZ3n7H1psVCI/view)

Для самостоятельного извлечения признаков понадобятся [веса](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle) и [оригинальный датасет](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle)

Результаты корреляции Спирмена

 Модель | TES | PCS |
 :---: |:---:| :---:|
 LSTM | 0,58 | 0,73 |
 Двунапр. LSTM | 0,54 | 0,70 |
 S-LSTM | **0,65** | 0,73 |
 M-LSTM | 0,62 | 0,74 |
 S-LSTM+M-LSTM | 0,62 | **0,74** |


