# Запуск эксперимента

Из корня проекта выполните одну команду:

source /home/sasha/Documents/venv/bin/activate && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 main.ipynb && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 comparison_methods.ipynb

Команда последовательно формирует PCA-датасеты в dataset/, затем обучает и сравнивает четыре метода. Итоговые таблицы CSV и графики сохраняются в results/.


python comparison_methods.py --records rec_2_9_chr01_MAYA rec_2_10_chr01_MAYA rec_2_12_chr01_MAYA rec_2_14_chr01_MAYA rec_2_15_chr01_MAYA rec_2_16_chr01_MAYA


То что я тестил с аугментироваными движениями:
python comparison_methods.py --records rec_2_9_chr01_MAYA rec_2_10_chr01_MAYA rec_2_12_chr01_MAYA rec_3_5_chr01_MAYA rec_2_15_chr01_MAYA rec_3_6_chr01_MAYA

Имена записей передаются целиком, без пути и расширения. Добавляются только .bvh и .txt.
Порядок: первые три записи - train, четвёртая - conf, остальные - test.
В Python используйте список или кортеж, не множество set.
