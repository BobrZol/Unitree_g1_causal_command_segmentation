# Запуск эксперимента

Из корня проекта выполните одну команду:

source /home/sasha/Documents/venv/bin/activate && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 main.ipynb && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 comparison_methods.ipynb

Команда последовательно формирует PCA-датасеты в dataset/, затем обучает и сравнивает четыре метода. Итоговые таблицы CSV и графики сохраняются в results/.


python comparison_methods.py --records 2_9 2_10 2_12 2_14 2_15 2_16


То что я тестил с аугментироваными движениями:
python comparison_methods.py --records 2_9 2_10 2_12 3_5 2_15 3_6