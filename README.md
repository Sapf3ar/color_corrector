# :notebook_with_decorative_cover: Table of Contents
<!-- Table of Contents -->
- [Building project](#compass-steps-to-run-code)
- [Setting up models](#gear-link-to-weights)
- [Inferencing video](#movie-camera-single-video-inference)

<!-- Building project -->
### 🧭 Steps to run Code

- Крайне рекомендуем использовать виртуальное окружение
```bash
python -m venv env
source env/bin/activate
```
- Install the dependecies
```bash
pip install -r requirements.txt
```
Запускаем приложение:
```bash
streamlit run create_origin.py
```

Описание вкладок:
  - На первой вкладке после загрузки модели вы можете создать и сохранить эталон
  - На второй вкладке, приложение уже будет искать созданный эталон и вы сможете использовать эталон для экстраполяции цветового распределения
