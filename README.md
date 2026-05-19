# FACTOR.IA — Detector de Noticias Falsas
### Proyecto PTIA · Ingeniería en Sistemas · 2026

## Descripción
Aplicación web para detección y clasificación de noticias falsas utilizando
Regresión Logística con vectorización TF-IDF e interfaz Flask.

## Estructura del proyecto
```
fake_news_detector/
├── app.py              # Backend Flask + modelo de IA
├── requirements.txt    # Dependencias Python
├── README.md
└── templates/
    └── index.html      # Interfaz web completa
```

## Instalación

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. (Opcional) Entrenar con el dataset completo de Kaggle
Descarga el dataset desde:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data

Coloca `True.csv` y `Fake.csv` en la carpeta del proyecto, luego descomenta
el bloque de entrenamiento completo en `app.py` (ver comentario "DATASET COMPLETO").

### 3. Ejecutar la aplicación
```bash
python app.py
```

### 4. Abrir en el navegador
```
http://localhost:5000
```

## Cómo funciona

1. El usuario pega el texto de una noticia en la interfaz
2. El backend preprocesa el texto (minúsculas, stopwords, stemming)
3. El vectorizador TF-IDF transforma el texto en vector numérico
4. La Regresión Logística clasifica entre REAL (1) y FAKE (0)
5. El sistema devuelve: etiqueta, probabilidades, señales lingüísticas y palabras clave

## Señales lingüísticas analizadas

**Indicadores de desinformación:**
- Abuso de mayúsculas (CAPS LOCK)
- Exceso de signos de exclamación
- Lenguaje de urgencia (BREAKING, SHARE NOW)
- Frases conspirativas (deep state, wake up)
- Fuentes anónimas (sources say, experts claim)
- Triggers emocionales (shocking, bombshell)

**Indicadores de credibilidad:**
- Fuentes nombradas (Reuters, AP)
- Referencias institucionales
- Lenguaje hedgeado (may, according to)
- Citas de datos y estadísticas

## Métricas del modelo (dataset completo)
- Accuracy: 98.6%
- F1-Score FAKE: 0.987
- F1-Score REAL: 0.985
- AUC-ROC: 0.997

## Autores
- Daniel Patiño Mejia
- Isaac David Burgos Cervantes
