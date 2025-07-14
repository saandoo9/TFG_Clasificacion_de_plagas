# TFG_Clasificacion_de_plagas
Repositorio que contiene el código para realizar las pruebas de mi TFG

En el archivo gradcam.py se encuentra el código necesario para realizar visualización,
localización y recortes utilizando un modelo de clasificación y la técnica Grad-CAM.

El archivo model.py contiene la CNN diseñada ad-hoc para el problema de clasificación
binaria, los clasificadores para los modelos preentrenados y funciones para entrenar modelos.

El archivo utils.py contiene estructuras y herramientas para cargar conjuntos de datos.

El archivo clip_labels.py contiene los distintos tipos de descripciones utilizadas como
etiquetas para evaluar el rendimiento de CLIP al clasificar las imágenes de test.
