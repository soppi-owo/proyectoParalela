# Organizar directorio
El proyecto usa una carpeta files que contiene el atlasRo y los sujetos, debe ubicarse en la carpeta principal de esta forma\
<img width="434" height="462" alt="image" src="https://github.com/user-attachments/assets/bd01e524-585b-456d-9ca5-dbfbb350c7e9" /> \
El archivo puede encontrarse en el siguiente link: https://drive.google.com/drive/folders/1rdJo1MxeGCyt6ZOBpb5mb8FzdEbD-xVV?usp=sharing
la subcarpeta "sujetos" tiene 2 sujetos de prueba, la carpeta "sujetosTODOS" tiene los usados para la experimentación.\
# Instrucciones de ejecución:
## 1- Instalar dependencias:
Se debe ejecutar ./install.sh para iniciar, notar que la línea de pycuda toolkit está comentada, si se requiere, se puede descomentar.
## 2- Archivo c++
a) Para compilar se utiliza el comando g++ -fopenmp -O3 -g main.cpp -o main -std=c++17\
b) Para ejecutar simplemente ./main
## 3- Archivo Python
a) Activar venv con el comando source .venv/bin/activate\
b) Ejecutar con python main.py
