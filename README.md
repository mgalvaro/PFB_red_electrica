# PFB_red_electrica

## 🎯 Project Objective
The goal was to build a fully functional interactive web dashboard using Streamlit that enables users to monitor and analyze real-time and historical data from the Spanish Electric Grid via the official API provided by Red Eléctrica de España (REE). Here is the link to it: https://www.ree.es/en/datos/apidata.
The app offers deep insights into:
- Electricity demand
- Energy generation by source
- Energy balance
- Interregional and international energy exchanges
- Forecasting of future demand

## 📚 Stack: 
Python · Streamlit · Pandas · SQL · Plotly · REE API · TensorFlow/Keras · Scikit-learn

## 🛠️ Use instructions / Instrucciones de uso
🇬🇧
🛠️ How to use the app:
1. The app loads the data from a MySQL local database that has to be created and populated. Go to src > database and execute the following files:
    - "1. CREACION_BBDD.ipynb" -> Creates the DB
    - "2. POBLACION_bbdd.ipynb" -> Populates the DB

2. Once the DB is created and filled up, the app is ready to be used. In the terminal, execute the file "main.py" located in the directory "StreamlitApp":
    cd StreamlitApp
    streamlit run main.py
<br><br>
🇪🇸
🛠️ Cómo usar la app:
1. La app carga los datos desde una base local que hay que crear y poblar. En src > database, abrir y ejecutar estos archivos para crear la base de datos:
    - "1. CREACION_BBDD.ipynb".
    - "2. POBLACION_bbdd.ipynb".
2. Con la base creada y poblada, ya puede usarse la app. En la terminal, ejecutar el archivo main.py en la carpeta StreamlitApp:
    cd StreamlitApp
    streamlit run main.py

## Demo & details
- Video demo: https://www.youtube.com/watch?v=C655JYUN52M
- Project details: https://alvarodsci.wixsite.com/alvaro-mejia/post/energy-insights-dashboard-visualizing-the-spanish-electric-grid
