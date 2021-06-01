import os                                            # ubicacion de archivo
import pandas as pd                                  # lectura y creacion de DataFrame
import matplotlib.pyplot as plt                      # permite hacer uso de las graficas
from sklearn.preprocessing import PolynomialFeatures # permite calcular polinomios
from sklearn.linear_model import LinearRegression    # permite calcular regresiones lineales


os.chdir("C:/Users/sebas/Documents/Estudio/U/7mo/AnalisisNumerico/Covid/csv")
dat = pd.read_csv("Covid.csv",header=0)

### DECLARACION DE FUNCIONES ###

# graficar
def graficar(ciudad,x,y,regresion):
    if any(regresion):
        plt.plot(x,regresion,color = "orange", label = "Regresion") # Regresion
        plt.legend()
        
    plt.scatter(x,y)
    plt.xlabel("Semanas")
    plt.ylabel(f"Contagios {ciudad}")
    plt.show()


# prediccion
def prediccion(mes_actual,mes, term, term_x, term_x2, term_x3):
    x = mes_actual + mes
    res = term + ( term_x*x ) + ( term_x2*x**2 ) + ( term_x3*x**3 )
    return int(res)

      
_x = dat.SEMANA.values.reshape(-1,1)
_ybogota = dat.BOGOTA_PACIENTES.values.reshape(-1,1)

_ycolombia = dat.COLOMBIA_PACIENTES.values.reshape(-1,1)
    
# Comportamiento de los datos
graficar('BOGOTA',_x, _ybogota,[])
graficar('COLOMBIA',_x, _ycolombia,[])  


dataContagious_bogota=[]
dataContagious_colombia=[]
dataWeeks=[] 

df = pd.DataFrame()


for i in range(len(dat)):
    if i != 0:        
        wb = dat.BOGOTA_PACIENTES[i] + dataContagious_bogota[i-1]
        wc = dat.COLOMBIA_PACIENTES[i] + dataContagious_colombia[i-1]
        dataContagious_bogota.append( wb )
        dataContagious_colombia.append( wc )
    else:
        dataContagious_bogota.append( dat.BOGOTA_PACIENTES[i] )
        dataContagious_colombia.append( dat.COLOMBIA_PACIENTES[i] )

    dataWeeks.append( dat.SEMANA[i] )

df = df.assign(SEMANA=dataWeeks, PACIENTES_BOGOTA=dataContagious_bogota, PACIENTES_COLOMBIA=dataContagious_colombia)
x = df.SEMANA.values.reshape(-1,1)
yb = df.PACIENTES_BOGOTA.values.reshape(-1,1)
yc = df.PACIENTES_COLOMBIA.values.reshape(-1,1)


graficar('BOGOTA',x,yb,[]) # comportamiento acomulativo Bogota
graficar('COLOMBIA',x,yc,[]) # comportamiento acomulativo Colombia


polynomial_regression = PolynomialFeatures(degree=3)                         # Polinomio de grado 3
x_polynomial = polynomial_regression.fit_transform(x.reshape(-1,1))          # Transformar la entrada en polinomica

linear_regressionBogota = LinearRegression()                                 # Se crea la instancia de LinearRegression
linear_regressionColombia = LinearRegression()                               # Se crea la instancia de LinearRegression
linear_regressionBogota.fit(x_polynomial,yb)                                 # Se le instruye a la regresion lineal que aprenda de los datos (ahora polinomicos)
linear_regressionColombia.fit(x_polynomial,yc)                               # Se le instruye a la regresion lineal que aprenda de los datos (ahora polinomicos)


# Imprimir parametros que ha estimado la regresion lineal
print("\n" + "Coeficientes polinomio - Bogota = " + str(linear_regressionBogota.coef_) + ", b = " + str(linear_regressionBogota.intercept_))
print("Coeficientes polinomio - Colombia = " + str(linear_regressionColombia.coef_) + ", b = " + str(linear_regressionColombia.intercept_) + "\n")


# Se predicen los valores Y para datos usados en el entrenamiento
y_headBogota = linear_regressionBogota.predict(x_polynomial)
y_headColombia = linear_regressionColombia.predict(x_polynomial)


## GRAFICACION DE LA REGRESION
graficar('BOGOTA',x,yb,y_headBogota)
graficar('COLOMBIA',x,yc,y_headColombia)


# Calculo del coeficiente de determinacion R2
r2b = linear_regressionBogota.score(x_polynomial,yb)
r2c = linear_regressionColombia.score(x_polynomial,yc)

print( "Coeficiente de determinacion R2 BOGOTA   = " + str(r2b))
print( "Coeficiente de determinacion R2 COLOMBIA = " + str(r2c) + "\n")
 

# Extrayendo los terminos de los polinomios para enviarlos como parametros
termb    = float( linear_regressionBogota.intercept_[0] )
term_xb  = float( linear_regressionBogota.coef_[0][1] )
term_x2b = float( linear_regressionBogota.coef_[0][2] )
term_x3b = float( linear_regressionBogota.coef_[0][3] )

termc    = float( linear_regressionColombia.intercept_[0] )
term_xc  = float( linear_regressionColombia.coef_[0][1] )
term_x2c = float( linear_regressionColombia.coef_[0][2] )
term_x3c = float( linear_regressionColombia.coef_[0][3] )

mes_actual = len(x)

agosto = 14
diciembre = 32

y_prediccionBogota1   = prediccion(mes_actual, agosto, termb, term_xb, term_x2b, term_x3b )
y_prediccionColombia1 = prediccion(mes_actual, agosto, termc, term_xc, term_x2c, term_x3c )

y_prediccionBogota2   = prediccion(mes_actual, diciembre, termb, term_xb, term_x2b, term_x3b )
y_prediccionColombia2 = prediccion(mes_actual, diciembre, termc, term_xc, term_x2c, term_x3c )


print(f" Prediccion Agosto BOGOTA    = {y_prediccionBogota1}")
print(f" Prediccion Agosto COLOMBIA  = {y_prediccionColombia1}")

print(f" Prediccion Diciembre BOGOTA    = {y_prediccionBogota2}")
print(f" Prediccion Diciembre COLOMBIA  = {y_prediccionColombia2}")


# modelando las variables para graficar la prediccion
nueva_yb = dataContagious_bogota.copy()
nueva_yc = dataContagious_colombia.copy()

for i in range( len(x)+1, len(x)+diciembre+1 ,1 ):
    mes = i - mes_actual
    dataWeeks.append( i )
    dataContagious_bogota.append( prediccion(mes_actual, mes, termb, term_xb, term_x2b, term_x3b ) )
    dataContagious_colombia.append( prediccion(mes_actual, mes, termc, term_xc, term_x2c, term_x3c ) )
    nueva_yb.append(0)
    nueva_yc.append(0)

nd_pred = pd.DataFrame()
nd_pred = nd_pred.assign(SEMANA=dataWeeks, BOGOTA_PACIENTES=dataContagious_bogota, COLOMBIA_PACIENTES=dataContagious_colombia)

x_pred  = nd_pred.SEMANA.values.reshape(-1,1)

y_pred  = nd_pred.BOGOTA_PACIENTES.values.reshape(-1,1)
x_polynomialPrediccion = polynomial_regression.fit_transform(x_pred.reshape(-1,1))
linear_regressionBogota_prediccion = LinearRegression()
linear_regressionBogota_prediccion.fit(x_polynomialPrediccion, y_pred)
y_headBogotaPrediccion = linear_regressionBogota_prediccion.predict(x_polynomialPrediccion)


y_predc  = nd_pred.COLOMBIA_PACIENTES.values.reshape(-1,1)
x_polynomialPrediccionc = polynomial_regression.fit_transform(x_pred.reshape(-1,1))
linear_regressionColom_prediccion = LinearRegression()
linear_regressionColom_prediccion.fit(x_polynomialPrediccionc, y_predc)
y_headColomtaPrediccion = linear_regressionColom_prediccion.predict(x_polynomialPrediccionc)



graficar('BOGOTA Diciembre', x_pred, nueva_yb, y_headBogotaPrediccion)
graficar('COLOMBIA Diciembre', x_pred, nueva_yc, y_headColomtaPrediccion)






