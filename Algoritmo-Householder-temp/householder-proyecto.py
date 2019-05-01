import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model

def householder(A,B):
    a = np.array(A)
    b = np.array(B)
    m = a.shape[0]
    n = a.shape[1]
    w = np.zeros((m), dtype = 'f')
    H = np.zeros((m,m), dtype = 'f')
    Q = np.identity(m)
    for j in range(0, n):
        if  max(np.abs(a[j:,j])) == 0:
            break
        tetha   = np.linalg.norm(a[j:,j])
        if a[j,j] != 0.0:
            tetha   = float(tetha*np.sign(a[j,j]))
        w[j:]   = a[j:,j]
        w[j]    = w[j] + tetha
        beta    = 2.0/sum(w[k]**2 for k in range(j,m))
        #a[j,j]  = (-1)*tetha
        a[j:,j] = a[j:,j] - w[j:]            
        for l in range( j + 1, n):
            s           = w[j:].dot(a[j:,l])
            a[j:,l]     = a[j:,l] - w[j:]*s*beta
        s       = w[j:].dot(b[j:])
        b[j:]   = b[j:] - w[j:]*s*beta
    for i in range(0, m):
        for j in range(0, n):
            if a[i,j] != 0:
                a[i, j] *= (-1)
    for i in range(0, m):
        b[i] = b[i]*(-1)
    return a,b

def resolucion(A,b):
    m = A.shape[0]
    n = A.shape[1]
    print "Datos Totales:", m
    x = np.zeros((n),dtype = 'f')
    for j in range(n-1,-1,-1):
        x[j] = ( b[j] - sum(A[j,k]*x[k] for k in range(j+1,n)))/A[j,j]
    return x
#A = np.array([0, -4, 0, 0, -5, -2],dtype='f').reshape(3,2)
#B = np.array([2, 3, 6 ],dtype = 'f')

A = np.array([1, 37.2, 1, 18.2, 1, 13.9, 1, 11.3, 1, 24.9, 1, 25.0, 1, 8.8, 1, 15.9, 1, 27.5, 1, 26.1, 1, 12.5, 1, 26.9, 1, 14.7, 1, 15.4, 1, 9.2, 1, 8.4, 1, 17.6, 1, 58.7, 1, 30.6, 1, 41.7], dtype = 'f').reshape(20,2)
B = np.array([36.4, 15.7, 18.0, 11.1, 23.3, 23.9, 7.0, 18.1, 24.8, 27.4, 11.8, 26.8, 13.6, 11.4, 9.2, 8.4, 17.6, 57.5, 29.1, 38.3], dtype = 'f')
#A = np.array([1, 2, 3, 1, 2, 3, 2, 5, 7],dtype='f').reshape(3,3)
#A = np.array([4, 2, 2, 1, 2, -3, 1, 1, 2, 1, 3, 2, 1, 1, 1, 2],dtype='f').reshape(4,4)
#B = np.array([2, 3, 6, 4],dtype = 'f')
#A = np.array([1, 2, 3, 1, 2, 3, 2, 5, 7],dtype='f').reshape(3,3)
#A = np.array([12, -51, 4, 6, 167, -68, -4, 24, -41],dtype='f').reshape(3,3)
#A = np.array([1, 2, 4, 5, 7, 8],dtype='f').reshape(3 ,2)
print ("A:")
print (A)
print
print ("B:")
print (B)
R,b = householder(A,B)
x = resolucion(R,b)
print
print ("R:")
print (R)
print
print ("Q*b:")
print (b)
print
print ("La ecuacion de regresion es:")
print "y =", x[0], "+", x[1], "* x"
print


# ANALISIS DE REGRESION

# Coeficiente de determinacion
X= np.array([37.2, 18.2, 13.9, 11.3, 24.9, 25.0, 8.8, 15.9, 27.5, 26.1, 12.5, 26.9, 14.7, 15.4, 9.2, 8.4, 17.6, 58.7, 30.6, 41.7])
Xc=np.array([37.2, 18.2, 13.9, 11.3, 24.9, 25.0, 8.8, 15.9, 27.5, 26.1, 12.5, 26.9, 14.7, 15.4, 9.2, 8.4, 17.6, 58.7, 30.6, 41.7]).reshape((-1,1))
print("X:")
print(X)
Y=B
print
print("Y:")
print(Y)
print
print
print('DATOS DEL MODELO REGRESION LINEAL SIMPLE QUE COMPARAREMOS')
print('---------------------------------------------------------')
lr = linear_model.LinearRegression()
lr.fit(Xc, Y)
xc=np.array([lr.intercept_,lr.coef_])

print
print('La ecuacion a comparar:')
print 'y = ',xc[0], '+', xc[1], '* x'
print
print('verificando la calidad de la solucion:')

izq=np.linalg.norm(x-xc)/np.linalg.cond(A)
der=np.linalg.cond(A)*np.linalg.norm(x-xc)
print 'limite izquierda:', izq[0]
print 'limite derecha:', der[0]
print


# y residual
print("'y' residual:")
Yr= x[0]+ x[1]*X
print(Yr)
print

# SSE
SSE=np.sum((Yr-Y)**2)
SSE=round(SSE,9)
print "SSE:", SSE
print

# SST
SST=np.sum((Y-np.mean(Y))**2)
SST=round(SST,9)
print "SST:", SST
print

# SSR
SSR=np.sum((Yr-np.mean(Y))**2)
SSR=round(SSR,9)
print "SSR:", SSR
print

# Coeficiente de determinacion  SSR/SST
# r^2
r_2= SSR/SST
r_2=round(r_2,9)
print "r^2 (coeficiente de determinacion):", r_2
print

# Coeficiente de correlacion
# r_1  signo de la pendiente
r_1=(+1)*math.sqrt(r_2)
r_1=round(r_1,9)
print "r^1 (coeficiente de correlacion):", r_1
print

# Error cuadrado medio (s^2) grado de libertad: 2
s_2=SSE/(len(X)-2)
s_2=round(s_2,9)
print "s^2 (error cuadrado medio):", s_2
print

# Desviacion estandar de la estimacion
s_1=math.sqrt(s_2)
s_1=round(s_1,9)
print "s^1 (desviacion estandar de la estimacion):", s_1
print

# SST = STC ; SSE = SCE ; SSR = SCR

#PRUEBA T

sb1=s_1/math.sqrt(np.sum((X-np.mean(X))**2))
sb1=round(sb1,9)
print "sb1:", sb1
print
# B1=0  t=b1/sb1  b1=x[1]

t=x[1]/sb1
t=round(t,9)
print "t:", t
print

# Para este valor de t y un valor grado de libertad 18,
# nos da un valor de 0.00000000 del valor de la funcion 
# de densidad de probabilidad en la distribucion t.

print
print "Queremos estimar en la region Tacna donde la pobreza en 2015 es 10.6"
print "--------------------------------------------------------------------"
print

print ("Estimado del intervalo de confianza del valor medio de y:")
yp=x[0]+x[1]*10.6
yp=round(yp,9)
print "yp de 10.6:", yp
print

print ("Estimando la desviacion estandar para yp:")
syp=s_1*(math.sqrt(1/(len(X)-2)+ ((10.6-np.mean(X))**2)/np.sum((X-np.mean(X))**2) ))
syp=round(syp,9)
print "Syp: ", syp
print

print ("t_0.025 para 18 es 2.1009")
t_a=2.1009
print

# Confianza 95%
# Estimado del intervalo de confianza
print("Estimado del intervalo de confianza para x=10.6:")
print "10.6 ", syp*t_a
print

# Estimado del intervalo de prediccion para un valor particular de y
yp=x[0]+x[1]*10.6
yp=round(yp,9)
print "yp de 10.6:", yp
print

# Estimado de la varianza de un valor individual
sind=s_1*(math.sqrt(1+1/(len(X)-2)+ ((10.6-np.mean(X))**2)/np.sum((X-np.mean(X))**2) ) )
sind=round(sind,9)
print "Sind:", sind
print

print("Estimado del intervalo de prediccion para x=10.6:")
print "10.6 +-", sind*t_a
print

#Estimacion de los parametros del modelo de regresion lineal

#beta_0= 0.114103064   beta_1=0.9608952
#beta 0 estimado
b0_i=2.1009*s_1*(math.sqrt(1/(len(X)-2))+(np.mean(X)**2/np.sum((X-np.mean(X))**2)))
b0_i=round(b0_i,9)
print "Intervalo beta 0 estimado:", x[0], "+-", b0_i

#beta 1  estimado
b1_i=2.1009*s_1/math.sqrt(np.sum((X-np.mean(X))**2))
b1_i=round(b1_i,9)
print "Intervalo beta 1 estimado:", x[1], "+-", b1_i


#plot
xx=np.arange(60)
yyy=x[0]+xx*x[1]

plt.plot(yyy,color='red',label='Ajuste lineal')

plt.scatter(X,Y,color='darkcyan',label='Diagrama de dispersion')
plt.xlabel('Porcentaje de pobreza en 2015',color='blue') 
plt.ylabel('Porcentaje de pobreza en 2016',color='blue')
plt.legend() 
plt.show()

