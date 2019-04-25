import numpy as np

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
    print "m",m
    x = np.zeros((n),dtype = 'f')
    for j in range(n-1,-1,-1):
        x[j] = ( b[j] - sum(A[j,k]*x[k] for k in range(j+1,n)))/A[j,j]
    return x
#A = np.array([0, -4, 0, 0, -5, -2],dtype='f').reshape(3,2)
#B = np.array([2, 3, 6 ],dtype = 'f')

A = np.array([1, 43.2, 1, 39.7, 1, 39.5, 1, 39.3, 1, 34.2, 1, 26.1, 1, 25.4, 1, 26.6, 1, 26.8, 1, 23.8, 1, 22.0, 1, 15.5, 1, 10.9], dtype = 'f').reshape(13,2)
B = np.array([40.6, 37.5, 34.7, 33.9, 33.4, 25.1, 23.0, 22.0, 21.6, 19.2, 19.2, 13.1, 10.7], dtype = 'f')
#A = np.array([1, 2, 3, 1, 2, 3, 2, 5, 7],dtype='f').reshape(3,3)
#A = np.array([4, 2, 2, 1, 2, -3, 1, 1, 2, 1, 3, 2, 1, 1, 1, 2],dtype='f').reshape(4,4)
#B = np.array([2, 3, 6, 4],dtype = 'f')
#A = np.array([1, 2, 3, 1, 2, 3, 2, 5, 7],dtype='f').reshape(3,3)
#A = np.array([12, -51, 4, 6, 167, -68, -4, 24, -41],dtype='f').reshape(3,3)
#A = np.array([1, 2, 4, 5, 7, 8],dtype='f').reshape(3 ,2)
print  A
R,b = householder(A,B)
x = resolucion(R,b)

print "R"
print R

print "Q*b"
print b
print "x"
print x
