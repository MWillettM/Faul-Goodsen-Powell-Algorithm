import numpy as np

#Functions used for each stage of the algorithm.

def MQ(x,c):
    k = np.sqrt(c**2+np.linalg.norm(x)**2)
    return k

#x is the input, l is the vector of values of lambda, a is the value of alpha, X is the vector of x_i, c is the shape parameter
def interp(x, l, a, X, c):
    radial_shifts = np.array([MQ(np.abs(x - i),c) for i in X])
    l = np.array(l)
    interp = np.dot(l, radial_shifts) + a
    return interp

def smallest_distance(x):
    min_distance = float('inf')
    for i in range(len(x)-1):
        dist = x[i+1][1]-x[i][1]
        if dist < min_distance:
            min_distance = dist
            Beta = i
            Gamma = i+1
    return min_distance, Beta, Gamma

#X is the input vector, x is the value with which to evaluate d(x), d is delta, c is shape parameter - this returns d^k(x)
def search_direction(X,x,d,c):
    values = [MQ(x-i,c) for i in X]
    k = np.dot(values, d)
    return k


def step3(omeg, phi, q, m, n):
    omeg1 = omeg.copy()
    ell = omeg1[m-1]
    loopcount = 0
    if n - m + 1 > q:
        flag = True

        while flag:
            loopcount+=1

            dist2ell = np.zeros(n-m+1)
            for j in range(m, n+1):
                jj = omeg1[j-1]
                dist2ell[j-m] = phi[ell - 1, jj-1]

            Sorted_distances = np.argsort(dist2ell)
            Lset = [omeg1[i+m-1] for i in Sorted_distances[:q]]
            mindist2 = 0.5 * min(dist2ell[1:])
            minbeta = 0
            mingamma = 1
            mindist = np.inf
            
            for beta in range(q):
                jbeta = Lset[beta]

                for gamma in range(beta+1, q):
                    jgamma = Lset[gamma]
                    dist2betagamma = phi[jbeta-1, jgamma-1]
                    mindistnew = min(mindist, dist2betagamma)

                    if mindistnew != mindist:
                        minbeta = jbeta
                        mingamma = jgamma
                        mindist = mindistnew
            if mindist < mindist2:
                dist2gammaell = phi[mingamma - 1, ell -1 ]
                dist2betaell = phi[minbeta -1, ell -1]

                if dist2gammaell < dist2betaell:
                    minbeta, mingamma = mingamma, minbeta
                
                mhat = omeg1.index(minbeta)
                ell = minbeta
                dummy = omeg1[m-1]
                omeg1[m-1] = omeg1[mhat]
                omeg1[mhat] = dummy
            else:
                flag = False
    else:
        Lset = omeg1[m-1:n]
    
    return omeg1, Lset, ell


def step4(lset, x_values,c):
    size = len(lset)
    x_ell = [x_values[i-1] for i in lset]   
    z_matrix = np.ones((size+1, size+1))    
    z_matrix[size][size] = 0      
    dirac_vector = np.zeros(size+1)
    dirac_vector[0] = 1

    for i, point in enumerate(x_ell):
        for j, center in enumerate(x_ell):
            z_matrix[i,j] = MQ(point-center,c)

    zeta = np.linalg.solve(z_matrix,dirac_vector)
    zeta = zeta[:-1]
    zeta[-1] = -sum(zeta[:-1])
    return zeta

def step5(lset, zeta, r):
    n = len(lset)
    taudummy = np.zeros(len(r))
    sum = 0
    for i in range(n):
        sum+= zeta[i]*r[lset[i]-1]
    myuell = sum/zeta[0]

    for j in range(n):
        taudummy[lset[j]-1] = myuell*zeta[j]

    return taudummy


def step8(lambdas, alpha, r, d, delta):
    gamma = np.dot(delta, r)/np.dot(delta, d)
    r1 = r - gamma * d
    err = np.max(np.abs(r1))
    c = 0.5*(np.max(r1) + np.min(r1))
    alpha1 = alpha + c
    r2 = r1 - c*np.ones(len(r))
    lambdas1 = lambdas + gamma*delta

    return lambdas1, alpha1, r2, err


#Different underlying distributions to draw from in the DEMO version.

def points_in_unit_ball(n,d):
    cube = np.random.standard_normal(size=(n, d))
    norms = np.linalg.norm(cube,axis=1)
    surface_sphere = cube/norms[:,np.newaxis]
    scales = np.random.uniform(0,1, size= n)
    points = surface_sphere * (scales[:, np.newaxis])**(1/d)
    return points

def points_in_cube(n,d):
    points = np.random.uniform(0,1,size=(n,d))
    return points

def points_normal(n,d):
    points = np.random.normal(0,1,size=(n,d))
    return points

def points_integer_grid(n,d, s_min, s_max):
    points = np.random.randint(s_min,s_max, size = (n,d))
    return points
