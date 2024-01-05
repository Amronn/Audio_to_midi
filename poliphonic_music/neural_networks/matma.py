import numpy as np

tab = [np.random.randint(0,10000) for i in range(1000+1)]

roz = []
for i in range(len(tab)-1):
    roz.append(abs(tab[i+1]-tab[i]))
roz = np.sort(roz)

liczba = 0
liczby = np.zeros(1000)
roznice = np.zeros(1000)
it = 0
for i in range(1000-1):
    if roz[i]==roz[i+1]:
        liczba += 1
    else:
        liczby[it] = liczba
        roznice[it] = roz[i]
        liczba = 0
        it += 1
m = np.argmax(liczby)
print(liczby[m], roznice[m])
