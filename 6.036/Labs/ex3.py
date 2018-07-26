import numpy as np

fx_1 = lambda x: x
fx_2 = lambda x: x+1
sq_b = []
sq_g = []
xy_pairs = [(1,3),(1,1),(3,2),(3,6)]

for x,y in xy_pairs:
    sq_b.append((y - fx_1(x))**2)
    sq_g.append((y - fx_2(x))**2)
print ('squared_error_blue:', sq_b,'squared_error_green:', sq_g)

#gradient_contribution
thetas_b = []
thetas_g = []
for x,y in xy_pairs:
    thetas_b.append((-2*(y - fx_1(x))*x, -2*(y - fx_1(x))))
    thetas_g.append((-2*(y - fx_2(x))*x, -2*(y - fx_2(x))))
print ('gradient blue:', thetas_b, 'gradient green:', thetas_g)

#3)Ridge regression
#unsure how lambda affects ridge regression. large lambda means smaller theta
    
