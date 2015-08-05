import CO2simulation
reload(CO2simulation)
from CO2simulation import CO2simulation
CO2 = CO2simulation('low')
for i in range(45):
  data = CO2.move_and_sense()
  print data[100]