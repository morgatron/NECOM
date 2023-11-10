import time
temps = range(70,110, 20)

T = lambda: dev.therm.sensor.T()
for T in temps:
	dev.therm.pid.setpoint(T)
	print(f"current setpoint: {T}")
	for k in range(10):
		print("{:.2f}".format(dev.therm.sensor.T()))
		time.sleep(50)
		