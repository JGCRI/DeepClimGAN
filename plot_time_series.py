import csv
import matplotlib.pyplot as plt
legend_lines = ["ocean_high", "ocean_mid", "ocean_low", "land_high", "land_mid", "land_low"]

fname = '0.csv'


with open(fname, 'r') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	#each row - each location on the grid
	fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
	x = [i for i in range(1,33)]
	fig.suptitle("Precipitation Time series")
	for row in csv_reader:
       
		y = [float(r) for r in row]
		y1 = y[:len(y) // 2]
		ax1.set_xlabel("Random generated month (32 days)")
		ax1.plot(x, y1)
		ax1.legend(legend_lines, loc=0)

		y = [float(r) for r in row]
		y2 = y[(len(y) // 2):]
		ax2.set_xlabel("Random generated month (32 days)")
		ax2.plot(x, y2)
		ax2.legend(legend_lines, loc=1)

plt.show()
