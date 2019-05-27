import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys

if len(sys.argv) == 2:
   folder = 'results-' + str(sys.argv[1])
else:
   folder = '' 

#plt.rc('text', usetex=True)
S = 14		# size of labels

def plot_A(data, output_file):

    # initial position of point A
    #A_x = 0.6
    #A_y = 0.2

    col_Ax = 1		# number of column with x-coordinate of position of A
    col_Ay = 2 		# number of column with y-coordinate of position of A

    fig = plt.figure()
    plot_Ax = fig.add_subplot(211)
    plot_Ay = fig.add_subplot(212)

    plot_Ax.plot(data[:, 0], data[:, col_Ax])# - A_x)
    plot_Ay.plot(data[:, 0], data[:, col_Ay])# - A_y)

    plot_Ax.set_ylabel('displacement $x$', size=S)
    plot_Ay.set_ylabel('displacement $y$', size=S)
    plot_Ax.set_xlabel('time', size=S)
    plot_Ay.set_xlabel('time', size=S)

    #plot_Ax.set_ylim(range)

    fig.savefig(output_file, bbox_inches='tight')
    fig.clf()


data_file = folder+'/data.csv'
data = np.genfromtxt(data_file, delimiter=';', skip_header=1)
plot_A(data, folder+'/A_position.png')

N = 4*len(data[:, 0])//5
end_data = data[N:, :]
plot_A(end_data, folder+'/end_A_position.png')
